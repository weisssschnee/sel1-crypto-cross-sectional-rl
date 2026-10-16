"""Score trained candidates on DEV with the SAME ruler the trainer optimises.

There is one scoring function (m1_ruler.eval_formula). This file only tiles the
DEV months into back-to-back windows, runs each formula at its OWN hold, and
puts the handwritten N=1 baseline through the identical code at every hold.

  statistic   net bp/day = 1e4 * mean(window net log NAV) / (episode / 1440),
              lag-2 rank book, 1/99 winsorised returns ('win'; 'raw' is shown
              beside it), Rule 0 hold policy, raw-ms funding, chosen cost model
  SR_0        Bailey & Lopez de Prado hurdle at the run's N and reward variance
              V. V is the variance of the trainer's reward = log NAV per
              window, so SR_0 is in log NAV per window and is converted with the
              same factor as the statistic. Nothing is compared across units.
  baseline    CS_RANK(NEG(close_to_open_bps)) at every hold; a candidate is
              compared with the baseline at ITS hold
  verdict     reported for two selections, never mixed: the candidate with the
              best TRAIN reward (no DEV selection) and the best of the top-K on
              DEV (selected on DEV, so K-fold optimistic; labelled as such)

The old lag-0 rank-IC statistic is gone: at lag 2 it was ~0 while dollars were
negative, i.e. it measured bid-ask bounce, not tradable edge.
DEV shards are 2026-01..06; oos_guard runs before anything is read.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'D:/riskpg_stateful_v5s_governor_nextgen_beem/src')
sys.path.insert(0, r'D:/riskpg_stateful_v5s_governor_nextgen_beem/runtime')

import m1_common as C
import m1_ruler as R
import m2_trainer as T
from oos_guard import guard_shard_dir, OOSViolation
from search_accounting import SearchAccounting, format_report

ART = r'D:/riskpg_stateful_v5s_governor_nextgen_beem/artifacts/m2_v1'
VARIANTS = ('win', 'raw')


def baseline_body(vm):
    """CS_RANK(NEG(close_to_open_bps)) -- the N=1 handwritten control.

    Token indices come from the VM's own layout (operator_base / end_token)
    rather than arithmetic on the operator list."""
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import OPERATOR_NAMES
    t = C.TERMINALS.index('close_to_open_bps')
    neg = vm.operator_base + OPERATOR_NAMES.index('NEG')
    cs = vm.operator_base + OPERATOR_NAMES.index('CS_RANK')
    return (t, neg, cs, vm.end_token)


def load_month(month, shard_dir, device, cost_models, depth_dir, winsor_pct):
    shard = R.prepare_shard(T.load_shard(month, shard_dir), winsor_pct,
                            depth_dir=depth_dir if 'depth' in cost_models else None)
    starts = R.tile_starts(shard['n_min'])
    return dict(month=month, shard=shard, starts=starts,
                slices=T.gpu_episode_slices(shard, starts, device))


def score_formula(vm, body, holds, months, *, costs, band, nav_usd, invalid_cls, policy_mod):
    """{hold: {variant: {cost: stats}, turnover_day, gross_bp_day, funding_bp_day}}.

    Raises invalid_cls if any month has a zero-coverage window."""
    per = {h: {v: {c: [] for c in costs} for v in VARIANTS} for h in holds}
    mon = {h: {v: {c: {} for c in costs} for v in VARIANTS} for h in holds}
    parts = {h: [] for h in holds}
    live = []
    for m in months:
        res = R.eval_formula(vm, list(body), holds, m['slices'], m['shard'], m['starts'],
                             policy_mod, invalid_cls, costs=costs, variants=VARIANTS,
                             band=band, nav_usd=nav_usd, ic=False)
        live.append(res['live_frac'])
        for h in holds:
            parts[h] += res['parts'][h]['win']
            for v in VARIANTS:
                for c in costs:
                    arr = res['per_hold'][h][v][c]
                    per[h][v][c] += arr
                    mon[h][v][c][m['month']] = R.bp_per_day(float(np.mean(arr)))
    days = R.EPISODE_MINUTES / 1440.0
    out = {}
    for h in holds:
        o = {}
        for v in VARIANTS:
            o[v] = {c: dict(mean_lognav=float(np.mean(per[h][v][c])),
                            bp_day=R.bp_per_day(float(np.mean(per[h][v][c]))),
                            windows=len(per[h][v][c]),
                            pos_windows=int(sum(x > 0 for x in per[h][v][c])),
                            per_month_bp_day=mon[h][v][c]) for c in costs}
        o['turnover_day'] = float(np.mean([p['turnover'] for p in parts[h]])) / days
        o['gross_bp_day'] = 1e4 * float(np.mean([p['gross'] for p in parts[h]])) / days
        o['funding_bp_day'] = 1e4 * float(np.mean([p['funding'] for p in parts[h]])) / days
        o['live_frac'] = float(np.mean(live))
        out[h] = o
    return out


def load_candidates(run_dir, top):
    """Best `top` distinct (body, hold) by the trainer's holdout reward."""
    path = os.path.join(run_dir, 'FORMULAS.jsonl')
    rows, seen = [], set()
    if not os.path.exists(path):
        return rows
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if line:
            j = json.loads(line)
            r = j.get('reward')
            if j.get('invalid_reason') is None and r is not None and np.isfinite(r):
                rows.append((float(r), tuple(j['body']), int(j['hold_min']), j.get('expr', '')))
    rows.sort(key=lambda x: -x[0])
    out = []
    for r, body, hold, expr in rows:
        if (body, hold) not in seen:
            seen.add((body, hold))
            out.append((r, body, hold, expr))
        if len(out) >= top:
            break
    return out

def hurdle_lognav(acc, variance):
    """SR_0 in log NAV per window (the trainer's reward unit), or None without a usable V."""
    if variance is None or not np.isfinite(variance) or variance <= 0:
        return None
    return float(acc.hurdle(variance=variance)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='training run name under artifacts/m2_v1')
    ap.add_argument('--dev-shard-dir', default='D:/ChengboRemote/data/m1_shards_dev')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--top', type=int, default=20, help='trained candidates to score (by trainer holdout reward)')
    ap.add_argument('--months', default=None, help='comma list to restrict DEV months (smoke); default all')
    ap.add_argument('--cost', default='flat6', choices=R.COSTS, help='cost model of the verdict statistic')
    ap.add_argument('--band', type=float, default=R.BAND)
    ap.add_argument('--nav-usd', type=float, default=R.NAV_USD)
    ap.add_argument('--depth-dir', default=None, help='depth_{month}.npz dir; adds the depth cost column')
    ap.add_argument('--winsor-pct', type=float, default=R.WINSOR_PCT)
    ap.add_argument('--vm-block-elems', type=int, default=1 << 27)
    ap.add_argument('--baseline-only', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    run_dir = os.path.join(ART, args.run)
    dev_months = sorted(os.path.basename(p)[len('m1_shard_'):-len('.npz')]
                        for p in glob.glob(os.path.join(args.dev_shard_dir, 'm1_shard_*.npz')))
    if not dev_months:
        raise SystemExit(f'no DEV shards in {args.dev_shard_dir}')
    try:
        guard_shard_dir(args.dev_shard_dir, 'DEV', expect_months=dev_months)
    except OOSViolation as exc:
        raise SystemExit(f'OOS GUARD REFUSED TO SCORE: {exc}')
    if args.months:
        want = args.months.split(',')
        missing = [m for m in want if m not in dev_months]
        if missing:
            raise SystemExit(f'months not in DEV shards: {missing}')
        dev_months = want

    costs = ['flat6', 'maker2'] + (['depth'] if args.depth_dir else [])
    if args.cost not in costs:
        raise SystemExit(f'--cost {args.cost} needs its inputs (depth needs --depth-dir)')

    import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vm_mod
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import XsTsCsStackVM, CandidateLocalInvalid
    from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod
    vm_mod.BLOCK_ELEMS = int(args.vm_block_elems)
    device = torch.device(args.device)
    vm = XsTsCsStackVM(len(C.TERMINALS), windows=C.WINDOWS)

    receipt = {}
    rp = os.path.join(run_dir, 'TRAIN_RECEIPT_V1.json')
    if os.path.exists(rp):
        receipt = json.load(open(rp, encoding='utf-8'))
    acc_d = receipt.get('search_accounting') or {}
    n_eval = int(acc_d.get('formulas_evaluated') or 0)
    V = acc_d.get('reward_variance_V')   # pooled over scored formulas: E[r^2] - E[r]^2 of the holdout reward
    if n_eval == 0:
        print('WARNING: no training receipt / search accounting; SR_0 cannot be computed')

    months = [load_month(m, args.dev_shard_dir, device, costs, args.depth_dir, args.winsor_pct)
              for m in dev_months]
    n_win = sum(len(m['starts']) for m in months)
    kw = dict(costs=tuple(costs), band=args.band, nav_usd=args.nav_usd,
              invalid_cls=CandidateLocalInvalid, policy_mod=policy_mod)
    print(f'DEV months {dev_months}  windows {n_win} x {R.EPISODE_MINUTES} min  cost {args.cost}  '
          f'band {args.band}  N {n_eval}  V {V}', flush=True)

    bl_body = baseline_body(vm)
    bl = score_formula(vm, bl_body, list(C.HOLDS), months, **kw)
    print('=' * 100)
    print('N=1 BASELINE  CS_RANK(NEG(close_to_open_bps))  net bp/day by HOLD '
          f'({args.cost}; win = 1/99 winsorised, raw = unclipped)')
    print('=' * 100)
    print(f'  {"hold":>6} {"win":>9} {"raw":>9} {"turn/d":>8} {"gross":>8} {"fund":>7}   '
          + '  '.join(f'{c:>8}' for c in costs))
    for h in C.HOLDS:
        b = bl[h]
        print(f'  {h:>6} {b["win"][args.cost]["bp_day"]:>+9.1f} {b["raw"][args.cost]["bp_day"]:>+9.1f} '
              f'{b["turnover_day"]:>8.2f} {b["gross_bp_day"]:>+8.1f} {b["funding_bp_day"]:>+7.1f}   '
              + '  '.join(f'{b["win"][c]["bp_day"]:>+8.1f}' for c in costs))

    rows = []
    if not args.baseline_only:
        cands = load_candidates(run_dir, args.top)
        print()
        print('=' * 100)
        print(f'TRAINED CANDIDATES (top {len(cands)} by trainer holdout reward), each at its own hold, '
              f'{args.cost}, net bp/day on DEV')
        print('=' * 100)
        for i, (r, body, hold, expr) in enumerate(cands):
            try:
                sc = score_formula(vm, body, [hold], months, **kw)[hold]
            except CandidateLocalInvalid as exc:
                print(f'  [{i+1:>2}] INVALID on DEV ({exc})  h={hold:<5} {expr[:56]}')
                rows.append(dict(train_reward=r, hold=hold, body=list(body), expr=expr[:160],
                                 invalid=str(exc)))
                continue
            w = sc['win'][args.cost]
            rows.append(dict(train_reward=r, hold=hold, body=list(body), expr=expr[:160], dev=sc,
                             baseline_same_hold_lognav=bl[hold]['win'][args.cost]['mean_lognav']))
            print(f'  [{i+1:>2}] win {w["bp_day"]:>+8.1f}  raw {sc["raw"][args.cost]["bp_day"]:>+8.1f}  '
                  f'base@h {bl[hold]["win"][args.cost]["bp_day"]:>+8.1f}  turn/d {sc["turnover_day"]:>7.2f}  '
                  f'pos {w["pos_windows"]}/{w["windows"]}  train_r {r:+.5f}  h={hold:<5} {expr[:44]}')

    scored = [x for x in rows if 'dev' in x]

    def stat_of(x):
        return x['dev']['win'][args.cost]['mean_lognav']

    sels = {}
    if scored:
        sels['best_by_train_reward (no DEV selection)'] = scored[0]
        sels[f'best_of_top{len(scored)}_on_DEV (selected on DEV, optimistic)'] = max(scored, key=stat_of)

    sa = SearchAccounting(updates=int(receipt.get('config', {}).get('updates', 0)),
                          batch=int(receipt.get('config', {}).get('batch', 0)),
                          formulas_evaluated=n_eval, reward_variance_V=V,
                          notes='from training receipt')
    hurdles = {'pooled_trial_variance': hurdle_lognav(sa, V) if n_eval else None}
    Vs = {'pooled_trial_variance': V}

    verdicts = {}
    print()
    print('=' * 100)
    print('VERDICT  (statistic and SR_0 are both mean log NAV per '
          f'{R.EPISODE_MINUTES}-min window; x{1e4 / (R.EPISODE_MINUTES / 1440.0):.3f} = bp/day; '
          'V = pooled per-formula variance of the trainer holdout reward)')
    print('=' * 100)
    for k, h0 in hurdles.items():
        print(f'  SR_0 [{k:<18}] V={Vs[k]}  ->  '
              + ('n/a' if h0 is None else f'{h0:+.6f} lognav/window = {R.bp_per_day(h0):+.1f} bp/day'))
    for name, x in sels.items():
        s = stat_of(x)
        b = x['baseline_same_hold_lognav']
        v = dict(stat_lognav=s, stat_bp_day=R.bp_per_day(s), hold=x['hold'], expr=x['expr'],
                 baseline_same_hold_lognav=b, beats_baseline=bool(s > b), net_positive=bool(s > 0),
                 clears_sr0={k: (None if h0 is None else bool(s > h0)) for k, h0 in hurdles.items()})
        v['verdict'] = ('PASSES' if v['beats_baseline'] and v['net_positive']
                        and all(c is True for c in v['clears_sr0'].values() if c is not None)
                        and any(c is not None for c in v['clears_sr0'].values()) else 'FAILS_AT_LEAST_ONE')
        verdicts[name] = v
        print(f'  {name}')
        print(f'      {R.bp_per_day(s):+.1f} bp/day at h={x["hold"]}  baseline@h {R.bp_per_day(b):+.1f}  '
              f'beats_baseline={v["beats_baseline"]}  net>0={v["net_positive"]}  clears_SR0={v["clears_sr0"]}  '
              f'-> {v["verdict"]}')

    best_stat = max((stat_of(x) for x in scored), default=None)
    bl_best = max((bl[x['hold']]['win'][args.cost]['mean_lognav'] for x in scored), default=None)
    report = format_report(sa, measured_test_stat=best_stat, baseline_n1=bl_best)
    print()
    print(report)

    out = args.out or os.path.join(run_dir, 'DEV_SCORING.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = dict(run=args.run, dev_months=dev_months, cost=args.cost, band=args.band,
                   nav_usd=args.nav_usd, winsor_pct=args.winsor_pct, episode_minutes=R.EPISODE_MINUTES,
                   windows=n_win, n_formulas_evaluated=n_eval, V=Vs, sr0_lognav_per_window=hurdles,
                   baseline=dict(expr='CS_RANK(NEG(close_to_open_bps))', body=list(bl_body), by_hold=bl),
                   candidates=rows, verdicts=verdicts,
                   unit_note='all statistics are mean log NAV per window or bp/day; no IC anywhere')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=1, default=str)
    if os.path.isdir(run_dir):
        with open(os.path.join(run_dir, 'DEV_SCORING.txt'), 'w', encoding='utf-8') as fh:
            fh.write(report + '\n')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
