"""DEV 6-month report for a stage-1 run. Pure python (no GPU): reads s1_dev_eval_<run>.json.

Pre-registered reading rules (work order 2026-10-08, locked before the run):
  rule 8   only DEV counts; TRAIN is in-sample and never enters this report
  rule 9   a VERDICT is drawn only at hold tiers with many hold cycles per window: hold <= 240 min
           (>= 42 cycles per 7-day window). "edge found" in a tier = candidates that are positive on the
           6-month mean AND positive in >= 5 of 6 months occur clearly more often than the random
           expectation: P(>=5/6 positive months | fair coin) = 7/64 = 0.109, expected count = 0.109 * n_eff
           (n_eff = behaviour-deduped candidates in the tier); "clearly" is fixed here as binomial
           upper-tail P(X >= k_obs) <= ALPHA.
  rule 10  slow holds (1440 / 2880 / 10080; <= 7 cycles per window): NO t, p, cross-month sign count and no
           point values. Printed: sign of the 6-month mean and whether GROSS (cost0) has the same sign in
           all months ("worth watching" - a weak flag, never a decision).
  rule 11  every row carries post-beta behaviour: net exposure, beta to the equal-weight market, gross
           exposure and turnover/day, so an idle book (A) or a market-long book (B) is visible.
  rule 7   candidates are deduped by their DEV per-month net-score vector (flat6 bp/day), not by tokens.
  rule 5   maker2 / depth appear only in the final comparison block, never as a selection or verdict input.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

P_FIVE_OF_SIX = 7.0 / 64.0          # P(>=5 positive of 6 months) for a fair coin
ALPHA = 0.05                        # "clearly above random": binomial upper tail
VERDICT_MAX_HOLD = 240              # verdicts only at hold <= 240 min
BEHAV_TOL_BP = 0.1                  # same behaviour: every month's net bp/day within 0.1 (R11/R12 differ by 0.007)
IDLE_GROSS_EXPOSURE = 0.5           # flag (A): mean gross exposure below this = book mostly empty
MKT_BETA_ABS = 0.3                  # flag (B): |beta to equal-weight market| above this
NET_EXPOSURE_ABS = 0.1              # flag (B): |net dollar exposure| above this in any month


def binom_tail(k, n, p):
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def vec(row, key, months):
    return np.array([row['per_month'][m][key] for m in months], dtype=float)


def analyse(row, months):
    f6, c0 = vec(row, 'flat6', months), vec(row, 'cost0', months)
    nx, bt = vec(row, 'net_exposure', months), np.array([row['per_month'][m]['beta_mkt'] if row['per_month'][m]['beta_mkt'] is not None else np.nan for m in months])
    ge, td = vec(row, 'gross_exposure', months), vec(row, 'turn_day', months)
    flags = []
    if float(np.mean(ge)) < IDLE_GROSS_EXPOSURE:
        flags.append('A:idle')
    if float(np.nanmax(np.abs(bt))) > MKT_BETA_ABS or float(np.max(np.abs(nx))) > NET_EXPOSURE_ABS:
        flags.append('B:mkt')
    return dict(f6=f6, c0=c0, mean=float(f6.mean()), npos=int((f6 > 0).sum()),
                gross_mean=float(c0.mean()),
                gross_same_sign=bool((c0 > 0).all() or (c0 < 0).all()),
                gross_sign='+' if c0.mean() > 0 else '-',
                net_exp_max=float(np.max(np.abs(nx))), beta_mean=float(np.nanmean(bt)),
                beta_absmax=float(np.nanmax(np.abs(bt))), gross_exp=float(ge.mean()),
                turn=float(td.mean()), flags=flags)


def dedup(rows, months):
    kept, groups = [], []
    for r in sorted(rows, key=lambda r: -(r['train_reward'] if r['train_reward'] is not None else -1e9)):
        v = vec(r, 'flat6', months)
        for g in groups:
            if float(np.abs(g['v'] - v).max()) <= BEHAV_TOL_BP:
                g['members'].append(r['name'])
                break
        else:
            groups.append(dict(v=v, rep=r, members=[r['name']]))
            kept.append(r)
    return kept, [g['members'] for g in groups if len(g['members']) > 1]


def main(path, out_txt=None, receipt_path=None):
    d = json.load(open(path))
    months = [m for m in d['months']]
    rows = [r for r in d['rows'] if not r['invalid'] and all(m in r['per_month'] for m in months)]
    bad = [r['name'] for r in d['rows'] if r['invalid']]
    base = [r for r in rows if r['kind'] == 'BASELINE']
    cands = [r for r in rows if r['kind'] == 'CAND']
    uniq, collapsed = dedup(cands, months)
    L = []
    P = L.append
    if len(months) != 6:
        P(f"WARNING: {len(months)} months scored; the verdict rules (>=5 of 6 positive, p=7/64) are defined for the "
          "6 DEV months 2026-01..06 - this output is a plumbing check, not a verdict.")
    P(f"DEV report  run={d['run']}  months={months}  windows/month={[d['windows'][m] for m in months]} "
      f"(total {sum(d['windows'][m] for m in months)})  episode={d['episode']} min")
    P(f"candidates scored {len(cands)} (top-{d['topk']} by TRAIN holdout reward), invalid {len(bad)} {bad[:5]}; "
      f"behaviour-deduped n_eff = {len(uniq)} (tol {BEHAV_TOL_BP} bp/day on every month)")
    for g in collapsed:
        P(f"  collapsed same-behaviour group: {g}")
    if receipt_path:
        try:
            rc = json.load(open(receipt_path))
            sa = rc.get('search_accounting', {})
            P(f"run N (formulas scored) = {sa.get('formulas_evaluated')}; within-update distinct behaviours "
              f"(upper bound) = {rc.get('behaviour_dedup', {}).get('n_behaviours_cum_upper_bound')}")
        except OSError:
            P(f"(receipt {receipt_path} not readable)")
    P('')
    P('units: flat6 net bp/day = training cost; cost0 = gross; netexp = max|mean net dollar exposure| over months; '
      'beta = mean beta to the equal-weight market; gE = mean gross exposure; trn = turnover/day. '
      'Flags are descriptive (A:idle gE<0.5, B:mkt |beta|>0.3 or |netexp|>0.1), not gates.')

    def hdr_fast():
        return (f"{'name':<9}{'hold':>5} {'expr':<44}" + ''.join(f'{m[5:]:>8}' for m in months)
                + f"{'mean':>8}{'#pos':>6}{'gross':>8}{'netexp':>8}{'beta':>7}{'gE':>6}{'trn':>8}  flags")

    def line_fast(r, a):
        return (f"{r['name']:<9}{r['hold']:>5} {r['expr'][:44]:<44}" + ''.join(f'{x:>+8.1f}' for x in a['f6'])
                + f"{a['mean']:>+8.1f}{a['npos']:>3}/{len(months)}{a['gross_mean']:>+8.1f}{a['net_exp_max']:>8.4f}"
                  f"{a['beta_mean']:>+7.2f}{a['gross_exp']:>6.2f}{a['turn']:>8.2f}  {','.join(a['flags'])}")

    # ---- verdict tiers (hold <= 240): per-hold table + one tier verdict ----------------------------
    P('=' * 30 + ' VERDICT TIERS (hold <= 240 min; rules 9 + 11) ' + '=' * 30)
    fast_c = [r for r in uniq if r['hold'] <= VERDICT_MAX_HOLD]
    P(hdr_fast())
    for r in sorted(base, key=lambda r: r['hold']):
        if r['hold'] <= VERDICT_MAX_HOLD:
            P(line_fast(r, analyse(r, months)))
    P('-' * 8)
    ana = {}
    for r in sorted(fast_c, key=lambda r: (r['hold'], r['name'])):
        ana[r['name']] = analyse(r, months)
        P(line_fast(r, ana[r['name']]))
    P('')
    raw_pass = [r['name'] for r in fast_c if ana[r['name']]['mean'] > 0 and ana[r['name']]['npos'] >= 5]
    flagged_pass = [x for x in raw_pass if ana[x]['flags']]
    passing = [x for x in raw_pass if not ana[x]['flags']]   # an A/B book is not cross-sectional edge (work order)
    n = len(fast_c)                                          # flagged books stay in n: they were draws too
    k = len(passing)
    exp = P_FIVE_OF_SIX * n
    tail = binom_tail(k, n, P_FIVE_OF_SIX) if n else float('nan')
    P(f"tier hold<=240: n_eff={n}; mean>0 and >=5/6 positive months: {len(raw_pass)} {raw_pass}; "
      f"of these A/B-flagged (excluded from k): {flagged_pass}")
    P(f"  k_obs={k} {passing}; random expectation {exp:.2f} (p={P_FIVE_OF_SIX:.3f} each); "
      f"binomial upper tail P(X>={k}) = {tail:.3f} (threshold {ALPHA})")
    by_hold = {}
    for r in fast_c:
        b = by_hold.setdefault(r['hold'], [0, 0])
        b[0] += 1
        b[1] += int(r['name'] in passing)
    P('  by hold (n_eff / k_obs, descriptive only): '
      + ', '.join(f'h{h}: {b[0]}/{b[1]}' for h, b in sorted(by_hold.items())))
    six = [r['name'] for r in fast_c if ana[r['name']]['mean'] > 0 and ana[r['name']]['npos'] == len(months)
           and not ana[r['name']]['flags']]
    six_g = [x for x in six if ana[x]['gross_same_sign'] and ana[x]['gross_sign'] == '+']
    P(f"  LEADS (descriptive, NOT a verdict): unflagged, all {len(months)} months positive: {six}; "
      f"chance expectation {n / 2 ** len(months):.2f}; of these gross also positive every month: {six_g}")
    if n == 0:
        verdict = 'NO CANDIDATES in this tier - nothing to judge'
    elif k >= 1 and tail <= ALPHA:
        verdict = 'EDGE FOUND in tier (hold<=240) under the correct ruler (unflagged books only)'
    else:
        verdict = 'NO EDGE in tier (hold<=240) under the correct ruler: positive-and-consistent count is not above chance'
    P('VERDICT: ' + verdict)
    P('')

    # ---- slow holds: sign only (rule 10) ---------------------------------------------------------
    P('=' * 30 + ' SLOW HOLDS (1440/2880/10080): NO verdict, NO t/p/sign-count (rule 10) ' + '=' * 14)
    P("shown: sign of the 6-month mean net and of the gross (cost0) mean; 'gross same-sign' = gross has one sign in "
      "ALL months (the only condition under which a slow-hold sign is a weak 'worth watching' flag). No magnitudes.")
    P(f"{'name':<9}{'hold':>5} {'expr':<44}{'net':>5}{'gross':>7}{'g-same':>8}{'netexp':>8}{'beta':>7}{'gE':>6}{'trn':>8}  flags")
    slow_all = [r for r in base if r['hold'] > VERDICT_MAX_HOLD] + [r for r in uniq if r['hold'] > VERDICT_MAX_HOLD]
    watch = []
    for r in sorted(slow_all, key=lambda r: (r['kind'] != 'BASELINE', r['hold'], r['name'])):
        a = analyse(r, months)
        P(f"{r['name']:<9}{r['hold']:>5} {r['expr'][:44]:<44}{('+' if a['mean'] > 0 else '-'):>5}{a['gross_sign']:>7}"
          f"{('YES' if a['gross_same_sign'] else 'no'):>8}{a['net_exp_max']:>8.4f}{a['beta_mean']:>+7.2f}"
          f"{a['gross_exp']:>6.2f}{a['turn']:>8.2f}  {','.join(a['flags'])}")
        if r['kind'] == 'CAND' and a['mean'] > 0 and a['gross_same_sign'] and a['gross_sign'] == '+' and not a['flags']:
            watch.append(r['name'])
    P(f"worth-watching (net sign +, gross + in every month, no A/B flag): {watch if watch else 'none'}  "
      "- a flag for later attention, not an edge claim")
    P('')

    # ---- final comparison block (rule 5): optimistic bounds, sign for slow holds ------------------
    P('=' * 30 + ' FINAL COST COMPARISON (not reward, not a verdict input) ' + '=' * 22)
    P("6-month mean net bp/day under each cost; maker2/depth are optimistic lower bounds. Slow holds: sign only.")
    P(f"{'name':<9}{'hold':>5} {'expr':<44}{'flat6':>9}{'cost0':>9}{'maker2':>9}{'depth':>9}")
    for r in sorted([x for x in base] + uniq, key=lambda r: (r['kind'] != 'BASELINE', r['hold'], r['name'])):
        m = {c: float(vec(r, c, months).mean()) for c in ('flat6', 'cost0', 'maker2', 'depth')}
        if r['hold'] > VERDICT_MAX_HOLD:
            cells = ''.join(f"{('+' if m[c] > 0 else '-'):>9}" for c in ('flat6', 'cost0', 'maker2', 'depth'))
        else:
            cells = ''.join(f'{m[c]:>+9.1f}' for c in ('flat6', 'cost0', 'maker2', 'depth'))
        P(f"{r['name']:<9}{r['hold']:>5} {r['expr'][:44]:<44}{cells}")
    text = '\n'.join(L)
    print(text)
    if out_txt:
        open(out_txt, 'w', encoding='utf-8').write(text + '\n')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None, sys.argv[3] if len(sys.argv) > 3 else None)
