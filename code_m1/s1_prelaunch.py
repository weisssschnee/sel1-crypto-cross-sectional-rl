"""Pre-launch checks for the budget-capped stage-1 run. Read-only: no training, no OOS.

PRE_PART=ema   EMA kernel cache 64 vs 1: same formula, same slices -> bit-equal?
PRE_PART=beta  anti-beta self-check on the REAL rank book (C.rank_weights + held_path + settle)
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'D:/riskpg_stateful_v5s_governor_nextgen_beem/src')
sys.path.insert(0, r'D:/riskpg_stateful_v5s_governor_nextgen_beem/runtime')

import m1_common as C
import m1_ruler as R
import m2_trainer as T
from oos_guard import guard_shard_dir

PART = os.environ.get('PRE_PART', 'ema')
OUT = os.environ.get('PRE_OUT', f'D:/ChengboRemote/runtime/m1_stage1/prelaunch_{PART}.json')
DEV_DIR = r'D:/ChengboRemote/data/m1_shards_dev'
TRAIN_DIR = r'D:/ChengboRemote/data/m1_shards_v2'
HOLDS_CHECK = (60, 1440)
K_RANDOM = int(os.environ.get('PRE_K', '6'))


def part_ema():
    import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vm_mod
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import (
        XsTsCsStackVM, OPERATOR_NAMES, CandidateLocalInvalid)
    from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod
    vm_mod.BLOCK_ELEMS = 1 << 25
    dev = torch.device('cuda')
    vm = XsTsCsStackVM(len(C.TERMINALS), windows=C.WINDOWS)
    ema_names = [n for n in OPERATOR_NAMES if 'EMA' in n]
    print('EMA operators:', ema_names, flush=True)
    ema = vm.operator_base + OPERATOR_NAMES.index(ema_names[0])
    sub = vm.operator_base + OPERATOR_NAMES.index('SUB')
    w1, w2 = (vm.window_base + C.WINDOWS.index(w) for w in (C.WINDOWS[1], C.WINDOWS[-1]))
    t1, t2 = C.TERMINALS.index('close_to_open_bps'), C.TERMINALS.index('quantity')
    # two EMAs at different windows -> two distinct kernel keys, so cap=1 must evict between them
    body = [t1, w1, ema, t2, w2, ema, sub, vm.end_token]
    expr = vm.describe(body, list(C.TERMINALS))
    print('formula:', expr, flush=True)
    assert 'EMA' in expr

    shard = R.prepare_shard(T.load_shard('2025-07', TRAIN_DIR))
    starts = R.tile_starts(shard['n_min'])[:2]
    slices = T.gpu_episode_slices(shard, starts, dev)

    def run(cap):
        vm_mod.EMA_KERNEL_CACHE_MAX = cap
        vm_mod._EMA_KERNEL_CACHE.clear()
        sigs = []
        for _ in range(2):                        # 2nd pass: cap=64 hits the cache, cap=1 rebuilds
            for sl in slices:
                sigs.append(vm.execute(list(body), sl).detach().cpu().numpy())
        n_cached = len(vm_mod._EMA_KERNEL_CACHE)
        res = R.eval_formula(vm, list(body), list(HOLDS_CHECK), slices, shard, starts, policy_mod,
                             CandidateLocalInvalid, costs=('flat6',), variants=('win',), ic=False)
        return sigs, n_cached, res

    s64, n64, r64 = run(64)
    s1, n1, r1 = run(1)
    out = dict(formula=expr, slice_shape=list(slices[0].shape), cached_kernels_cap64=n64, cached_kernels_cap1=n1)
    assert n64 == 2 and n1 == 1, f'eviction path not exercised: cap64 holds {n64}, cap1 holds {n1}'
    bit_equal, max_abs = True, 0.0
    for a, b in zip(s64, s1):
        eq = np.array_equal(a, b, equal_nan=True)
        bit_equal &= eq
        if not eq:
            max_abs = max(max_abs, float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
    rep_equal = all(np.array_equal(s64[i], s64[i + len(slices)], equal_nan=True) for i in range(len(slices)))
    rep_equal_1 = all(np.array_equal(s1[i], s1[i + len(slices)], equal_nan=True) for i in range(len(slices)))
    rew_equal = all(r64['per_hold'][h]['win']['flat6'] == r1['per_hold'][h]['win']['flat6'] for h in HOLDS_CHECK)
    out.update(signal_bit_equal=bool(bit_equal), signal_max_abs_diff=max_abs,
               repeat_pass_equal_cap64=bool(rep_equal), repeat_pass_equal_cap1=bool(rep_equal_1),
               reward_bit_equal=bool(rew_equal),
               rewards_cap64={h: r64['per_hold'][h]['win']['flat6'] for h in HOLDS_CHECK},
               rewards_cap1={h: r1['per_hold'][h]['win']['flat6'] for h in HOLDS_CHECK},
               n_finite=int(np.isfinite(s1[0]).sum()), n_total=int(s1[0].size))
    print(json.dumps(out, indent=1), flush=True)
    return out


def tie_rank_weights(sig, eligible, min_live=10):
    """Prototype: C.rank_weights with TIED values given the AVERAGE rank (stable argsort gives
    ties distinct ranks by asset index). Identical to C.rank_weights when there are no ties."""
    live = np.isfinite(sig) & eligible
    n = live.sum(axis=1)
    big = np.where(live, sig, np.inf)
    order = np.argsort(big, axis=1, kind='stable')
    sv = np.take_along_axis(big, order, axis=1)
    t_, a_ = sig.shape
    pos = np.broadcast_to(np.arange(a_), (t_, a_))
    newg = np.ones((t_, a_), bool)
    newg[:, 1:] = sv[:, 1:] != sv[:, :-1]
    start = np.maximum.accumulate(np.where(newg, pos, 0), axis=1)
    last = np.ones((t_, a_), bool)
    last[:, :-1] = newg[:, 1:]
    end = np.minimum.accumulate(np.where(last, pos, a_)[:, ::-1], axis=1)[:, ::-1]
    ranks = np.empty_like(sig, dtype=np.float64)
    rows = np.arange(t_)[:, None]
    ranks[rows, order] = (start + end) / 2.0
    ranks = np.where(live, ranks, 0.0)
    ok = n >= min_live
    cent = ranks - np.maximum(n - 1, 0)[:, None] / 2.0
    denom = np.abs(np.where(live, cent, 0.0)).sum(axis=1)
    safe = ok & (denom > 0)
    w = np.where(live, cent, 0.0) / np.where(safe, denom, 1.0)[:, None]
    return np.where(safe[:, None], w, 0.0)


def book(sigT, shard, s0, E, hold, policy_mod, weights_fn):
    el = shard['eligible'][:, s0:s0 + E].T
    fim = shard['fi'][:, s0:s0 + E].T
    frt = shard['fr'][:, s0:s0 + E].T
    fw = shard['fwd_w'][:, s0:s0 + E].T
    desired = weights_fn(sigT, el)
    held = R.held_path(desired, hold, R.BAND, policy_mod)
    specs = {'flat6': R.cost_specs()['flat6'], 'cost0': R.cost_specs()['cost0']}
    res, parts = R.settle_multi(held, fw, fim, frt, specs)
    days = E / 1440.0
    return dict(net=R.bp_per_day(res['flat6']['log_nav']), cost0=R.bp_per_day(res['cost0']['log_nav']),
                gross=1e4 * parts['gross'] / days, turn_day=parts['turnover'] / days,
                net_exposure=float(held.sum(axis=1).mean()), gross_exposure=float(np.abs(held).sum(axis=1).mean()),
                max_abs_sum_desired=float(np.abs(desired.sum(axis=1)).max())), desired


def part_beta():
    from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod
    months = [(f'2026-0{k}', DEV_DIR, 'DEV') for k in range(1, 7)] + [('2025-07', TRAIN_DIR, 'TRAIN')]
    want = os.environ.get('PRE_MONTHS')
    if want:
        months = [m for m in months if m[0] in want.split(',')]
    rng = np.random.default_rng(7)
    E, lag = R.EPISODE_MINUTES, C.EXECUTION_LAG_MIN
    out = {}
    for month, sdir, role in months:
        guard_shard_dir(sdir, role, expect_months=None)
        shard = R.prepare_shard(T.load_shard(month, sdir))
        starts = R.tile_starts(shard['n_min'])
        A = len(shard['symbols'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            mkt = float(np.nansum(np.nanmean(shard['fwd_w'], axis=0)))
        rec = dict(role=role, windows=len(starts), n_assets=A, symbols_first=shard['symbols'][:3],
                   symbols_last=shard['symbols'][-3:],
                   market_ew_bp_day=1e4 * mkt / (shard['n_min'] / 1440.0))

        def mk(kind):
            s = {'const': lambda: np.zeros((E, A)), 'ones': lambda: np.ones((E, A)),
                 'signs': lambda: np.sign(rng.normal(size=(E, A))),
                 'random': lambda: rng.normal(size=(E, A))}[kind]()
            s[:lag] = np.nan
            return s

        for kind, reps in (('const', 1), ('ones', 1), ('signs', 2), ('random', K_RANDOM)):
            for hold in HOLDS_CHECK:
                rows = []
                for r in range(reps):
                    for s0 in starts:
                        sig = mk(kind)
                        b, desired = book(sig, shard, s0, E, hold, policy_mod, C.rank_weights)
                        rows.append(b)
                        if kind in ('const', 'ones', 'signs') and r == 0 and s0 == starts[0]:
                            tied = tie_rank_weights(sig, shard['eligible'][:, s0:s0 + E].T)
                            rec.setdefault(f'{kind}_tiefix', dict(
                                max_abs_weight=float(np.abs(tied).max()),
                                current_max_abs_weight=float(np.abs(desired).max()),
                                gross_weight_current=float(np.abs(desired).sum(axis=1).max())))
                        if kind == 'random' and r == 0 and s0 == starts[0]:
                            sig2 = sig
                            same = np.array_equal(desired, tie_rank_weights(sig2, shard['eligible'][:, s0:s0 + E].T))
                            rec['tiefix_identical_without_ties'] = bool(same)
                agg = {k: float(np.mean([x[k] for x in rows])) for k in rows[0]}
                if kind == 'random':
                    g = np.array([x['gross'] for x in rows])
                    agg['gross_se'] = float(g.std(ddof=1) / np.sqrt(len(g)))
                    agg['n_draws'] = len(g)
                rec[f'{kind}_h{hold}'] = agg
        out[month] = rec
        c = rec['const_h1440']
        print(f"{month} {role} mkt_ew {rec['market_ew_bp_day']:+8.1f} | const h1440: gross {c['gross']:+8.2f} "
              f"net {c['net']:+8.2f} turn {c['turn_day']:.2f}/d netexp {c['net_exposure']:+.4f} grossexp "
              f"{c['gross_exposure']:.3f} | random h1440 gross {rec['random_h1440']['gross']:+6.2f} "
              f"se {rec['random_h1440']['gross_se']:.2f}", flush=True)
    json.dump(out, open(OUT, 'w'), indent=1, default=str)
    return out


if __name__ == '__main__':
    res = part_ema() if PART == 'ema' else part_beta()
    if PART == 'ema':
        json.dump(res, open(OUT, 'w'), indent=1, default=str)
    print('wrote', OUT)
