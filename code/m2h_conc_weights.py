"""BASIN5 s4 concentrated book (BASIN5_ORDER s4). New file; m1_ruler and m1_common are not
modified, the book goes THROUGH them.

Per signal row: among the live coins (finite signal, eligible, in the CONC pool) the K highest
get +1, the K lowest -1, the rest 0; coins outside the pool are NaN. That ternary signal is fed to
the ruler unchanged (R.score_window -> C.rank_weights): average ranks put every +1 at centred
+(n-K)/2 and every -1 at -(n-K)/2, the zeros at 0, so the weights are exactly +-1/(2K):
equal weight, sum w = 0, sum |w| = 1. A row with fewer than 2K live pool coins holds nothing
(all NaN). Hold policy, band, costs and funding are the ruler's (H168 as trained).

Ties at the K boundary are broken by coin index (stable sort), deterministic.

    pool_cols(syms, path)          -> (A,) bool, coin in CONC_POOL.json
    conc_signal(sigT, elT, pool, K) -> (E, A) ternary signal
    score_conc(...)                -> per-tile {cost: log NAV} + parts, like D.score_dense
    held_conc(sigT, elT, pool, K, hold, policy_mod) -> the held weight path (for OKX settlement)
"""
from __future__ import annotations

import json
import os

import numpy as np

import m1_common as C
import m1_ruler as R
import m2h_common as M

CONC_POOL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CONC_POOL.json")
K_DEFAULT = 10
K_SENS = (5, 20)


def pool_cols(syms, path=CONC_POOL):
    pool = json.load(open(path, encoding="utf-8"))["pool"]
    return np.asarray([s in pool for s in syms], dtype=bool)


def conc_signal(sigT, elT, pool, K=K_DEFAULT):
    """sigT, elT (E, A); pool (A,) bool -> ternary (E, A) float, NaN outside the book."""
    sigT = np.asarray(sigT, dtype=np.float64)
    live = np.isfinite(sigT) & np.asarray(elT, dtype=bool) & np.asarray(pool, dtype=bool)[None, :]
    out = np.where(pool[None, :], 0.0, np.nan)
    out = np.where(live, out, np.nan)
    n = live.sum(axis=1)
    big = np.where(live, sigT, np.inf)
    order = np.argsort(big, axis=1, kind="stable")          # live ascending, dead at the end
    rows = np.arange(sigT.shape[0])
    ok = n >= 2 * K
    for t in rows[ok]:
        o = order[t, :n[t]]
        out[t, o[:K]] = -1.0
        out[t, o[n[t] - K:]] = 1.0
    out[~ok] = np.nan
    return out


def score_conc(vm, body, slices, starts, univ, shard, policy_mod, pool, *, K=K_DEFAULT,
               hold=168, costs=M.COSTS, band=R.BAND, nav_usd=R.NAV_USD, want_sig=False):
    """D.score_dense's loop (window_signal + coverage probe on the FULL pool, as trained) with the
    ternary book. Raises M.ZeroCoverage like it. -> dict(logs={cost: [per tile]}, parts, sig)."""
    E = M.EPISODE
    logs = {c: [] for c in costs}
    parts, sigs = [], []
    for s0, sl in zip(starts, slices):
        sigT = M.window_signal(vm, body, sl, s0, univ)
        el = shard["eligible"][:, s0:s0 + E].T
        live = np.isfinite(sigT) & el
        probe_t = [t for t in [M.LAG] + list(range(E // 4, E, E // 4)) if t < E]
        pf = [float(live[t].sum() / max(el[t].sum(), 1)) for t in probe_t]
        if min(pf, default=0.0) < M.MIN_COVERAGE:
            raise M.ZeroCoverage(f"ZERO_COVERAGE_MIN_{min(pf, default=0.0):.2f}")
        cs = conc_signal(sigT, el, pool, K)
        sc = R.score_window(cs, shard, s0, E, [hold], policy_mod, costs=costs, variants=("win",),
                            band=band, nav_usd=nav_usd)
        lg, p = sc[hold]["win"]
        for c in costs:
            logs[c].append(lg[c] if np.isfinite(lg[c]) else M.REWARD_FLOOR)
        parts.append(p)
        if want_sig:
            sigs.append(sigT)
    return dict(logs=logs, parts=parts, sig=sigs)


def held_conc(sigT, elT, pool, K, hold, policy_mod, band=R.BAND):
    """The held path the ruler builds for the ternary book (score_window's own two calls)."""
    return R.held_path(C.rank_weights(conc_signal(sigT, elT, pool, K), elT), hold, band, policy_mod)


def selftest(seed=0):
    """Weights of the ternary book through C.rank_weights: +-1/(2K), sum 0, sum|.| 1, pool only,
    no book under 2K live coins."""
    rng = np.random.default_rng(seed)
    E, A = 50, 60
    sig = rng.normal(size=(E, A))
    sig[rng.random((E, A)) < 0.1] = np.nan
    el = rng.random((E, A)) > 0.05
    pool = rng.random(A) > 0.3
    res = {}
    for K in (5, 10, 20):
        w = C.rank_weights(conc_signal(sig, el, pool, K), el)
        live_n = (np.isfinite(sig) & el & pool[None, :]).sum(1)
        good = live_n >= 2 * K
        nz = np.abs(w) > 0
        res[K] = bool(np.allclose(w[good].sum(1), 0) and np.allclose(np.abs(w[good]).sum(1), 1)
                      and np.allclose(np.abs(w[nz]), 1.0 / (2 * K))
                      and (nz[good].sum(1) == 2 * K).all() and not nz[:, ~pool].any()
                      and not nz[~good].any()
                      # longs are the K highest live pool signals
                      and all(set(np.flatnonzero(w[t] > 0)) == set(
                          np.flatnonzero(np.isfinite(sig[t]) & el[t] & pool)[
                              np.argsort(sig[t][np.isfinite(sig[t]) & el[t] & pool], kind="stable")[-K:]])
                          for t in np.flatnonzero(good)))
    return res
