"""TV1 dense-reward layer (TV1_ORDER + TV1_ADDENDUM_A2 + TV1_A2_BASIN4_ORDER, 2026-10-10).

Shared by the A table (m2h_snr_table), m2h_dense_trainer, m2h_plant, the TV1 self-checks and
the per-round report, so all of them score, rank and weight through one code path.

  score_dense   m2h_common.score_formula's loop verbatim (window_signal, coverage probe,
                m1_ruler.score_window) plus the per-tile IC and, on request, the signal.
                TV1 check 2 holds it bit-equal to score_formula.
  ic_tile       TV1 section 2: per hour, CS Spearman of the lagged, pool-masked signal with the
                next 168 h cumulative log return (the ruler's 1/99-winsorised fwd_w; a missing
                hour adds 0, as Rule 0 freezes the position there); only rows whose 168 h lie
                inside the tile; a constant side or < 20 coins counts 0, it is not skipped.
  weights       per-family q95 (select_risk_elite, as basin_risk_update), whole-batch q95
                (stage-1), full-sample within-family z clipped +-3, C1 / C2 (A2).
  level_flag    AGENTS.md pit: SIGN / ABS / GATE-condition on a raw price or funding level.
  plant / null  TV1 section 4. Both return a NEW shard built from the TRAIN arrays and run
                prepare_shard on it, so fwd_w is always recomputed from the modified fwd.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch

import m1_ruler as R
import m2h_common as M

IC_H = 168
IC_MIN_COINS = 20
ZCLIP = 3.0
PLANT_SEED = 20261009
PLANT_MIN_COINS = 20
LEVEL_TERMS = frozenset(("log_close", "log_high", "log_low", "log_quote_volume",
                         "log_oi_value", "funding_level"))
_LEVEL_PASS = frozenset(("NEG", "ABS", "LAG", "TS_MEAN", "TS_MAX", "TS_MIN", "EMA"))


# --------------------------------------------------------------------------- IC
def _ic_future(shard, s0):
    cache = shard.setdefault("_ic_future", {})
    if s0 not in cache:
        E = M.EPISODE
        rows = np.arange(M.LAG, E - IC_H + 1)            # 168 rows: forward 168 h inside the tile
        lr = np.log1p(np.nan_to_num(shard["fwd_w"][:, s0:s0 + E], nan=0.0))
        cs = np.concatenate([np.zeros((lr.shape[0], 1)), np.cumsum(lr, axis=1)], axis=1)
        fut = (cs[:, rows + IC_H] - cs[:, rows]).T       # (rows, A): hours s0+t .. s0+t+167
        cache[s0] = (rows, np.where(shard["eligible"][:, s0 + rows].T, fut, np.nan))
    return cache[s0]


def cs_spearman_rows(a, b, min_coins=IC_MIN_COINS):
    """Per-row Spearman of two (n, A) arrays on the jointly finite entries (average ranks).
    A row with fewer than min_coins entries or a constant side gives 0."""
    m = np.isfinite(a) & np.isfinite(b)
    ra = pd.DataFrame(np.where(m, a, np.nan)).rank(axis=1).to_numpy()
    rb = pd.DataFrame(np.where(m, b, np.nan)).rank(axis=1).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ra = np.nan_to_num(ra - np.nanmean(ra, axis=1, keepdims=True))
        rb = np.nan_to_num(rb - np.nanmean(rb, axis=1, keepdims=True))
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra * ra).sum(axis=1) * (rb * rb).sum(axis=1))
    ok = (m.sum(axis=1) >= min_coins) & (den > 0)
    return np.where(ok, num / np.where(den > 0, den, 1.0), 0.0)


def ic_tile(sigT, shard, s0):
    rows, fut = _ic_future(shard, s0)
    return float(cs_spearman_rows(sigT[rows], fut).mean())


def ic_future_rows(shard, s0):
    """(rows, future) used by ic_tile; TV1 check 3 feeds the future itself back as a signal."""
    return _ic_future(shard, s0)


# --------------------------------------------------------------------------- scoring
def score_dense(vm, body, slices, starts, univ, shard, policy_mod, *, holds=(168,),
                costs=M.COSTS, band=R.BAND, nav_usd=R.NAV_USD, want_ic=True, want_sig=False):
    """m2h_common.score_formula plus IC per tile. Raises M.ZeroCoverage like it does."""
    E = M.EPISODE
    holds = [int(h) for h in holds]
    logs = {h: {c: [] for c in costs} for h in holds}
    parts = {h: [] for h in holds}
    live_fracs, ics, sigs = [], [], []
    for s0, sl in zip(starts, slices):
        sigT = M.window_signal(vm, body, sl, s0, univ)
        el = shard["eligible"][:, s0:s0 + E].T
        live = np.isfinite(sigT) & el
        probe_t = [t for t in [M.LAG] + list(range(E // 4, E, E // 4)) if t < E]
        pf = [float(live[t].sum() / max(el[t].sum(), 1)) for t in probe_t]
        if min(pf, default=0.0) < M.MIN_COVERAGE:
            raise M.ZeroCoverage(f"ZERO_COVERAGE_MIN_{min(pf, default=0.0):.2f}")
        live_fracs.append(float(np.mean(pf)))
        sc = R.score_window(sigT, shard, s0, E, holds, policy_mod,
                            costs=costs, variants=("win",), band=band, nav_usd=nav_usd)
        for h in holds:
            lg, p = sc[h]["win"]
            for c in costs:
                logs[h][c].append(lg[c] if np.isfinite(lg[c]) else M.REWARD_FLOOR)
            parts[h].append(p)
        if want_ic:
            ics.append(ic_tile(sigT, shard, s0))
        if want_sig:
            sigs.append(sigT)
    return dict(per_hold=logs, parts=parts, live_frac=float(np.mean(live_fracs)), ic=ics, sig=sigs)


# --------------------------------------------------------------------------- weights
def group_slices(groups):
    off = 0
    for n in groups:
        yield slice(off, off + int(n))
        off += int(n)


def zfam(r, valid, groups, clip=ZCLIP):
    """Within-family z over the valid entries, clipped to +-clip; invalid entries -> nan."""
    r = np.asarray(r, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    out = np.full(len(r), np.nan)
    for sl in group_slices(groups):
        m = valid[sl]
        if not m.any():
            continue
        v = r[sl][m]
        sd = float(v.std())
        z = np.zeros_like(v) if sd == 0.0 else (v - v.mean()) / sd
        seg = np.full(m.shape, np.nan)
        seg[m] = np.clip(z, -clip, clip)
        out[sl] = seg
    return out


def combo(a, b, valid, groups):
    """A2's C1 / C2: z within family of (0.5 z(a) + 0.5 z(b)), inner z clipped to +-3."""
    return zfam(0.5 * zfam(a, valid, groups) + 0.5 * zfam(b, valid, groups), valid, groups,
                clip=np.inf)


def q95_family(r, groups, epsilon):
    """basin_risk_update's selection: select_risk_elite per family, weight = r - q on the
    elite (the advantage the update multiplies neglogp by), 0 elsewhere."""
    from experiments.dso_reference.dso_math import select_risk_elite
    r = np.asarray(r, dtype=np.float64)
    w = np.zeros(len(r))
    for sl in group_slices(groups):
        sel = select_risk_elite(r[sl], epsilon)
        w[sel.elite_indices + sl.start] = np.asarray(sel.clipped_rewards) - sel.baseline
    return w


def q95_global(r, epsilon):
    """Stage-1's selection: one select_risk_elite over the whole batch."""
    from experiments.dso_reference.dso_math import select_risk_elite
    r = np.asarray(r, dtype=np.float64)
    sel = select_risk_elite(r, epsilon)
    w = np.zeros(len(r))
    w[sel.elite_indices] = np.asarray(sel.clipped_rewards) - sel.baseline
    return w
def wcorr_fam(w, y, groups):
    """A-table read-out: Pearson(w, y) inside each family, averaged over the three. A family
    whose weights are all equal (e.g. no elite under whole-batch q95) gets no gradient
    signal and counts 0."""
    out = []
    for sl in group_slices(groups):
        a, b = np.asarray(w[sl], dtype=np.float64), np.asarray(y[sl], dtype=np.float64)
        if len(a) < 3 or a.std() == 0.0 or b.std() == 0.0:
            out.append(0.0)
        else:
            out.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(out))


# --------------------------------------------------------------------------- level flag
PRICE_FUND_LEVEL = frozenset(("log_close", "log_high", "log_low", "funding_level"))


def level_flag(vm, body):
    """AGENTS.md pit: SIGN / ABS of a raw price or funding LEVEL ("crosses $1", "funding is
    exactly 0"). A value is a level if it is such a terminal, or a level passed through NEG,
    LAG, TS_MEAN/MAX/MIN, EMA, or the ADD of two levels. Flagged: SIGN or ABS of a level,
    GATE whose condition is a level, PER_ASSET_THRESH on a level signal."""
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import OPERATOR_ARITIES, WINDOWED_OPERATORS
    arity = dict(OPERATOR_ARITIES)
    stack, flag = [], False
    for t in body:
        t = int(t)
        if vm.is_end(t):
            break
        if vm.is_terminal(t):
            stack.append(M.TERMINALS[t] in PRICE_FUND_LEVEL)
        elif vm.is_window(t):
            stack.append(None)
        else:
            name = vm.operator_name(t)
            if name in WINDOWED_OPERATORS:
                stack.pop()
            args = [stack.pop() for _ in range(arity[name])][::-1]
            if name in ("SIGN", "ABS", "GATE", "PER_ASSET_THRESH") and args[0]:
                flag = True
            stack.append(bool((name in _LEVEL_PASS and args[0]) or (name == "ADD" and all(args))))
    return flag


# --------------------------------------------------------------------------- plant / null
def plant_body(vm):
    """DELTA(TS_MEAN(log_quote_volume, 168), 168): a T1 formula, unrelated to momentum."""
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import OPERATOR_NAMES
    w = vm.window_base + list(M.WINDOWS).index(168)
    op = lambda n: vm.operator_base + list(OPERATOR_NAMES).index(n)
    return [M.TERMINALS.index("log_quote_volume"), w, op("TS_MEAN"), w, op("DELTA"), vm.end_token]


def plant_signal(vm, feats, univ, device):
    """s (A, T): the plant formula's TOP100 cross-sectional average rank, demeaned, unit std
    per hour; NaN outside the pool or where the formula is undefined."""
    x = torch.from_numpy(np.ascontiguousarray(feats)).to(device=device, dtype=torch.float64)
    raw = vm.execute(plant_body(vm), x).detach().cpu().numpy()
    del x
    v = np.where(univ & np.isfinite(raw), raw, np.nan)
    rk = pd.DataFrame(v.T).rank(axis=1).to_numpy()            # (T, A)
    n = np.isfinite(rk).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rk = rk - np.nanmean(rk, axis=1, keepdims=True)
        sd = np.nanstd(rk, axis=1, keepdims=True)
    s = np.where((n[:, None] >= PLANT_MIN_COINS) & (sd > 0), rk / np.where(sd > 0, sd, 1.0), np.nan)
    return s.T


def _new_shard(fwd, eligible, fi, fr):
    sh = dict(fwd=fwd, eligible=eligible, fi=fi, fr=fr)
    R.prepare_shard(sh, R.WINSOR_PCT)           # fwd_w recomputed from THIS fwd
    return sh


def plant_shard(base, s, alpha):
    """fwd[i,t] += alpha * s[i,t-1] (LAG 1); an untradable hour stays NaN."""
    add = np.zeros_like(base["fwd"])
    add[:, 1:] = np.nan_to_num(s[:, :-1], nan=0.0)
    return _new_shard(base["fwd"] + float(alpha) * add, base["eligible"].copy(),
                      base["fi"].copy(), base["fr"].copy())


def null_shard(base, univ, seed=PLANT_SEED):
    """Per hour t, one random permutation of the coins in the pool of the signal row t-1,
    applied jointly to (fwd, eligible, funding settlement fi/fr). Terminals untouched."""
    rng = np.random.default_rng(seed)
    fwd, el, fi, fr = (base[k].copy() for k in ("fwd", "eligible", "fi", "fr"))
    for t in range(1, fwd.shape[1]):
        idx = np.flatnonzero(univ[:, t - 1])
        if len(idx) < 2:
            continue
        p = rng.permutation(idx)
        for a, b in ((fwd, base["fwd"]), (el, base["eligible"]), (fi, base["fi"]), (fr, base["fr"])):
            a[idx, t] = b[p, t]
    return _new_shard(fwd, el, fi, fr)


def corr_with_s(sigT, s, s0, step=24):
    """Mean CS Spearman of a tile signal (E, A) with s at the same signal hour, every step h."""
    rows = np.arange(M.LAG, M.EPISODE, step)
    return float(cs_spearman_rows(sigT[rows], s[:, s0 - M.LAG + rows].T).mean())


def bpd(log_nav):
    return 1e4 * float(log_nav) / (M.EPISODE / 24.0)
