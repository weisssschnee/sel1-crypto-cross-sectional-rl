"""M2H shared infrastructure: hourly grid, three strategy-family basins (run BASIN3_H1_20261009).

Derived from m1_common.py (stage-1, sha-pinned, NOT modified). The book, the hold policy and the
accounting are not re-implemented here: rank_weights / forward_returns come from m1_common and
score_window / held_path / settle_multi from m1_ruler, exactly as h1_handwritten_baselines.py calls
them, so the handwritten baselines and this trainer share one ruler.

Hourly contract (BASIN3 handoff 2026-10-09):
  grid      1 h; lag 1 = the signal of row t is first held at row t+1
  tiles     336 h (2 weeks), disjoint, after a 672 h lookback; each tile starts flat
  universe  top 100 by trailing 720 h quote volume (min 360 h), rank method "first"; applied to the
            formula output at the SIGNAL row (pool-external values -> NaN), as in h1
  basins    T1 momentum / T2 crowding / T3 basis; the model's first input token is the basin
            token B_T1/B_T2/B_T3 (it replaces the sampler's BOS slot), and the legality table is
            built per basin, so a program may only reference its own basin's terminals. T2 and
            T3 share funding_level, so the basin token is what tells the policy which it is in.
  VM        xs_tscs_vm_v2 unchanged; windows are hours
TRAIN 2024-01..2025-12; DEV 2026-01..06 eval-only; OOS 2026-07+ is never read.

Why window_signal is re-written here instead of calling R.window_signal: m1_ruler.LAG is
C.EXECUTION_LAG_MIN = 2 MINUTES, and m1_ruler is sha-pinned. The hourly grid runs a 1-row lag, so
the episode columns are taken at an offset of LAG=1 instead. Everything downstream (score_window,
held_path, settle_multi) is the untouched ruler.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

import m1_common as C
import m1_ruler as R

WINDOWS = (6, 24, 72, 168, 336, 504, 672)
HOLDS = (4, 24, 72, 168)
HOLD_NAMES = [f"H{h}h" for h in HOLDS]
LAG = 1
EPISODE = 336
LOOKBACK = 672
MAX_PROGRAM_LEN = 48
COSTS = ("flat6", "cost0")
REWARD_FLOOR = C.REWARD_FLOOR
MIN_COVERAGE = 0.1
UNIV_N, UNIV_WIN, UNIV_MIN = 100, 720, 360
OOS_FIRST = pd.Timestamp("2026-07-01")

TERMINALS = ("log_close", "log_high", "log_low", "log_quote_volume",
             "funding_level", "log_oi_value", "global_ls_account_ratio", "taker_buy_share",
             "premium_close_bps", "mark_index_basis_bps", "premium_high")
BASIN_NAMES = ("T1", "T2", "T3")
BASIN_TOKEN_NAMES = tuple(f"B_{b}" for b in BASIN_NAMES)
# T2 and T3 deliberately hold no price field, so they cannot degenerate into momentum.
BASIN_TERMS = {"T1": (0, 1, 2, 3),
               "T2": (4, 5, 6, 7),
               "T3": (8, 9, 10, 4)}
PANEL_COLS = ["trade_close", "trade_high", "trade_low", "trade_quote_volume", "taker_buy_quote_volume",
              "funding_rate", "open_interest_value_last", "global_long_short_account_ratio_last",
              "premium_close_bps", "mark_index_basis_bps", "premium_high"]


class ZeroCoverage(Exception):
    """Raised when a window's live-asset fraction falls under MIN_COVERAGE."""


def quota(n, k=len(BASIN_NAMES)):
    """86/85/85 for 256: the remainder goes to the first basins."""
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


def basin_terms(name):
    return tuple(BASIN_TERMS[name])


def _pos_log(x):
    return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)


def long_to_grids(t, cols, syms=None, ts=None):
    """Long rows (symbol, timestamp, cols) -> {col: (A, T) float64}; absent rows stay NaN."""
    syms = sorted(t.symbol.unique()) if syms is None else list(syms)
    ts = pd.date_range(t.timestamp.min(), t.timestamp.max(), freq="h") if ts is None else ts
    si = pd.Index(syms).get_indexer(t.symbol)
    ti = ((t.timestamp - ts[0]) / pd.Timedelta(hours=1)).to_numpy()
    if not np.all(ti == np.round(ti)):
        raise RuntimeError("timestamps are not on the hourly grid")
    ti = ti.astype(int)
    ok = (si >= 0) & (ti >= 0) & (ti < len(ts))
    g = {}
    for c in cols:
        a = np.full((len(syms), len(ts)), np.nan)
        a[si[ok], ti[ok]] = t[c].to_numpy(dtype=np.float64)[ok]
        g[c] = a
    return syms, ts, g


def load_train_grids(path):
    C.guard_no_oos(path)
    t = pq.read_table(path, columns=["symbol", "timestamp"] + PANEL_COLS + ["forward_trade_return_1h"]).to_pandas()
    if t.timestamp.max() >= pd.Timestamp("2026-01-01"):
        raise RuntimeError("TRAIN panel reaches DEV")
    return long_to_grids(t, PANEL_COLS + ["forward_trade_return_1h"])


def build_world(g):
    """Grids -> terminals (A, F, T), universe mask, ruler shard. Same constructions as h1."""
    close = g["trade_close"]
    qv = g["trade_quote_volume"]
    fr_raw = g["funding_rate"]
    feats = np.stack([
        _pos_log(close), _pos_log(g["trade_high"]), _pos_log(g["trade_low"]), _pos_log(qv),
        pd.DataFrame(fr_raw.T).ffill().to_numpy().T,
        _pos_log(g["open_interest_value_last"]),
        g["global_long_short_account_ratio_last"],
        np.where(qv > 0, g["taker_buy_quote_volume"] / np.where(qv > 0, qv, 1.0), np.nan),
        g["premium_close_bps"], g["mark_index_basis_bps"], g["premium_high"],
    ], axis=1)
    vol30 = pd.DataFrame(qv.T).rolling(UNIV_WIN, min_periods=UNIV_MIN).sum()
    univ = (vol30.rank(axis=1, ascending=False, method="first") <= UNIV_N).to_numpy().T
    fwd = C.forward_returns(close, None)
    shard = dict(fwd=fwd, eligible=np.isfinite(fwd), fi=np.isfinite(fr_raw),
                 fr=np.where(np.isfinite(fr_raw), fr_raw, 0.0))
    R.prepare_shard(shard, R.WINSOR_PCT)
    return dict(feats=feats, univ=univ, shard=shard, fwd=fwd)


def tile_starts(T, first=LOOKBACK):
    """Disjoint back-to-back 336 h tiles after the lookback; 50 on the TRAIN panel."""
    return list(range(first, T - EPISODE + 1, EPISODE))


def pad_feats(feats):
    """Prepend LAG all-NaN columns so absolute hour h lives at index h + LAG.

    A tile at s0 = LOOKBACK needs the signal row s0 - LAG, which is one hour before the
    lookback window starts; the padding gives the VM a real column there instead of
    shifting the whole grid."""
    A, F, T = feats.shape
    out = np.full((A, F, T + LAG), np.nan, dtype=feats.dtype)
    out[:, :, LAG:] = feats
    return out


def tile_slices(feats_pad, starts, device, dtype=torch.float32):
    """GPU feature slices: pad indices [s0-LOOKBACK, s0+LAG+EPISODE) -> absolute hours
    [s0-LOOKBACK-LAG, s0+EPISODE). The episode columns then carry LAG rows of pre-signal
    history plus EPISODE signal rows.

    float32 for training (the 2026-10-02 float32-GPU ruling; float64 would double the 12 GiB
    peak). The h1 reproduction gate runs float64, and the float32 delta is reported beside it."""
    out = []
    for s0 in starts:
        lo, hi = s0 - LOOKBACK, s0 + LAG + EPISODE
        if lo < 0 or hi > feats_pad.shape[2]:
            raise RuntimeError(f"tile {s0} needs pad columns [{lo},{hi}) of {feats_pad.shape[2]}")
        t = torch.from_numpy(np.ascontiguousarray(feats_pad[:, :, lo:hi]))
        out.append(t.to(device=device, dtype=dtype))
    return out


def window_signal(vm, body, sl, s0, univ):
    """VM signal for one tile, lag applied, universe mask applied at the SIGNAL row, (E, A).

    The slice spans LOOKBACK + LAG + EPISODE columns; the VM value at slice column k is the
    value at absolute hour s0 - LOOKBACK - LAG + k, so the episode is read at
    k in [LOOKBACK, LOOKBACK+EPISODE) = absolute hours [s0-LAG, s0-LAG+EPISODE). That is
    exactly h1's ``sigL[:, LAG:] = sig[:, s0:s0+E-LAG]`` with the TOP100 mask taken at the
    same absolute hour as the signal."""
    sig = vm.execute(list(body), sl).detach().cpu().numpy()
    ep = sig[:, LOOKBACK:LOOKBACK + EPISODE]
    ep = np.where(univ[:, s0 - LAG: s0 - LAG + EPISODE], ep, np.nan)
    # The tile starts flat: a signal computed before the tile's first row is not carried in.
    # This is h1's ``sigL`` (its first LAG columns stay NaN) and the handoff's "each tile starts
    # flat". Without it the book would already be positioned on row 0.
    if LAG:
        ep = ep.copy()
        ep[:, :LAG] = np.nan
    return ep.T


def score_formula(vm, body, slices, starts, univ, shard, holds, policy_mod, *,
                  costs=COSTS, band=R.BAND, nav_usd=R.NAV_USD):
    """Score one formula body at every hold on every tile through the shared ruler.

    per_hold[h][cost] -> [net log NAV per tile]; parts[h] -> [parts dict per tile].
    Raises ZeroCoverage for an extreme window, as the M2 contract defines it (probe
    live fraction < MIN_COVERAGE)."""
    E = EPISODE
    holds = [int(h) for h in holds]
    logs = {h: {c: [] for c in costs} for h in holds}
    parts = {h: [] for h in holds}
    live_fracs = []
    for s0, sl in zip(starts, slices):
        sigT = window_signal(vm, body, sl, s0, univ)
        el = shard["eligible"][:, s0:s0 + E].T
        live = np.isfinite(sigT) & el
        probe_t = [t for t in [LAG] + list(range(E // 4, E, E // 4)) if t < E]
        pf = [float(live[t].sum() / max(el[t].sum(), 1)) for t in probe_t]
        if min(pf, default=0.0) < MIN_COVERAGE:
            raise ZeroCoverage(f"ZERO_COVERAGE_MIN_{min(pf, default=0.0):.2f}")
        live_fracs.append(float(np.mean(pf)))
        sc = R.score_window(sigT, shard, s0, E, holds, policy_mod,
                            costs=costs, variants=("win",), band=band, nav_usd=nav_usd)
        for h in holds:
            lg, p = sc[h]["win"]
            for c in costs:
                logs[h][c].append(lg[c] if np.isfinite(lg[c]) else REWARD_FLOOR)
            parts[h].append(p)
    return dict(per_hold=logs, parts=parts, live_frac=float(np.mean(live_fracs)))


def reward_of(res, hold, cost="flat6", how="mean"):
    """The scalar the optimiser sees: aggregate of per-tile net log NAV."""
    arr = np.asarray(res["per_hold"][int(hold)][cost], dtype=np.float64)
    return R.aggregate(arr, how), arr


def diagnostics(parts, days=None):
    """Per-formula book diagnostics (never a reward term), averaged over tiles."""
    days = days or EPISODE / 24.0
    betas = [p["beta_mkt"] for p in parts if np.isfinite(p["beta_mkt"])]
    return dict(turnover_per_day=float(np.mean([p["turnover"] for p in parts])) / days,
                net_exposure=float(np.mean([p["net_exposure_mean"] for p in parts])),
                gross_exposure=float(np.mean([p["gross_exposure_mean"] for p in parts])),
                beta_mkt=float(np.mean(betas)) if betas else float("nan"),
                gross_bp_day=1e4 * float(np.mean([p["gross"] for p in parts])) / days)
