"""M1 shared infrastructure: 1-min decision grid data, episodes, evaluation.

Reuses, unchanged: the stack VM (xs_tscs_vm_v2.XsTsCsStackVM with a minute-scale
window ladder), the stateful action policy (its minimum_hold counts TIMESTEPS,
so minutes pass straight through), and the accounting oracle (grid-agnostic
fee/slippage/funding). New here: month-shard loading of the top100 aggtrades
source, 16-terminal vocabulary, episode slicing on the 1-min grid, the
execution lag, and per-formula HOLD.

M1 time contract: TRAIN 2024-01..2025-12, DEV 2026-01..06, sealed OOS at
2026-07-01 (never read; see M1_SPLIT_LEDGER_V1.json).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# ---- vocabulary (ruling 2026-10-02; +funding_level 2026-10-07) -------------
# funding_level is the ffill of the raw-ms funding settlement rate, 0.0 where a
# symbol has no funding that month. It is 0.0 rather than NaN on purpose: the
# economically neutral funding rate is 0, and a NaN column would zero the VM
# weight and pollute the cross-section operators. funding_covered is stored
# alongside so the fill can be audited (see m1_build_shards.build).
TERMINALS = (
    "close_price", "vwap", "price_std", "price_range_bps", "close_to_open_bps",
    "notional", "quantity", "agg_trade_count", "buy_notional", "sell_notional",
    "signed_aggressor_notional", "volume_imbalance", "buy_sell_notional_ratio",
    "large_notional_ratio_100k_plus", "avg_agg_trade_notional", "max_trade_notional",
    "funding_level",
)
# Columns read straight from the aggtrades source parquet. funding_level is not
# among them: it comes from the funding tables, not from the aggtrades file.
SOURCE_COLUMNS = tuple(t for t in TERMINALS if t != "funding_level")
WINDOWS = (1, 5, 15, 30, 60, 240, 1440)           # TS-operator windows, minutes
HOLDS = (1, 5, 15, 30, 60, 240, 1440, 2880, 10080)  # per-formula minimum hold, minutes
FEE_RATE = 0.0005
SLIPPAGE_RATE = 0.0001
REWARD_FLOOR = -2.0
MAX_PROGRAM_LEN = 48
EPISODE_MINUTES = 360                             # 6h ruling
N_EPISODES = 8
LOOKBACK_MINUTES = 1440                           # longest TS window
EXECUTION_LAG_MIN = 2                             # verified from execution_time_min

ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem"
SOURCE = os.environ.get("XS_M1_SOURCE",
                        r"G:\AlphaFactory_CryptoData\silver\binance_hf_flow_1m_202401_202606_v1\binance_aggtrades_top100_compact_1m_202401_202606\compact_1m")
FUNDING_SILVER = os.environ.get("XS_M1_FUNDING_SILVER",
                                r"G:\AlphaFactory_CryptoData\silver\binance_api\funding_rate")
M1_STAGE = os.path.join(ROOT, "runtime", "m1_staging")
OOS_FIRST = pd.Timestamp("2026-07-01 00:00:00+00:00")


def guard_no_oos(path):
    """Parquet footer check: refuse any file whose timestamp statistics reach the seal."""
    md = pq.ParquetFile(path).metadata
    names = md.schema.to_arrow_schema().names
    limit = OOS_FIRST
    for col in ("timestamp",):
        if col not in names:
            continue
        j = names.index(col)
        for g in range(md.num_row_groups):
            st = md.row_group(g).column(j).statistics
            if st is None or not st.has_min_max:
                raise RuntimeError(f"{path}: no {col} statistics; cannot verify the seal")
            top = pd.Timestamp(st.max)
            if top.tzinfo is not None:
                top = top.tz_convert("UTC").tz_localize(None)
            limit_naive = limit.tz_convert("UTC").tz_localize(None) if limit.tzinfo is not None else limit
            if top >= limit_naive:
                raise RuntimeError(f"{path}: {col} reaches {top}, inside the sealed OOS window")


def month_symbols(month: str, membership):
    return [s for s, r in membership.items() if r["first"] <= month <= r["last"]]


def funding_level_grid(fund_rows, n_assets, n_minutes):
    """Expand (sym_idx, minute_idx, rate) events into a dense (A, T) grid.

    The rate is held forward from each settlement until the next one (ffill),
    because the funding a position actually pays is the rate in force, not the
    rate that happened to settle on that minute. A symbol with no event that
    month stays 0.0 -- its economically neutral funding rate -- and is flagged
    False in the returned mask so the fill is auditable after the fact.

    Returns (grid, covered) with grid float32 (A, T) and covered bool (A,).
    """
    grid = np.zeros((n_assets, n_minutes), dtype=np.float32)
    covered = np.zeros(n_assets, dtype=bool)
    if not fund_rows:
        return grid, covered
    by_asset = {}
    for sym_idx, minute_idx, rate in fund_rows:
        by_asset.setdefault(int(sym_idx), []).append((int(minute_idx), float(rate)))
    for sym_idx, events in by_asset.items():
        if not (0 <= sym_idx < n_assets):
            raise IndexError(f"funding symbol index {sym_idx} outside universe of {n_assets}")
        events.sort()
        covered[sym_idx] = True
        col = grid[sym_idx]
        last = 0.0
        pos = 0
        for minute_idx, rate in events:
            if minute_idx >= n_minutes:
                break
            # hold `last` across [pos, minute_idx), then settle at minute_idx
            if minute_idx > pos:
                col[pos:minute_idx] = last
            last = rate
            pos = minute_idx
        if pos < n_minutes:
            col[pos:] = last
    return grid, covered


def load_month(month: str, symbols):
    """One month shard: (A, F, T) float32 features + close grid + minute axis.

    Rows are (symbol, minute); absent rows stay NaN (no zero fill). Returns
    dense grids padded to the month's full minute span. funding_level is left
    at 0.0 here because its values come from the funding tables rather than the
    aggtrades parquet; funding_level_grid fills it at shard-build time.
    """
    first = pd.Timestamp(month + "-01", tz="UTC")
    n_min = first.days_in_month * 1440
    times = pd.date_range(first, periods=n_min, freq="1min")
    feats = np.full((len(symbols), len(TERMINALS), n_min), np.nan, dtype=np.float32)
    close = np.full((len(symbols), n_min), np.nan, dtype=np.float64)
    for i, sym in enumerate(symbols):
        p = os.path.join(SOURCE, f"symbol={sym}", f"month={month}", "part.parquet")
        if not os.path.exists(p):
            continue
        t = pq.ParquetFile(p).read(columns=["timestamp", "close_price"] + list(SOURCE_COLUMNS)).to_pandas()
        ts = pd.DatetimeIndex(t["timestamp"])
        assert ts.min() >= first and ts.max() < first + pd.Timedelta(minutes=n_min)
        idx = np.round((ts - first) / pd.Timedelta(minutes=1)).astype("int64")
        assert idx.min() >= 0 and idx.max() < n_min
        close[i, idx] = t["close_price"].to_numpy(dtype=np.float64)
        for k, f in enumerate(SOURCE_COLUMNS):
            feats[i, k, idx] = t[f].to_numpy(dtype=np.float32)
    return dict(month=month, symbols=list(symbols), times=times, feats=feats, close=close)


def funding_events(month: str, symbols, raw_dir):
    """(symbol_idx, minute_idx, rate) for settlements inside the month.

    Raw ms instants from the m1_funding_raw_v1 CSVs (fapi) -- floor_to_minute,
    per correction A. Never the 1h line's floored hours.
    """
    first = pd.Timestamp(month + "-01", tz="UTC")
    lo = int(first.timestamp() * 1000)
    hi = lo + first.days_in_month * 86400_000
    out = []
    for i, sym in enumerate(symbols):
        p = os.path.join(raw_dir, f"{sym}.csv")
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        ms = t["fundingTime"].astype("int64")
        rate = t["fundingRate"].astype(float)
        m = (ms >= lo) & (ms < hi)
        if not m.any():
            continue
        minute_idx = ((ms[m] - lo) // 60_000).to_numpy()
        for mi, r in zip(minute_idx, rate[m]):
            out.append((i, int(mi), float(r)))
    return out


def episode_slices(rng, n_min, n_episodes=N_EPISODES):
    """Random 6h episode starts with full 1440-min lookback inside an n_min shard."""
    usable = n_min - EPISODE_MINUTES
    edges = np.linspace(LOOKBACK_MINUTES, usable, n_episodes + 1).astype(int)
    return [int(rng.integers(edges[k], max(edges[k] + 1, edges[k + 1]))) for k in range(n_episodes)]


def forward_returns(close, eligible):
    """close[m+1]/close[m]-1 on contiguous minutes; NaN elsewhere (no fill)."""
    nxt = np.full_like(close, np.nan)
    contig = np.zeros(close.shape, dtype=bool)
    contig[:, :-1] = np.isfinite(close[:, 1:]) & np.isfinite(close[:, :-1])
    nxt[:, :-1] = np.where(contig[:, :-1], close[:, 1:] / close[:, :-1] - 1.0, np.nan)
    return nxt


def rank_weights(sig, eligible, min_live=10):
    """Vectorised basin_common.signal_to_weights for (T, A) signals.

    Per minute: rank live assets (ascending), centre, divide by sum of |centred|
    — the B-line's per-timestep semantics, but one argsort per episode slice.
    Minutes with < min_live live assets -> 0.

    Tied values share their average rank. With plain stable-argsort ranks a tie
    is broken by asset index, so a constant or all-+1 signal became a fixed
    long/short book ordered by symbol (gross 1.0, month P&L of +-10 bp/day):
    an unintended zero-turnover bet. Now it has zero centred rank everywhere and
    holds nothing. Without ties the result is bit-identical to the old version.
    """
    live = np.isfinite(sig) & eligible
    n = live.sum(axis=1)
    big = np.where(live, sig, np.inf)
    order = np.argsort(big, axis=1, kind="stable")
    sv = np.take_along_axis(big, order, axis=1)
    n_t, n_a = sig.shape
    pos = np.arange(n_a)[None, :]
    first = np.ones((n_t, n_a), dtype=bool)
    first[:, 1:] = sv[:, 1:] != sv[:, :-1]
    last = np.ones((n_t, n_a), dtype=bool)
    last[:, :-1] = first[:, 1:]
    lo = np.maximum.accumulate(np.where(first, pos, 0), axis=1)
    hi = np.minimum.accumulate(np.where(last, pos, n_a)[:, ::-1], axis=1)[:, ::-1]
    ranks = np.empty_like(sig)
    rows = np.arange(n_t)[:, None]
    ranks[rows, order] = (lo + hi) / 2.0
    ranks = np.where(live, ranks, 0.0)
    ok = n >= min_live
    cent = ranks - np.maximum(n - 1, 0)[:, None] / 2.0
    denom = np.abs(np.where(live, cent, 0.0)).sum(axis=1)
    safe = ok & (denom > 0)
    w = np.where(live, cent, 0.0) / np.where(safe, denom, 1.0)[:, None]
    return np.where(safe[:, None], w, 0.0)


def settle_episode(held, fwd, funding_mask, funding_rate, fee, slip):
    """M1's defined accounting: vectorised per-minute NAV of a weight path.

    net_t = sum_a(w[t,a] * fwd[t,a]) - (fee+slip)*turnover_t - funding_t;
    episode log NAV = sum_t log(1+net_t). held comes from the stateful action
    policy (which already zeroes ineligible coordinates), so eligibility is
    respected upstream; fwd is NaN where not tradable and contributes 0.

    Turnover treats NaN-fwd minutes as a "frozen" window: the position is
    carried forward and no entry/exit is charged across the gap. We
    forward-fill held along the time axis, mask the turnovers that would
    otherwise be charged by the re-entry at the start of the gap, and
    collapse any held change that straddles NaN minutes into a single
    re-entry at the moment trading resumes.

    held: (T, A) weights AFTER the execution lag is applied by the caller.
    This is THE M1 cost definition (declared in the preflight); training and
    scoring share it, so train/score divergence cannot happen.
    """
    tradable = np.isfinite(fwd)
    held_ff = np.where(tradable, held, 0.0)
    if held.shape[1] > 0 and held.shape[0] > 0:
        prev = held_ff[0].copy()
        for t in range(1, held.shape[0]):
            cur = held[t]
            mask = tradable[t]
            out = np.where(mask, cur, prev)
            held_ff[t] = out
            prev = np.where(mask, cur, prev)
    turn = np.abs(np.diff(held_ff, axis=0, prepend=0.0)).sum(axis=1)
    gross = np.nansum(held * np.nan_to_num(fwd), axis=1)
    cost = (fee + slip) * turn
    fund = np.nansum(held * np.where(funding_mask, funding_rate, 0.0), axis=1)
    net = gross - cost - fund
    with np.errstate(divide="ignore", invalid="ignore"):
        log_nav = float(np.sum(np.log1p(np.clip(net, -0.9, None))))
    return log_nav, dict(gross=float(gross.sum()), cost=float(cost.sum()), funding=float(fund.sum()), turnover=float(turn.sum()))


def settle_episode_ts(held, fwd_per_asset, funding_mask_per_asset,
                       funding_rate_per_asset, fee, slip):
    """M4 TS-first per-asset accounting.

    The CS eval assumes "alpha = a cross-sectional rotation". On the 1-min grid
    that hypothesis was rejected -- the audit showed TS-IC is 3-9x CS-IC for
    the flow terminals (close_to_open_bps, signed_aggressor_notional,
    volume_imbalance), and the eval axis was throwing that information away by
    ranking all assets together every minute.

    The TS pathway instead runs one independent mark-to-market per asset:
    a long/short/sign exit signal opens per asset, the position persists for
    HOLD minutes (cross-asset unsynchronised), and log NAV is summed per asset
    first then averaged cross-sectionally. Cross-sectional features can be
    used as GATES ("trade this asset") but not as the whole position.

    held:                  (A, T) per-asset signed weight path AFTER execution lag
    fwd_per_asset:           (A, T) forward returns for this asset's bars
    funding_mask_per_asset:  (A, T) mask of funding settlement minutes
    funding_rate_per_asset:  (A, T) funding rate at each minute
    fee, slip:               per-unit turnover cost components

    Returns (mean log_nav across active assets, per_asset_log_nav (A,), parts).
    """
    held = np.asarray(held, dtype=np.float64)
    if held.ndim == 1:
        held = held.reshape(-1, 1)
    fw = np.nan_to_num(fwd_per_asset)         # (A, T)
    mask = funding_mask_per_asset             # (A, T)
    rate = funding_rate_per_asset             # (A, T)

    bar_pnl = held * fw
    turn = np.abs(np.diff(held, axis=1, prepend=0.0))
    cost = (fee + slip) * turn
    fund = held * np.where(mask, rate, 0.0)
    net = bar_pnl - cost - fund
    with np.errstate(divide="ignore", invalid="ignore"):
        per_asset_log = np.where(
            np.isfinite(fwd_per_asset).any(axis=1),
            np.sum(np.log1p(np.clip(net, -0.9, None)), axis=1),
            np.nan)
    valid = np.isfinite(per_asset_log)
    if not valid.any():
        return 0.0, per_asset_log, dict(gross=0.0, cost=0.0, funding=0.0, turnover=0.0)
    mean_log = float(np.nanmean(per_asset_log))
    parts = dict(
        gross=float(np.nansum(bar_pnl)),
        cost=float(np.nansum(cost)),
        funding=float(np.nansum(fund)),
        turnover=float(turn.sum()))
    return mean_log, per_asset_log, parts


def settle_episode_ts_depth(held, fwd_per_asset, funding_mask_per_asset,
                            funding_rate_per_asset, bid_depth_usd, ask_depth_usd,
                            fee, slip, depth_fraction=0.01):
    """Per-asset accounting with realistic slip from bookDepth.

    Same shape contract as settle_episode_ts, but slip is computed from
    |delta_pos| / bid_depth (or ask_depth for shorts) at each minute.
    bid_depth_usd / ask_depth_usd: (A, T) USD notional within
    `depth_fraction` (default 1%) of mid -- typically from
    bid_notional_1pct / ask_notional_1pct of the bookDepth data.

    Entry fee stays as the flat `fee` rate (per leg). Slip from depth is
    ADDED on top. NaN depth -> fall back to flat `slip` for that minute.
    """
    held = np.asarray(held, dtype=np.float64)
    if held.ndim == 1:
        held = held.reshape(-1, 1)
    A, T = held.shape
    fw = np.nan_to_num(fwd_per_asset)
    mask = funding_mask_per_asset
    rate = funding_rate_per_asset

    # Turnover in position units (asset notional). Convert to USD using
    # bid/ask depth as price proxy: trade |delta| assets, ask is the
    # natural price for buying, bid for selling. This is a coarse proxy;
    # the real implementation would consume the full L2 book per minute.
    delta = np.abs(np.diff(held, axis=1, prepend=0.0))  # (A, T)
    is_buy = (np.diff(held, axis=1, prepend=0.0)) > 0  # entry/long
    is_sell = (np.diff(held, axis=1, prepend=0.0)) < 0  # exit/short
    bid = np.asarray(bid_depth_usd, dtype=np.float64)
    ask = np.asarray(ask_depth_usd, dtype=np.float64)
    # depth fraction multiplier: |delta| / depth * mid_proxy -> impact in USD
    # Use bid for sells (close longs), ask for buys (open/increase longs)
    depth_used = np.where(is_buy, ask, bid)  # (A, T)
    depth_used = np.where(is_sell, bid, depth_used)
    depth_finite = np.isfinite(depth_used) & (depth_used > 0)
    # impact in fraction-of-mid: |delta| * price / depth, where price ~ 1
    # for normalized weights. For raw asset counts the impact is the trade
    # size as fraction of depth. We assume |delta| is in USD notional (already
    # scaled by price); if the caller passes raw asset counts, the impact is
    # in price-relative terms and should be multiplied by mid-price.
    impact_frac = np.where(depth_finite, delta / np.where(depth_finite, depth_used, 1.0), np.nan)
    impact_bps = impact_frac * 1e4
    # fallback: when depth NaN, use flat `slip` rate (legacy model)
    fallback_bps = np.full_like(delta, slip * 1e4)
    slip_bps = np.where(depth_finite, impact_bps, fallback_bps)

    bar_pnl = held * fw
    cost = (fee * 1e4 + slip_bps) * 1e-4 * delta  # bps to fraction * |delta|
    fund = held * np.where(mask, rate, 0.0)
    net = bar_pnl - cost - fund
    with np.errstate(divide="ignore", invalid="ignore"):
        per_asset_log = np.where(
            np.isfinite(fwd_per_asset).any(axis=1),
            np.sum(np.log1p(np.clip(net, -0.9, None)), axis=1),
            np.nan)
    valid = np.isfinite(per_asset_log)
    if not valid.any():
        return 0.0, per_asset_log, dict(gross=0.0, cost=0.0, funding=0.0, turnover=0.0,
                                        slip_from_depth_bps=float(np.nanmean(slip_bps)))
    mean_log = float(np.nanmean(per_asset_log))
    parts = dict(
        gross=float(np.nansum(bar_pnl)),
        cost=float(np.nansum(cost)),
        funding=float(np.nansum(fund)),
        turnover=float(delta.sum()),
        slip_from_depth_bps=float(np.nanmean(slip_bps)),
    )
    return mean_log, per_asset_log, parts


def ts_signal_to_per_asset_pos(sig_per_asset, threshold, min_live_frac=0.0):
    """Threshold a per-asset signal into {-1, 0, +1} by sign (if magnitude > thresh).

    Per-asset independent position decisions -- the  fundamental change vs
    M3 is that there is no cross-sectional ranking here. A formula with high
    TS-IC produces a sig_per_asset where each entry is informative in itself.
    """
    s = np.asarray(sig_per_asset, dtype=np.float64).reshape(-1)
    pos = np.zeros_like(s)
    finite = np.isfinite(s)
    pos[finite & (s > threshold)] = 1.0
    pos[finite & (s < -threshold)] = -1.0
    return pos
