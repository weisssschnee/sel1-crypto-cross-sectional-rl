"""M1 stage-1 ruler: ONE net-dollar scoring function for training, selection and DEV.

m2_trainer (select windows, holdout windows, elite re-check) and dev_score call
eval_formula below; there is no second implementation, so train/select/DEV
cannot drift apart (work order items 6, 10, 11).

What the ruler fixes, each measured before it was built (see the P1 gate-13 table):
  rule 0   The stateful policy sees every asset as eligible, so a missing 1-min
           bar freezes the position instead of forcing a whole-book retarget.
           settle_multi freezes through the same gap, so action and accounting
           agree. The old behaviour put a hold-independent turnover floor under
           the 6 bp cost (DEV 2026-03: 125 turnover/day at hold 1440).
  winsor   Forward returns are clipped per minute to the 1/99 cross-sectional
           percentiles before settlement, so rank-book dollars cannot be
           decided by one pump tail (inward nearest-rank: with n=59..100 live
           assets this clips the single extreme on each side).
  cost     flat6 = fee 5bp + slip 1bp (the old M1 cost); maker2 = 2bp, no slip
           (optimistic bound); depth = fee 5bp + book-walk impact over bookDepth
           bands 1..5% at an assumed NAV (impact only, the 1% band cannot
           resolve the half-spread, so this is an optimistic bound too).
  band     StatefulActionContract.hold_l1_threshold; the policy already holds
           inside it and moves halfway between band and band+0.5. The default
           0.5 is the contract default, i.e. the pre-existing behaviour.
Reward is net log NAV of the lag-2 rank book; nothing here rewards or
penalises turnover. IC is a logged diagnostic only.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

import m1_common as C

LAG = C.EXECUTION_LAG_MIN
HOLDS = C.HOLDS
EPISODE_MINUTES = max(HOLDS) + 60       # a window must contain the longest hold
NAV_USD = 1e5
WINSOR_PCT = 1.0
BAND = 0.5
COSTS = ("flat6", "maker2", "depth")
AGGS = ("mean", "q25")                  # second_best / max are permanently banned
REWARD_FLOOR = C.REWARD_FLOOR


def cost_specs(nav_usd=NAV_USD):
    return {
        "flat6": ("flat", C.FEE_RATE + C.SLIPPAGE_RATE),
        "maker2": ("flat", 0.0002),
        "depth": ("depth", C.FEE_RATE, C.SLIPPAGE_RATE * 1e4, float(nav_usd)),
        "cost0": ("flat", 0.0),         # gross edge, reference only
    }


def winsorize_cs(fwd, pct=1.0):
    """Clip each minute's cross-section (axis 1) to [pct, 100-pct] percentiles.

    NaN (untradable) is preserved; minutes with < 10 live assets are left alone.
    Nearest-rank taken INWARD ('higher' for the bottom cut, 'lower' for the top):
    linear interpolation is wrong at this universe size (n=59: the 99th
    percentile sits 42% of the way from the second-largest value to the max)."""
    fwd = np.asarray(fwd, dtype=np.float64)
    n = np.isfinite(fwd).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = np.nanpercentile(fwd, pct, axis=1, method="higher")
        hi = np.nanpercentile(fwd, 100.0 - pct, axis=1, method="lower")
    ok = n >= 10
    lo = np.where(ok, lo, -np.inf)
    hi = np.where(ok, hi, np.inf)
    return np.clip(fwd, lo[:, None], hi[:, None])


def impact_bps(q_usd, buy, bid_cum, ask_cum):
    """Average slippage (bp) of filling q_usd against a piecewise-linear book.

    bid_cum / ask_cum: (..., 5) CUMULATIVE USD notional within 1..5% of mid.
    The offset rises linearly 0->100bp across the first band, 100->200bp across
    the second, and so on; anything past the 5% band is charged at 500bp.
    NaN depth -> NaN (caller falls back). q == 0 -> 0."""
    cum = np.where(buy[..., None], ask_cum, bid_cum).astype(np.float64)
    cum = np.maximum.accumulate(cum, axis=-1)
    prev = np.zeros_like(q_usd)
    area = np.zeros_like(q_usd)
    for k in range(5):
        ck = cum[..., k]
        seg = np.maximum(ck - prev, 1e-9)
        take = np.minimum(np.maximum(q_usd - prev, 0.0), seg)
        area = area + take * (100.0 * k + 50.0 * take / seg)
        prev = ck
    area = area + np.maximum(q_usd - prev, 0.0) * 500.0
    pos = q_usd > 0
    return np.where(pos, area / np.where(pos, q_usd, 1.0), 0.0)


def _ffill_held(held, tradable):
    """Same frozen-window semantics as C.settle_episode (see its docstring)."""
    if tradable.all():
        return held
    held_ff = np.where(tradable, held, 0.0)
    prev = held_ff[0].copy()
    for t in range(1, held.shape[0]):
        out = np.where(tradable[t], held[t], prev)
        held_ff[t] = out
        prev = np.where(tradable[t], held[t], prev)
    return held_ff


def settle_multi(held, fwd, fmask, frate, models, bid=None, ask=None):
    """C.settle_episode's accounting for many cost models in one pass.

    models: name -> ('flat', rate) | ('depth', fee, fallback_slip_bps, nav_usd)
    Returns ({name: dict(log_nav, cost, entry_cost)}, dict(gross, funding,
    turnover, n_trade_minutes)). With a flat cost it equals C.settle_episode
    exactly (p1_selftest.py)."""
    tradable = np.isfinite(fwd)
    held_ff = _ffill_held(held, tradable)
    d = np.diff(held_ff, axis=0, prepend=0.0)
    ad = np.abs(d)
    turn = ad.sum(axis=1)
    gross = np.nansum(held * np.nan_to_num(fwd), axis=1)
    fund = np.nansum(held * np.where(fmask, frate, 0.0), axis=1)
    out = {}
    idx = np.flatnonzero(turn > 0)          # impact only where something trades
    first = int(idx[0]) if idx.size else -1
    for name, spec in models.items():
        if spec[0] == "flat":
            cost_t = spec[1] * turn
        else:
            if bid is None or ask is None:
                raise ValueError("depth cost model needs bid/ask book arrays")
            _, fee, fallback_bp, nav = spec
            cost_t = np.zeros(held.shape[0])
            if idx.size:
                slip = impact_bps(ad[idx] * nav, d[idx] > 0, bid[idx], ask[idx])
                slip = np.where(np.isfinite(slip), slip, fallback_bp)
                cost_t[idx] = ((fee * 1e4 + slip) * 1e-4 * ad[idx]).sum(axis=1)
        net = gross - cost_t - fund
        with np.errstate(divide="ignore", invalid="ignore"):
            ln = float(np.sum(np.log1p(np.clip(net, -0.9, None))))
        out[name] = dict(log_nav=ln, cost=float(cost_t.sum()),
                         entry_cost=float(cost_t[first]) if first >= 0 else 0.0)
    tradable_prev =np.vstack([np.ones((1, held.shape[1]), dtype=bool), tradable[:-1]])
    resume_turn = float((ad * (tradable & ~tradable_prev)).sum())
    parts = dict(gross=float(gross.sum()), funding=float(fund.sum()),
                 turnover=float(turn.sum()), n_trade_minutes=int(idx.size),
                 resume_turn_share=resume_turn / max(float(turn.sum()), 1e-12))
    return out, parts


def cs_ic(sigT, fwdT, rows, min_live=10):
    """Mean cross-sectional Spearman IC over the given minute indices (diagnostic)."""
    vals = []
    for t in rows:
        s, f = sigT[t], fwdT[t]
        m = np.isfinite(s) & np.isfinite(f)
        if int(m.sum()) < min_live:
            continue
        rs = np.argsort(np.argsort(s[m], kind="stable")).astype(np.float64)
        rf = np.argsort(np.argsort(f[m], kind="stable")).astype(np.float64)
        rs -= rs.mean()
        rf -= rf.mean()
        dd = float(np.sqrt((rs * rs).sum() * (rf * rf).sum()))
        if dd > 0:
            vals.append(float((rs * rf).sum() / dd))
    return float(np.mean(vals)) if vals else float("nan")


def held_path(desired, hold_min, band, policy_mod):
    """World rule 0 + hysteresis band on the unchanged stateful policy.

    The policy is told every asset is eligible, so a missing 1-min bar no longer
    forces a whole-book retarget. settle_multi freezes the position through the
    same gap, so action and accounting agree. A position that is open when an
    asset never returns is carried at its last tradable price: no forward
    return, no turnover, no liquidation charge (no ghost P&L either way).
    `band` is the L1 distance below which the policy holds once the minimum
    hold has elapsed; it moves halfway between band and band+0.5 and switches
    fully beyond that. 0.5 is the contract default (the pre-existing band)."""
    contract = policy_mod.StatefulActionContract(
        minimum_hold_hours=int(hold_min), hold_l1_threshold=float(band),
        switch_l1_threshold=float(band) + 0.5)
    held = policy_mod.apply_stateful_action_policy(
        desired_weights=desired, eligible=np.ones(desired.shape, dtype=bool),
        contract=contract)[0]
    return np.asarray(held).reshape(desired.shape)


def prepare_shard(shard, winsor_pct=WINSOR_PCT, depth_dir=None):
    """Cache the per-month arrays every window of every formula shares.

    fwd_w is the 1/99-winsorised forward-return grid (A, T): the clip is a
    per-minute cross-sectional operation, so it is window-independent and is
    computed once per month rather than once per formula-window. If depth_dir
    is given the bookDepth cumulative arrays are attached, symbol-order and
    length checked against the shard; a missing file raises (no silent flat
    fallback)."""
    if "fwd_w" not in shard or shard.get("winsor_pct") != winsor_pct:
        shard["fwd_w"] = (winsorize_cs(shard["fwd"].T, winsor_pct).T
                          if winsor_pct else shard["fwd"])
        shard["winsor_pct"] = winsor_pct
    if depth_dir and "bid" not in shard:
        z = np.load(os.path.join(depth_dir, f"depth_{shard['month']}.npz"), allow_pickle=False)
        if [str(s) for s in z["symbols"]] != list(shard["symbols"]) or int(z["n_min"]) != int(shard["n_min"]):
            raise RuntimeError(f"depth {shard['month']}: symbol order / length differs from the shard")
        shard["bid"], shard["ask"] = z["bid"], z["ask"]
    return shard


def disjoint_starts(rng, n_min, k, episode=None):
    """k NON-overlapping episode starts inside one month shard.

    The month is cut into floor((n_min-LOOKBACK)/episode) slots; k distinct slots
    are drawn and each start is jittered by a non-decreasing offset inside the
    slack, so adjacent windows can never overlap. k is capped at the slot count
    (4 for a 30-day month with 7-day episodes)."""
    E = int(episode or EPISODE_MINUTES)
    L = C.LOOKBACK_MINUTES
    cap = (n_min - L) // E
    if cap < 1:
        raise ValueError(f"month of {n_min} min cannot hold one {E}-min episode plus lookback")
    k = min(int(k), cap)
    slack = (n_min - L) - cap * E
    slots = np.sort(rng.choice(cap, size=k, replace=False))
    jit = np.sort(rng.integers(0, slack + 1, size=k))
    return [int(L + s * E + j) for s, j in zip(slots, jit)]


def tile_starts(n_min, episode=None):
    """Deterministic back-to-back tiling of a month (DEV scoring: no draws, no choice)."""
    E = int(episode or EPISODE_MINUTES)
    return [C.LOOKBACK_MINUTES + i * E for i in range((n_min - C.LOOKBACK_MINUTES) // E)]


def aggregate(logs, how="mean"):
    arr = np.asarray(logs, dtype=np.float64)
    if how == "mean":
        return float(arr.mean())
    if how == "q25":
        return float(np.quantile(arr, 0.25))
    raise ValueError(f"aggregate {how!r} is not allowed; use one of {AGGS} "
                     "(second_best / max / median are permanently disabled)")


def bp_per_day(log_nav, minutes=None):
    return 1e4 * float(log_nav) / ((minutes or EPISODE_MINUTES) / 1440.0)

def window_signal(vm, body, sl, episode):
    """VM signal for one episode slice, lag-2 applied, shaped (T, A).

    sl carries LOOKBACK_MINUTES of history before the episode; the last
    `episode` columns are the episode. The signal computed at minute t is first
    tradable at t+LAG, so the first LAG minutes are NaN (nothing to trade)."""
    sig = vm.execute(list(body), sl).detach().cpu().numpy()          # (A, T_slice)
    sig = sig[:, -episode:]
    sigL = np.concatenate([np.full((sig.shape[0], LAG), np.nan), sig[:, :-LAG]], axis=1)
    return sigL.T


def book_diagnostics(held, fwd):
    """Diagnostics only, never a reward term: how much of a book is market exposure.

    net_exposure_mean  mean over minutes of sum_a held (dollar net; 0 for a rank book)
    gross_exposure_mean mean of sum_a |held|
    beta_mkt           cov(book return, equal-weight market return) / var(market), the
                       book's loading on the live-asset average return. Dollar-neutral
                       does not imply beta-neutral (long high-vol vs short low-vol is
                       net zero and still long the market), so this is reported next to
                       net exposure. NaN if the market return has no variance."""
    live = np.isfinite(fwd)
    f0 = np.where(live, fwd, 0.0)
    n = live.sum(axis=1)
    mkt = np.where(n > 0, f0.sum(axis=1) / np.maximum(n, 1), 0.0)
    book = (held * f0).sum(axis=1)
    vm_ = float(mkt.var())
    beta = float(np.cov(book, mkt, bias=True)[0, 1] / vm_) if vm_ > 0 else float("nan")
    return dict(net_exposure_mean=float(held.sum(axis=1).mean()),
                gross_exposure_mean=float(np.abs(held).sum(axis=1).mean()), beta_mkt=beta)


def score_window(sigT, shard, s0, episode, holds, policy_mod, *, costs, variants, band, nav_usd):
    """Net-dollar score of one lag-2 signal window at each hold.

    Returns {hold: {variant: ({cost: log_nav}, parts)}}. The ONLY place the
    desired book, the hold policy and the settlement meet: trainer, selection,
    elite re-check and DEV all reach it through eval_formula."""
    E = int(episode)
    el = shard["eligible"][:, s0:s0 + E].T
    fim = shard["fi"][:, s0:s0 + E].T
    frt = shard["fr"][:, s0:s0 + E].T
    specs = {c: cost_specs(nav_usd)[c] for c in costs}
    bid = ask = None
    if "depth" in costs:
        if "bid" not in shard:
            raise RuntimeError("cost 'depth' needs prepare_shard(shard, depth_dir=...) first")
        bid, ask = shard["bid"][s0:s0 + E], shard["ask"][s0:s0 + E]
    desired = C.rank_weights(sigT, el)
    out = {}
    for h in holds:
        held = held_path(desired, h, band, policy_mod)
        out[h] = {}
        for v in variants:
            if v == "win":
                if "fwd_w" not in shard:
                    raise RuntimeError("variant 'win' needs prepare_shard(shard) first")
                fw = shard["fwd_w"][:, s0:s0 + E].T
            else:
                fw = shard["fwd"][:, s0:s0 + E].T
            res, parts = settle_multi(held, fw, fim, frt, specs, bid, ask)
            parts = dict(parts, final_abs=float(np.abs(held[-1]).sum()),
                         **book_diagnostics(held, fw),
                         **{f"cost_{c}": res[c]["cost"] for c in costs})
            out[h][v] = ({c: res[c]["log_nav"] for c in costs}, parts)
    return out


def eval_formula(vm, body, holds, slices, shard, starts, policy_mod, invalid_cls, *,
                 costs=("flat6",), variants=("win",), band=BAND, nav_usd=NAV_USD,
                 episode=None, ic=True):
    """Score one formula body at one or more holds on every window.

    The VM runs once per window and is shared across holds, cost models and
    return variants. Returns
      per_hold[h][variant][cost] -> [log_nav per window]   (REWARD_FLOOR if non-finite)
      parts[h][variant]          -> [parts dict per window]
      live_frac, cs_ic2          -> diagnostics (cs_ic2 = lag-2 CS rank IC, never a reward)
    Raises invalid_cls only for an extreme zero-coverage window (probe live
    fraction < 0.1), as the M2 contract defines it."""
    E = int(episode or EPISODE_MINUTES)
    holds = [int(h) for h in holds]
    logs = {h: {v: {c: [] for c in costs} for v in variants} for h in holds}
    parts = {h: {v: [] for v in variants} for h in holds}
    live_fracs, ics = [], []
    for s0, sl in zip(starts, slices):
        sigT = window_signal(vm, body, sl, E)
        el = shard["eligible"][:, s0:s0 + E].T
        live = np.isfinite(sigT) & el
        probe_t = [t for t in [LAG] + list(range(E // 4, E, E // 4)) if t < E]
        pf = [float(live[t].sum() / max(el[t].sum(), 1)) for t in probe_t]
        if min(pf, default=0.0) < 0.1:
            raise invalid_cls(f"ZERO_COVERAGE_MIN_{min(pf, default=0.0):.2f}")
        live_fracs.append(float(np.mean(pf)))
        sc = score_window(sigT, shard, s0, E, holds, policy_mod,
                          costs=costs, variants=variants, band=band, nav_usd=nav_usd)
        for h in holds:
            for v in variants:
                lg, p = sc[h][v]
                for c in costs:
                    logs[h][v][c].append(lg[c] if np.isfinite(lg[c]) else REWARD_FLOOR)
                parts[h][v].append(p)
        if ic:
            ics.append(cs_ic(sigT, shard["fwd"][:, s0:s0 + E].T, range(LAG, E, 60)))
    ic_ok = [x for x in ics if np.isfinite(x)]
    return dict(per_hold=logs, parts=parts, live_frac=float(np.mean(live_fracs)),
                cs_ic2=float(np.mean(ic_ok)) if ic_ok else float("nan"))


def reward_of(res, hold, *, cost, variant="win", how="mean"):
    """The scalar the optimiser sees: aggregate of per-window net log NAV."""
    arr = res["per_hold"][int(hold)][variant][cost]
    return aggregate(arr, how), np.asarray(arr, dtype=np.float64)
