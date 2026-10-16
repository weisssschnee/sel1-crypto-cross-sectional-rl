"""V4 world (contract XS_BASIN_TIME_CONTRACT_V4_20261013, order V4_COMBO_V1_CR1). New file; m1_ruler,
m1_common and m2h_common are imported, never modified.

  grids      V4_TRAIN_GRID.npz (2020-11..2025-12, 632 coins) / V4_DEV2026_HOURLY_PANEL.npz, built by
             v4_panel.py. univ = point-in-time TOP100 among coins alive at that hour (incl. later
             delistings), taken from the panel, not recomputed.
  delisting  weight layer (contract s4): a coin whose last real hour L (quote volume > 0) lies inside
             a tile gets target weight 0 from row L on; hour L is made tradable with fwd 0 (there is no
             close[L+1]) so the exit at L pays flat6 through settle_multi; after L fwd stays NaN and
             the position stays 0. Without the layer the ruler carries the dead position at its last
             price (R.held_path docstring). The ruler is unchanged: held_path / settle_multi are called.
  halves     A/B by alternating calendar months from 2021-01 (A = 2021-01, 2021-03, ...): one 336 h
             tile per month starting 168 h after the month start and ending >= 168 h before the next
             month (the >= 168 h buffer on both sides). 30 A tiles, 30 B tiles on V4.
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

import m1_common as C
import m1_ruler as R
import m2h_common as M
from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod

ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem"
PANEL_DIR = os.path.join(ROOT, "runtime", "xs_v4_panel_v1")
GRID = os.path.join(PANEL_DIR, "V4_TRAIN_GRID.npz")
DEVZ = os.path.join(PANEL_DIR, "V4_DEV2026_HOURLY_PANEL.npz")
E = M.EPISODE
HOLD = 168
DAYS = E / 24.0
TRAIN_FIRST = pd.Timestamp("2021-01-01")
DEV_FIRST = pd.Timestamp("2026-01-01")
SLACK_H = 24
MIN_COINS = 20


def bpd(x):
    return R.bp_per_day(x, E * 60)


def load(path=GRID):
    z = np.load(path, allow_pickle=False)
    ts = pd.DatetimeIndex(z["hours"].astype("datetime64[ns]"))
    if ts.max() >= M.OOS_FIRST:
        raise SystemExit("grid reaches OOS")
    g = {c: np.array(z[c], dtype=np.float64) for c in M.PANEL_COLS}
    return dict(syms=[str(s) for s in z["symbols"]], ts=ts, g=g, univ=np.array(z["univ"], dtype=bool),
                lidx=np.array(z["last_real_idx"]))


def world(w, feats=True):
    """Terminals as M.build_world builds them, the panel's TOP100, and the ruler shard + layer arrays."""
    g = w["g"]
    if feats:
        bw = M.build_world(g)                         # its own univ is replaced by the panel's
        w["feats"] = bw["feats"]
        sh = bw["shard"]
    else:
        fr_raw = g["funding_rate"]
        fwd = C.forward_returns(g["trade_close"], None)
        sh = dict(fwd=fwd, eligible=np.isfinite(fwd), fi=np.isfinite(fr_raw),
                  fr=np.where(np.isfinite(fr_raw), fr_raw, 0.0))
        R.prepare_shard(sh, R.WINSOR_PCT)
    T = sh["fwd"].shape[1]
    dead = np.flatnonzero((w["lidx"] >= 0) & (w["lidx"] < T - 1 - SLACK_H))
    w["dead"] = {int(a): int(w["lidx"][a]) for a in dead}
    install_layer(sh, w["dead"])
    w["shard"] = sh
    return w


def install_layer(sh, dead):
    """fwd_wx / fwd_x: fwd_w / fwd with [a, L] = 0 and [a, > L] = NaN for every dead coin."""
    for k_src, k_dst in (("fwd_w", "fwd_wx"), ("fwd", "fwd_x")):
        x = sh[k_src].copy()
        for a, L in dead.items():
            x[a, L] = 0.0
            x[a, L + 1:] = np.nan
        sh[k_dst] = x
    sh["dead"] = dead
    return sh


def held_of(sig, sh, s0, hold=HOLD, layer=True):
    el = sh["eligible"][:, s0:s0 + E].T
    held = R.held_path(C.rank_weights(sig, el), hold, R.BAND, policy_mod)
    if layer:
        for a, L in sh["dead"].items():
            if s0 <= L < s0 + E:
                held[L - s0:, a] = 0.0
    return held


def settle(held, sh, s0, layer=True, raw=False, costs=M.COSTS):
    key = ("fwd_x" if raw else "fwd_wx") if layer else ("fwd" if raw else "fwd_w")
    fw = sh[key][:, s0:s0 + E].T
    specs = {c: R.cost_specs(R.NAV_USD)[c] for c in costs}
    res, parts = R.settle_multi(held, fw, sh["fi"][:, s0:s0 + E].T, sh["fr"][:, s0:s0 + E].T, specs)
    parts.update(R.book_diagnostics(held, fw))
    return {c: res[c]["log_nav"] for c in costs}, parts


def tile(sig, sh, s0, layer=True, keep=False):
    """One tile through rank_weights -> held_path -> layer -> settle_multi. bp/day fields."""
    held = held_of(sig, sh, s0, layer=layer)
    lg, p = settle(held, sh, s0, layer=layer)
    out = dict(net=bpd(lg["flat6"]), gross=bpd(lg["cost0"]), turn=p["turnover"] / DAYS,
               beta=p["beta_mkt"], ln_net=lg["flat6"], ln_gross=lg["cost0"])
    if keep:
        out["_held"] = held
    return out


# --------------------------------------------------------------------------- tiles
def ab_tiles(ts, first=TRAIN_FIRST, last=pd.Timestamp("2025-12-01")):
    out = dict(A=[], B=[])
    for k, m in enumerate(pd.date_range(first, last, freq="MS")):
        s0 = int(ts.get_loc(m + pd.Timedelta(hours=168)))
        nxt = int(ts.get_loc(m + pd.offsets.MonthBegin(1))) if m + pd.offsets.MonthBegin(1) <= ts[-1] else len(ts)
        assert s0 + E + 168 <= nxt and s0 >= M.LOOKBACK, m
        out["A" if k % 2 == 0 else "B"].append(s0)
    return out


def seq_tiles(ts, first=TRAIN_FIRST):
    """Back-to-back 336 h tiles from `first` (COMBO_V1 / DEV)."""
    return M.tile_starts(len(ts), first=int(ts.get_loc(first)))

# --------------------------------------------------------------------------- composite / books
def zrank_rows(x, min_coins=MIN_COINS):
    """(n, A) -> per row average rank over finite entries, standardised; rows < min_coins -> NaN.
    combo_v0.zrank(x, mask) == zrank_rows(where(mask & finite, x, nan).T).T."""
    r = pd.DataFrame(x).rank(axis=1).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(r, axis=1, keepdims=True)
        sd = np.nanstd(r, axis=1, keepdims=True)
        z = np.where(sd > 0, (r - mu) / np.where(sd > 0, sd, 1.0), np.nan)
    z[np.isfinite(r).sum(axis=1) < min_coins] = np.nan
    return z


def tile_rows(x, s0):
    """(A, T) absolute grid -> (E, A) tile rows: row t = signal hour s0 - LAG + t, first LAG rows NaN
    (M.window_signal's alignment; the TOP100 mask is applied by the caller at the signal hour)."""
    L = np.full((E, x.shape[0]), np.nan)
    L[M.LAG:] = x[:, s0:s0 + E - M.LAG].T
    return L


def book_sig(sig, sh, s0, book, pool=None):
    if book[0] == "full":
        return sig
    import m2h_conc_weights as W
    el = sh["eligible"][:, s0:s0 + E].T
    return W.conc_signal(sig, el, np.ones(sig.shape[1], bool) if pool is None else pool, book[1])


def bk(book):
    return "conc_k%d" % book[1] if book[0] == "conc" else "full"


BOOKS = (("conc", 10), ("conc", 5), ("conc", 20), ("full",))
K10 = ("conc", 10)


def books(sig, sh, s0, which=BOOKS, pool=None, keep=False):
    return {bk(b): tile(book_sig(sig, sh, s0, b, pool), sh, s0, keep=keep and b == K10) for b in which}


# --------------------------------------------------------------------------- OKX column (DEV)
def okx_shard(dev_syms, dev_hours, dead, npz):
    """b5_verdict.okx_shard on the V4 DEV grid, plus the delisting layer."""
    z = np.load(npz, allow_pickle=False)
    oh = pd.DatetimeIndex(z["hours"].astype("datetime64[ns]"))
    if oh.max() >= M.OOS_FIRST:
        raise SystemExit("OKX grid reaches OOS")
    di = pd.Index(list(dev_syms))
    close = np.full((len(dev_syms), len(dev_hours)), np.nan)
    fr_raw = np.full_like(close, np.nan)
    col = pd.DatetimeIndex(dev_hours).get_indexer(oh)
    ok = col >= 0
    rows = di.get_indexer(list(z["symbols"]))
    for j, r in enumerate(rows):
        if r >= 0:
            close[r, col[ok]] = z["close"][j][ok]
            fr_raw[r, col[ok]] = z["funding_rate"][j][ok]
    fwd = C.forward_returns(close, None)
    sh = dict(fwd=fwd, eligible=np.isfinite(fwd), fi=np.isfinite(fr_raw), fr=np.where(np.isfinite(fr_raw), fr_raw, 0.0))
    R.prepare_shard(sh, R.WINSOR_PCT)
    install_layer(sh, dead)
    return sh, dict(okx_symbols=int(len(z["symbols"])), mapped=int((rows >= 0).sum()),
                    okx_hours=[str(oh.min()), str(oh.max())])


def okx_settle(helds, starts, osh):
    out = dict(net=[], gross=[])
    for held, s0 in zip(helds, starts):
        lg, _ = settle(held, osh, s0)
        out["net"].append(bpd(lg["flat6"]))
        out["gross"].append(bpd(lg["cost0"]))
    return {k: round(float(np.mean(v)), 4) for k, v in out.items()}


# --------------------------------------------------------------------------- selftest
def selftest(w, seed=7):
    """(1) layer off == R.score_window exactly; (2) layer on == off on tiles with no death inside;
    (3) negative test: a long position on a coin that dies inside a tile is 0 from L on with the layer,
    and is still open after L without it (the ruler carries it at its last price)."""
    sh, univ, ts = w["shard"], w["univ"], w["ts"]
    rng = np.random.default_rng(seed)
    starts = seq_tiles(ts)
    inside = {s0: [a for a, L in w["dead"].items() if s0 <= L < s0 + E] for s0 in starts}
    clean = [s0 for s0 in starts if not inside[s0]][:3]
    out = dict(eq_score_window=0.0, eq_layer_clean=0.0)
    for s0 in clean:
        sig = np.where(tile_rows(univ.astype(float), s0) > 0, rng.standard_normal((E, univ.shape[0])), np.nan)
        a = settle(held_of(sig, sh, s0, layer=False), sh, s0, layer=False)[0]
        b = R.score_window(sig, sh, s0, E, [HOLD], policy_mod, costs=M.COSTS, variants=("win",), band=R.BAND,
                           nav_usd=R.NAV_USD)[HOLD]["win"][0]
        c = settle(held_of(sig, sh, s0), sh, s0)[0]
        out["eq_score_window"] = max(out["eq_score_window"], max(abs(a[k] - b[k]) for k in a))
        out["eq_layer_clean"] = max(out["eq_layer_clean"], max(abs(a[k] - c[k]) for k in a))
    cases = []
    for s0 in starts:
        for a in inside[s0]:
            L = w["dead"][a]
            if L - s0 < 48:
                continue
            sig = np.where(tile_rows(univ.astype(float), s0) > 0, rng.standard_normal((E, univ.shape[0])), np.nan)
            sig[:L - s0 + 1, a] = 10.0                     # top rank while alive
            sig[L - s0 + 1:, a] = np.nan
            h0, h1 = held_of(sig, sh, s0, layer=False), held_of(sig, sh, s0)
            _, p0 = settle(h0, sh, s0, layer=False)
            _, p1 = settle(h1, sh, s0)
            r = L - s0
            # settle_multi's frozen book without the layer: the position at L-1 is carried to the end
            ff0 = R._ffill_held(h0, np.isfinite(sh["fwd_w"][:, s0:s0 + E].T))
            pos = float(ff0[r - 1, a])
            # every coin that dies inside this tile exits once at its own L (several can share a tile)
            exp_turn = float(sum(abs(ff0[w["dead"][b] - s0 - 1, b]) for b in inside[s0]
                                 if w["dead"][b] - s0 >= 1))
            cases.append(dict(symbol=w["syms"][a], L=str(ts[L]), pos_before_L=pos,
                              deaths_in_tile=len(inside[s0]), expected_exit_turnover=exp_turn,
                              layer_zero_from_L=bool(np.all(h1[r:, a] == 0.0)),
                              exit_tradable=bool(np.isfinite(sh["fwd_wx"][a, L])),
                              exit_turnover=float(p1["turnover"] - p0["turnover"]),
                              no_layer_stuck_at_end=float(ff0[-1, a])))
            if len(cases) >= 5:
                break
        if len(cases) >= 5:
            break
    # with the layer the exit at L is one extra trade of exactly |pos| (charged flat6 by settle_multi);
    # without it the position is frozen to the tile end and never pays an exit
    neg_ok = bool(cases) and all(c["layer_zero_from_L"] and c["exit_tradable"] and c["pos_before_L"] > 0
                                 and abs(c["exit_turnover"] - c["expected_exit_turnover"]) < 1e-12
                                 and c["no_layer_stuck_at_end"] == c["pos_before_L"] for c in cases)
    out.update(clean_tiles=[str(ts[s]) for s in clean], negative=cases, negative_ok=neg_ok,
               ok=bool(out["eq_score_window"] == 0.0 and out["eq_layer_clean"] == 0.0 and neg_ok))
    return out
