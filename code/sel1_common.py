"""XS_SEL1 shared world: sparse cross-sectional selection book (order XS_SEL1 s1-s2; ADR-0012; contract V4). New
file; v4_world, m1_ruler, m1_common, m2h_common, m2h_dense_common and the B5 dbg_memo trainer are imported, never
modified.

  grid      V4 TRAIN (v4_world.load + world(feats=True)): 632 coins x 45288 h, panel TOP100, delisting layer.
  signal    the formula runs per calendar-year chunk (VM xs_tscs_vm_v2, the 11 terminals of m2h_common, float32
            on the GPU): the chunk's own hours plus the 2160 h before them (2021: from the grid start,
            2020-11-01), so every hour sees >= 2160 h of history (m2h/CR1 tiles: 672 h); one VM call on all
            45288 h blew past the 16 GB card (34 GiB peak on one formula). Value at hour t uses data up to t.
            Kept only where the coin is in TOP100 at t, is not BTC and has a finite beta at t.
  action    formula + hold {4,24,72,168} + state switch {all, up_lowvol, up_highvol, down_lowvol, down_highvol}
            (BTC 720 h log return > 0 / 720 h vol of hourly log returns > 0.005196, ENR1 rule) + direction
            {both, long, short}.
  book      per calendar month window (2021-01..2025-12; A = odd months, B = even, V4 contract), starting flat.
            Row r of the window uses the signal of hour r-1 (lag 1); the first row holds nothing. At hour r-1:
            no position if the state switch does not match or fewer than 20 coins have a signal; otherwise the
            K=5 highest get +, the 5 lowest get - (stable sort, ties by coin index), equal weight: both = +-1/10,
            long = +1/5, short = -1/5. Hold semantics = m1_ruler.held_path's stateful policy (minimum hold h,
            band 0.5), run as apply_stateful_action_policy_batch over the month windows (bit-equality with
            held_path checked in sel1_bench), then the v4_world delisting rule (weight 0 from the last real hour
            L on). BTC leg per row = -sum_i held_i * beta_i, beta_i = trailing 720 h OLS of hourly log returns
            on BTC (>= 360) at the signal hour of coin i's last weight change (ENR1b: beta of the trigger hour).
  settle    m1_ruler.settle_multi on the columns [held coins..., BTC] (columns never held contribute nothing, so
            this equals the full book), fwd = m1_ruler.prepare_shard(winsor_pct=0) + v4_world.install_layer
            (no winsor), flat6 / cost0, funding, Rule 0. Held and fwd column counts asserted equal.
  reward    A half: sum over A months of cost0 log NAV / A days, bp/day (funding paid, no trading cost).
  null      the formula's signal shifted per coin circularly by a fixed offset (uniform in [720, T-720], seed
            NULL_SEED) before the TOP100 / beta mask; returns untouched.
  plant     pc: fwd[i,t] += alpha * zp[i,t-1], zp = D.plant_signal (TOP100 rank z of
            DELTA(TS_MEAN(log_quote_volume,168),168)); alpha calibrated in sel1_calib.py (SEL1_ALPHA.json).
            pcnull: the NULL world (same offsets) plus a plant; zp = the plant formula's NULL-shifted signal
            (plant_signal_shifted), so the plant is what a formula sees in that world; alpha explicit.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m2h_b5_trainer_dbg_memo as TB      # first: sys.path for m1_stage1 / okx_alpha_discovery / legacy
import m2h_common as M
import m2h_dense_common as D
import m1_ruler as R
import v4_world as X
from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod

K = 5
HOLDS = (4, 24, 72, 168)
STATES = ("all", "up_lowvol", "up_highvol", "down_lowvol", "down_highvol")
DIRS = ("both", "long", "short")
MIN_COINS = 20
COVER_MIN = 0.5
BETA_WIN, BETA_MIN, STATE_WIN, THR = 720, 360, 720, 0.005196
NULL_SEED = 20261015
POLICY_WIDTH = 256
CHUNK_LOOKBACK = 2160
MODES = ("real", "null", "pc", "pcnull")
A_ROOT = os.path.join(X.ROOT, "artifacts", "m2h_v1")
OUT_ROOT = os.path.join(A_ROOT, "XS_SEL1")
ALPHA_JSON = os.path.join(OUT_ROOT, "SEL1_ALPHA.json")
TRAIN_FIRST = pd.Timestamp("2021-01-01")
TRAIN_END = pd.Timestamp("2026-01-01")


class Invalid(Exception):
    pass


def sha16a(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def sha16f(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def no_winsor_shard(sh, dead, fwd=None):
    s0 = dict(fwd=sh["fwd"] if fwd is None else fwd, eligible=sh["eligible"], fi=sh["fi"], fr=sh["fr"])
    R.prepare_shard(s0, 0)
    X.install_layer(s0, dead)
    return s0


def rolling_beta(close, bi):
    lr = np.diff(np.log(close), axis=1, prepend=np.nan)
    x = lr[bi]
    msk = np.isfinite(lr) & np.isfinite(x)[None, :]
    X0, Y0 = np.where(msk, x[None, :], 0.0), np.where(msk, lr, 0.0)

    def rs(z):
        c = np.cumsum(z, axis=1)
        c[:, BETA_WIN:] -= c[:, :-BETA_WIN].copy()
        return c
    nn, sx, sy, sxy, sxx = rs(msk.astype(np.float64)), rs(X0), rs(Y0), rs(X0 * Y0), rs(X0 * X0)
    with np.errstate(all="ignore"):
        var = sxx - sx * sx / nn
        return np.where((nn >= BETA_MIN) & (var > 0), (sxy - sx * sy / nn) / var, np.nan)


def btc_state(close_btc):
    btc = np.log(close_btc)
    ret30 = pd.Series(btc).diff(STATE_WIN).to_numpy()
    vol30 = pd.Series(np.diff(btc, prepend=np.nan)).rolling(STATE_WIN, min_periods=STATE_WIN // 2).std().to_numpy()
    ok = np.isfinite(ret30) & np.isfinite(vol30)
    st = np.where(ret30 > 0, np.where(vol30 > THR, 1, 0), np.where(vol30 > THR, 3, 2))
    return np.where(ok, st, -1).astype(np.int8)


def months(ts):
    out = []
    for k, m in enumerate(pd.date_range(TRAIN_FIRST, TRAIN_END - pd.Timedelta(hours=1), freq="MS")):
        m0 = int(ts.get_loc(m))
        nx = m + pd.offsets.MonthBegin(1)
        m1 = int(ts.get_loc(nx)) if nx <= ts[-1] else len(ts)
        out.append(dict(m0=m0, m1=m1, half="A" if k % 2 == 0 else "B", year=int(m.year), month=str(m)[:7]))
    return out


def null_offsets(A, T, seed=NULL_SEED):
    rng = np.random.default_rng(seed)
    return rng.integers(720, T - 720 + 1, size=A)


def year_chunks(ts, first=TRAIN_FIRST, last=TRAIN_END):
    out = []
    for y in range(first.year, last.year):
        o0 = int(ts.searchsorted(pd.Timestamp("%d-01-01" % y))) - 1
        o1 = int(ts.searchsorted(pd.Timestamp("%d-01-01" % (y + 1)))) - 1
        o1 = len(ts) if y == last.year - 1 else o1
        out.append((max(0, o0 - CHUNK_LOOKBACK), o0, o1))
    return out


def load_alpha():
    return float(json.load(open(ALPHA_JSON, encoding="utf-8"))["alpha"])


def plant_add(zp):
    """Return add for a plant signal: fwd[:, t] += alpha * zp[:, t-1] (the signal hour before the return hour)."""
    add = np.zeros_like(zp)
    add[:, 1:] = np.nan_to_num(zp[:, :-1], nan=0.0)
    return add


def plant_signal_shifted(vm, feats, univ, device, offsets):
    """D.plant_signal as the NULL world sees the plant formula: its raw value rolled per coin by the NULL offsets
    (the same np.roll as mask_signal), then the same TOP100 cross-sectional rank, demeaned, unit std per hour."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(feats)).to(device=device, dtype=torch.float64)
    raw = vm.execute(D.plant_body(vm), x).detach().cpu().numpy()
    del x
    raw = np.stack([np.roll(raw[a], int(o)) for a, o in enumerate(offsets)])
    v = np.where(univ & np.isfinite(raw), raw, np.nan)
    rk = pd.DataFrame(v.T).rank(axis=1).to_numpy()
    n = np.isfinite(rk).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rk = rk - np.nanmean(rk, axis=1, keepdims=True)
        sd = np.nanstd(rk, axis=1, keepdims=True)
    s = np.where((n[:, None] >= D.PLANT_MIN_COINS) & (sd > 0), rk / np.where(sd > 0, sd, 1.0), np.nan)
    return s.T


# --------------------------------------------------------------------------- world
def build(mode, device, vm, alpha=None, log=print, keep_base=False):
    import torch
    if mode not in MODES:
        raise KeyError(mode)
    t0 = time.time()
    w = X.load(X.GRID)
    X.world(w, feats=True)
    ts, univ, syms = w["ts"], w["univ"], w["syms"]
    A, T = univ.shape
    bi = syms.index("BTCUSDT")
    close = w["g"]["trade_close"]
    beta = rolling_beta(close, bi)
    state = btc_state(close[bi])
    del w["g"]
    feats = w.pop("feats")
    base = no_winsor_shard(w["shard"], w["dead"])
    if not np.array_equal(base["fwd_wx"], w["shard"]["fwd_x"], equal_nan=True):
        raise RuntimeError("winsor 0 path != fwd_x")
    zp = None
    offsets = null_offsets(A, T) if mode in ("null", "pcnull") else None
    if mode == "pc" or (keep_base and offsets is None):
        zp = D.plant_signal(vm, feats, univ, device)
    elif mode == "pcnull" or keep_base:
        zp = plant_signal_shifted(vm, feats, univ, device, offsets)
    if mode in ("pc", "pcnull"):
        if mode == "pcnull" and alpha is None:
            raise ValueError("pcnull needs an explicit alpha (SEL1_ALPHA.json is the real-world plant)")
        a = load_alpha() if alpha is None else float(alpha)
        sh = no_winsor_shard(base, w["dead"], fwd=base["fwd"] + a * plant_add(zp))
    else:
        a = None
        sh = base
    mask = univ & np.isfinite(beta)
    mask[bi] = False
    Larr = np.full(A, T + 10, np.int64)
    for c, L in w["dead"].items():
        Larr[c] = L
    ctx = dict(mode=mode, ts=ts, syms=syms, univ=univ, mask=mask, bi=bi, beta=beta, state=state, Larr=Larr,
               dead=w["dead"], months=months(ts), shard=sh, alpha=a, zp=zp,
               feats_gpu=torch.from_numpy(np.ascontiguousarray(feats.astype(np.float32))).to(device),
               offsets=offsets)
    if keep_base:
        ctx["base"] = base
    del feats
    ctx["cover_rows"] = np.concatenate([np.arange(m["m0"], m["m1"]) for m in ctx["months"]])
    ctx["chunks"] = year_chunks(ts)
    log("SEL1 world %s: %d coins x %d h, %d months, %d delistings, alpha %s, %.0fs"
        % (mode, A, T, len(ctx["months"]), len(w["dead"]), a, time.time() - t0))
    return ctx


def raw_signal(ctx, vm, body):
    """GPU: the formula per calendar-year chunk (output hours of the year, shifted 1 h earlier so January's
    first row has its signal), each chunk run on its own hours plus the CHUNK_LOOKBACK hours before; (A, T)
    float64, NaN outside the chunks."""
    import torch
    out = np.full(ctx["mask"].shape, np.nan)
    ex = ctx.get("cse")                                         # sel1_cse.CachedExec or None (plain vm.execute)
    with torch.no_grad():
        for k, (in0, o0, o1) in enumerate(ctx["chunks"]):
            f = ctx["feats_gpu"][:, :, in0:o1]
            s = (vm.execute(list(body), f) if ex is None else ex.execute(list(body), f, k)).detach().cpu().numpy()
            out[:, o0:o1] = s[:, o0 - in0:]
    return out


def mask_signal(ctx, raw):
    """NULL shift (null world only), then TOP100 / not BTC / finite beta; coverage check."""
    s = raw
    if ctx["offsets"] is not None:
        s = np.full_like(raw, np.nan)
        T = raw.shape[1]
        for a, o in enumerate(ctx["offsets"]):
            s[a] = np.roll(raw[a], int(o))
    s = np.where(ctx["mask"] & np.isfinite(s), s, np.nan)
    n = np.isfinite(s[:, ctx["cover_rows"]]).sum(axis=0)
    cov = float((n >= MIN_COINS).mean())
    if cov < COVER_MIN:
        raise Invalid("LOW_COVERAGE_%.2f" % cov)
    return s


def signal(ctx, vm, body):
    return mask_signal(ctx, raw_signal(ctx, vm, body))


# --------------------------------------------------------------------------- book
def select_month(sig, m0, m1):
    """Signal hours m0-1 .. m1-2 for rows m0 .. m1-1 (lag 1); row 0 holds nothing. -> (hi, lo, n): the K highest
    / lowest coin indices per row (stable sort, ties by coin index) and the number of coins with a signal."""
    S = sig[:, m0 - 1:m1 - 1].T.copy()
    S[0] = np.nan
    fin = np.isfinite(S)
    lo = np.argsort(np.where(fin, S, np.inf), axis=1, kind="stable")[:, :K]
    hi = np.argsort(np.where(fin, -S, np.inf), axis=1, kind="stable")[:, :K]
    return hi.astype(np.int16), lo.astype(np.int16), fin.sum(axis=1).astype(np.int16)


def select_months(sig, ctx, halves=("A",)):
    return [select_month(sig, m["m0"], m["m1"]) for m in ctx["months"] if m["half"] in halves]


def compact_desired(sel, ctx, m0, m1, st, dr):
    """(E, n_cols) desired weights on the window's selected coins (sorted coin indices) and those indices."""
    hi, lo, n = sel
    ok = n >= MIN_COINS
    if st:
        ok = ok & (ctx["state"][m0 - 1:m1 - 1] == st - 1)
    rows = np.flatnonzero(ok)
    wl, ws = {0: (0.5 / K, 0.5 / K), 1: (1.0 / K, 0.0), 2: (0.0, 1.0 / K)}[dr]
    parts = ([hi[rows]] if wl else []) + ([lo[rows]] if ws else [])
    cols = np.unique(np.concatenate([p.ravel() for p in parts]).astype(np.int64)) if rows.size else np.zeros(0, np.int64)
    des = np.zeros((m1 - m0, cols.size))
    if rows.size:
        if wl:
            des[rows[:, None], np.searchsorted(cols, hi[rows])] = wl
        if ws:
            des[rows[:, None], np.searchsorted(cols, lo[rows])] = -ws
    return des, cols


def desired_window(sig, ctx, m0, m1, st, dr):
    """Full-width (E, A) desired weights (tests); production uses compact_desired."""
    des_c, cols = compact_desired(select_month(sig, m0, m1), ctx, m0, m1, st, dr)
    des = np.zeros((m1 - m0, ctx["mask"].shape[0]))
    des[:, cols] = des_c
    return des


def contract(h):
    """m1_ruler.held_path's contract (band 0.5, every asset eligible) with minimum hold h - 1: the controller
    then keeps a fresh position on exactly h rows (entry row + h - 1 holds), the ENR1b event length. (m2h's
    H{h} passes minimum hold h and so holds h + 1 rows.)"""
    return policy_mod.StatefulActionContract(minimum_hold_hours=int(h) - 1, hold_l1_threshold=float(R.BAND),
                                             switch_l1_threshold=float(R.BAND) + 0.5)


def width(n):
    """Fixed column width of a window's policy array: the L1 sums then reduce over the same number of
    elements on every code path (numpy's pairwise summation depends on the length)."""
    w = POLICY_WIDTH
    while w < n:
        w *= 2
    return w


def held_compact(pairs, h):
    """The stateful policy on every month window at once (apply_stateful_action_policy_batch: same float64
    controller, candidate axis vectorised). pairs = [(desired (E, n_cols), cols)]; every window is zero-padded to
    one fixed width; padding rows / columns are cut off again. -> list of (held (E, n_cols), cols)."""
    C = max((c.size for _, c in pairs), default=0)
    if C == 0:
        return [(np.zeros((d.shape[0], 0)), c) for d, c in pairs]
    W = width(C)
    Emax = max(d.shape[0] for d, _ in pairs)
    arr = np.zeros((len(pairs), Emax, W))
    for i, (d, c) in enumerate(pairs):
        arr[i, :d.shape[0], :c.size] = d
    hb = policy_mod.apply_stateful_action_policy_batch(arr, eligible=True, contract=contract(h))[0]
    return [(hb[i, :d.shape[0], :c.size].copy(), c) for i, (d, c) in enumerate(pairs)]


def held_months(des_list, h):
    """held_compact on full-width (E, A) desired weights (tests)."""
    cols = [np.flatnonzero(np.abs(d).sum(axis=0) > 0) for d in des_list]
    return held_compact([(d[:, c], c) for d, c in zip(des_list, cols)], h)


def settle_window(held, cols, ctx, m0, m1, sh=None, keep=False):
    sh = ctx["shard"] if sh is None else sh
    E = m1 - m0
    if cols.size == 0:
        return dict(net=0.0, cost0=0.0, turn=0.0, n_pos=0.0, days=E / 24.0)
    held = held.copy()
    for j, a in enumerate(cols):
        L = int(ctx["Larr"][a])
        if m0 <= L < m1:
            held[L - m0:, j] = 0.0
    # hedge ratio frozen at the signal hour of the coin's last weight change (ENR1b: beta of the trigger hour)
    rr = np.arange(E)[:, None]
    chg = np.vstack([held[:1] != 0, np.diff(held, axis=0) != 0])
    last = np.maximum.accumulate(np.where(chg, rr, -1), axis=0)
    bfull = ctx["beta"][cols, m0 - 1:m1 - 1].T
    b = np.nan_to_num(np.where(last >= 0, bfull[last.clip(0), np.arange(cols.size)[None, :]], 0.0))
    hb = -(held * b).sum(axis=1, keepdims=True)
    held2 = np.concatenate([held, hb], axis=1)
    c2 = np.r_[cols, ctx["bi"]]
    fw = sh["fwd_wx"][c2, m0:m1].T
    fi, fr = sh["fi"][c2, m0:m1].T, sh["fr"][c2, m0:m1].T
    assert held2.shape[1] == fw.shape[1] == fi.shape[1] == fr.shape[1], "held / fwd column count differs"
    specs = {c: R.cost_specs(R.NAV_USD)[c] for c in ("flat6", "cost0")}
    res, parts = R.settle_multi(held2, fw, fi, fr, specs)
    out = dict(net=res["flat6"]["log_nav"], cost0=res["cost0"]["log_nav"], turn=parts["turnover"],
               n_pos=float((np.abs(held) > 0).sum(axis=1).mean()), days=E / 24.0)
    if keep:
        out.update(held=held2, cols=c2, fw=fw, fi=fi, fr=fr)
    return out


def score_sel(sels, ctx, h, st, dr, halves=("A",), keep=False):
    """Per month of the requested halves (sels = select_months(sig, ctx, halves), same order):
    dict(month, half, year, net, cost0, turn, n_pos, days)."""
    ms = [m for m in ctx["months"] if m["half"] in halves]
    if len(ms) != len(sels):
        raise ValueError("selection / month count differs")
    pairs = [compact_desired(s, ctx, m["m0"], m["m1"], st, dr) for s, m in zip(sels, ms)]
    out = []
    for m, (held, cols) in zip(ms, held_compact(pairs, h)):
        r = settle_window(held, cols, ctx, m["m0"], m["m1"], keep=keep)
        r.update(month=m["month"], half=m["half"], year=m["year"])
        out.append(r)
    return out


def score(sig, ctx, h, st, dr, halves=("A",), keep=False):
    return score_sel(select_months(sig, ctx, halves), ctx, h, st, dr, halves, keep)


def bpd(rows, key="cost0"):
    d = sum(r["days"] for r in rows)
    return 1e4 * sum(r[key] for r in rows) / d if d else float("nan")


def tstat(v):
    v = np.asarray(v, dtype=np.float64)
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return float(v.mean() / (sd / np.sqrt(len(v)))) if sd > 0 else float("nan")


REWARDS = ("mean", "t", "t_dd")


def max_dd(rows):
    """Largest drawdown of the cost0 log NAV over the rows' months in their order, marked at month ends from 0
    (log units; a path inside a month is not seen)."""
    c = np.cumsum([r["cost0"] for r in rows])
    if c.size == 0:
        return 0.0
    return float((np.maximum.accumulate(np.r_[0.0, c])[1:] - c).max())


def reward(rows, kind="mean", dd_max=0.0):
    """Candidate score over the rows' months (the trainer's reward and the combo rule of the launch checks).
    mean: bpd(rows) (cost0 bp/day, the original reward); t: tstat of the monthly cost0 log NAV (a month that holds
    nothing is a 0 and counts; SEL1_REWARD_RULE); t_dd: t, or 0.0 when max_dd(rows) > dd_max (log NAV)."""
    if kind == "mean":
        return bpd(rows)
    c = [r["cost0"] for r in rows]
    if not np.isfinite(c).all():
        return float("nan")                           # NONFINITE in the trainer, as under mean
    t = tstat(c)
    t = 0.0 if np.isnan(t) else t                     # sd 0 (a book that never trades): no evidence either way
    if kind == "t":
        return t
    if kind == "t_dd":
        if not dd_max > 0:
            raise ValueError("t_dd needs dd_max > 0")
        return 0.0 if max_dd(rows) > dd_max else t
    raise KeyError(kind)


def summary(rows):
    c0 = [1e4 * r["cost0"] / r["days"] for r in rows]
    return dict(cost0_bpd=round(bpd(rows), 3), net_bpd=round(bpd(rows, "net"), 3),
                t_cost0=round(tstat(c0), 3), turn_day=round(sum(r["turn"] for r in rows) / sum(r["days"] for r in rows), 4),
                n_pos=round(float(np.mean([r["n_pos"] for r in rows])), 3), months=len(rows))


# --------------------------------------------------------------------------- events (gate 3)
def events(rows):
    """Holding spells per coin column inside each kept window; contribution = sum over the spell of
    held_i * (f_i - beta-weighted BTC f) - held_i * fund_i + its hedge's funding (arithmetic, cost0)."""
    ev = []
    for r in rows:
        held, fw, fi, fr = r["held"], np.nan_to_num(r["fw"]), r["fi"], r["fr"]
        hb = held[:, -1]
        fund = np.where(fi, fr, 0.0)
        tot = np.abs(held[:, :-1]).sum(axis=1)
        for j in range(held.shape[1] - 1):
            hj = held[:, j]
            on = hj != 0
            if not on.any():
                continue
            share = np.where(tot > 0, np.abs(hj) / np.where(tot > 0, tot, 1.0), 0.0)
            leg = hj * (fw[:, j] - fund[:, j]) + share * hb * (fw[:, -1] - fund[:, -1])
            edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
            for s0, s1 in zip(edges[::2], edges[1::2]):
                ev.append(float(leg[s0:s1].sum()))
    return np.asarray(ev)


def trimmed_ok(rows, frac=0.01):
    """Book cost0 (log, bp/day) minus the k most favourable spell contributions, k = max(1, ceil(frac n))."""
    ev = events(rows)
    if ev.size == 0:
        return float("nan"), 0
    k = max(1, int(np.ceil(frac * ev.size)))
    top = np.sort(ev)[::-1][:k].sum()
    d = sum(r["days"] for r in rows)
    return 1e4 * (sum(r["cost0"] for r in rows) - top) / d, k


# --------------------------------------------------------------------------- worker caches (VM on the GPU, book on CPU)
def write_cache(ctx, cdir):
    """What the workers need, as .npy (mmap-read) + one pickle; float64 kept for the book (bit-identical)."""
    import pickle
    os.makedirs(cdir, exist_ok=True)
    for k in ("fwd_wx", "fi", "fr"):
        np.save(os.path.join(cdir, "sh_" + k + ".npy"), ctx["shard"][k])
    for k in ("beta", "state", "Larr", "mask"):
        np.save(os.path.join(cdir, k + ".npy"), ctx[k])
    np.save(os.path.join(cdir, "feats32.npy"), ctx["feats_gpu"].cpu().numpy())
    pickle.dump(dict(months=ctx["months"], bi=ctx["bi"], chunks=ctx["chunks"], cover_rows=ctx["cover_rows"],
                     offsets=ctx["offsets"], mode=ctx["mode"], K=K), open(os.path.join(cdir, "meta.pkl"), "wb"))
    return cdir


def score_view(cdir):
    import pickle
    meta = pickle.load(open(os.path.join(cdir, "meta.pkl"), "rb"))
    v = {k: np.load(os.path.join(cdir, k + ".npy"), mmap_mode="r") for k in ("beta", "state", "Larr", "mask")}
    v["shard"] = {k: np.load(os.path.join(cdir, "sh_" + k + ".npy"), mmap_mode="r") for k in ("fwd_wx", "fi", "fr")}
    v.update(months=meta["months"], bi=meta["bi"])
    return v


def vm_view(cdir, device):
    import pickle
    import torch
    meta = pickle.load(open(os.path.join(cdir, "meta.pkl"), "rb"))
    v = dict(mask=np.load(os.path.join(cdir, "mask.npy")), months=meta["months"], chunks=meta["chunks"],
             cover_rows=meta["cover_rows"], offsets=meta["offsets"])
    v["feats_gpu"] = torch.from_numpy(np.load(os.path.join(cdir, "feats32.npy"))).to(device)
    return v


def vm_select(view, vm, body, halves=("A",)):
    """Body -> per-month (hi, lo, n) selections of the requested halves, or raises Invalid / VM invalid."""
    return select_months(mask_signal(view, raw_signal(view, vm, body)), view, halves)


def vm_select_gpu(view, vm, body, halves=("A",)):
    """vm_select with the signal kept on the GPU (sel1_selcheck: identical selections, Invalid identical).
    float32 -> float64 is exact; np.roll(x, o)[t] = x[(t - o) mod T] as a gather; numpy's stable argsort treats
    -0.0 == 0.0 and torch's sort does not, so keys get + 0.0 (only -0.0 changes); keys are finite or +inf."""
    import torch
    if view.get("cse") is not None:
        return vm_select(view, vm, body, halves)
    dev = view["feats_gpu"].device
    if "mask_gpu" not in view:
        A_, T_ = view["mask"].shape
        view["mask_gpu"] = torch.from_numpy(np.ascontiguousarray(view["mask"])).to(dev)
        view["cover_gpu"] = torch.from_numpy(np.asarray(view["cover_rows"], np.int64)).to(dev)
        if view["offsets"] is not None:
            off = torch.from_numpy(np.asarray(view["offsets"], np.int64)).to(dev)
            view["roll_idx"] = (torch.arange(T_, device=dev)[None, :] - off[:, None]) % T_
    A, T = view["mask"].shape
    s = torch.full((A, T), float("nan"), dtype=torch.float64, device=dev)
    with torch.no_grad():
        for in0, o0, o1 in view["chunks"]:
            r = vm.execute(list(body), view["feats_gpu"][:, :, in0:o1]).detach()
            s[:, o0:o1] = r[:, o0 - in0:].to(torch.float64)
        if view["offsets"] is not None:
            s = torch.gather(s, 1, view["roll_idx"])
        s = torch.where(view["mask_gpu"] & torch.isfinite(s), s, torch.full_like(s, float("nan")))
        n = torch.isfinite(s[:, view["cover_gpu"]]).sum(dim=0).cpu().numpy()
        cov = float((n >= MIN_COINS).mean())
        if cov < COVER_MIN:
            raise Invalid("LOW_COVERAGE_%.2f" % cov)
        inf = torch.tensor(float("inf"), dtype=torch.float64, device=dev)
        out = []
        for m in view["months"]:
            if m["half"] not in halves:
                continue
            Sm = s[:, m["m0"] - 1:m["m1"] - 1].T.clone()
            Sm[0] = float("nan")
            fin = torch.isfinite(Sm)
            lo = torch.argsort(torch.where(fin, Sm, inf) + 0.0, dim=1, stable=True)[:, :K]
            hi = torch.argsort(torch.where(fin, -Sm, inf) + 0.0, dim=1, stable=True)[:, :K]
            out.append((hi.to(torch.int16).cpu().numpy(), lo.to(torch.int16).cpu().numpy(),
                        fin.sum(dim=1).to(torch.int16).cpu().numpy()))
    return out


def rows_lite(rows):
    return [dict((k, r[k]) for k in ("month", "half", "year", "net", "cost0", "turn", "n_pos", "days")) for r in rows]


_WV, _WS = {}, {}


def set_k_from_cache(cdir):
    """Workers are spawned with the module default K; the cache carries the parent's K (caches written before
    K was stored ran K = 5)."""
    import pickle
    global K
    K = int(pickle.load(open(os.path.join(cdir, "meta.pkl"), "rb")).get("K", 5))


def init_vm_worker(cdir, cse_gb=0.0, fastts=False):
    import torch
    set_k_from_cache(cdir)
    import cr1_common as Q
    import sel1_cse
    warnings.simplefilter("ignore", RuntimeWarning)
    torch.set_num_threads(1)
    dev = torch.device("cuda")
    vm = Q.make_vm()
    if fastts:
        import sel1_fastts
        sel1_fastts.install()
    view = vm_view(cdir, dev)
    view["cse"] = sel1_cse.CachedExec(vm, cse_gb) if cse_gb else None
    _WV.update(view=view, vm=vm)


def vm_task(body):
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid
    try:
        return body, vm_select(_WV["view"], _WV["vm"], body)
    except (Invalid, CandidateLocalInvalid, M.ZeroCoverage) as e:
        return body, "INVALID:" + str(e)


def init_score_worker(cdir):
    set_k_from_cache(cdir)
    warnings.simplefilter("ignore", RuntimeWarning)
    _WS.update(view=score_view(cdir))


def score_task(task):
    key, sels, h, st, dr = task
    return key, rows_lite(score_sel(sels, _WS["view"], h, st, dr))
