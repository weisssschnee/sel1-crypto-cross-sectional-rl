"""ENR1b enrichment table: enr1.py (receipt ENR1_TABLE.json, not modified) with the three changes of ENR1b_PREREG.json
(sha16 a312ab8c9f674dd4, on top of ENR1_PREREG.json sha16 1db50993ce274590). CPU only, V4 TRAIN only, no DEV, no OOS.
  1 main return = fwd_x (no winsor, delisting layer), built through m1_ruler.prepare_shard(winsor_pct=0) +
    v4_world.install_layer and asserted bit-equal to world()'s fwd_x; the winsorised fwd_wx is report only.
  2 gate 6: per half drop the k = max(1, ceil(1 %)) most favourable events (d-direction flat6 net), mean of the rest > 0.
  3 data check before any gate: the 20 events with the largest |unhedged no-winsor log return| over the hold, closes
    checked against the source parquets; a bad price (source mismatch or a one-bar > 0.2 log spike that reverts to
    within 0.02 the next bar) stops the run before the gate (rc 13).
Everything else is enr1.py verbatim; enr1.py's notes follow.

  world    v4_world.load + world(feats=False): fwd_wx (1/99 winsor + delisting layer), fwd_x, fr / fi, panel TOP100.
           v4_world, m1_ruler and m2h_verdict_dense are imported, never modified.
  signals  1-8 m2h_verdict_dense.handwritten (h1 formulas); 9-12 below.
  events   flat over eligible coin-hours (TOP100, not BTC, beta finite, state defined, fwd_wx[t+1] finite, t+1 < L,
           t >= 2021-01-01). For direction d the hourly book on rows t+1..t+h is
             x = d (f_i - b f_btc - fund_i + b fund_btc), 0 from the delisting hour L on,
           summed as log1p(clip(x, -0.9)) exactly as m1_ruler.settle_multi does (cost0); flat6 puts c = 6bp (1+|b|)
           into the entry row and adds the exit row log1p(-c).
  cells    12 signals x 4 conditions x 5 states x 4 holds. Per (signal, condition, hold): month filter, then greedy
           per-coin dedup with gap h, then split by the trigger hour's BTC state and the A/B half. t = mean of the
           long cost0 log return / week-clustered SE.
  null     48 condition series shifted per coin along its eligible hours, one offset per coin per world, 200 worlds
           over a spawn process pool; T_null = q95 of the per-world max |t_A| over cells with n_A >= 30.

    python -I enr1.py            ENR1_TEST=1: the real world is replaced by a null world (seed +555), 4 null worlds,
                                 outputs *_TEST; used to time the full run before it is launched.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd

ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem"
TEST = os.environ.get("ENR1B_TEST") == "1"
SFX = "_TEST" if TEST else ""
OUT_DIR = os.path.join(ROOT, "artifacts", "m2h_v1", "ENR1")
PREREG = os.path.join(OUT_DIR, "ENR1_PREREG.json")
PREREG_SHA16 = "1db50993ce274590"
PREREG_B = os.path.join(OUT_DIR, "ENR1b_PREREG.json")
PREREG_B_SHA16 = "a312ab8c9f674dd4"
TABLE = os.path.join(OUT_DIR, "ENR1b_TABLE%s.json" % SFX)
DATACHECK = os.path.join(OUT_DIR, "ENR1b_DATACHECK%s.json" % SFX)
SCRATCH = os.path.join(OUT_DIR, "scratch_b%s" % SFX)
RTD = os.path.join(ROOT, "runtime")
SOURCES = (("BF3", os.path.join(RTD, "train2021_2023_panel_v1", "BACKFILL2020NOV_2023DEC_HOURLY11_LONG_V3.parquet")),
           ("TR", os.path.join(RTD, "train2024_2025_panel_v1", "TRAIN2024JAN_2025DEC_HOURLY54_TARGET_LONG_V1.parquet")),
           ("SM2", os.path.join(RTD, "xs_seam2024_2026_panel_v1", "SEAM2024JAN_2026JUN_HOURLY11_LONG_V2.parquet")))
TRIM = 0.01
NGATE = 6

SIGS = ("T1_mom_168h", "CTREND_lite_17", "T2_neg_funding_7d", "T2_neg_retail_LS_24h", "T2_neg_dlogOI_7d",
        "T3_neg_premium_24h", "T3_neg_mark_index_24h", "T3_abs_premium_high_1h",
        "S9_ret_24h", "S10_qv24_over_720mean", "S11_taker_share_24h", "S12_dlogOI_24h")
OI_LS = ("T2_neg_retail_LS_24h", "T2_neg_dlogOI_7d", "S12_dlogOI_24h")
CONDS = ("CS_hi", "CS_lo", "TS_hi", "TS_lo")
STATES = ("all", "up_lowvol", "up_highvol", "down_lowvol", "down_highvol")
HOLDS = (4, 24, 72, 168)
YEARS = (2021, 2022, 2023, 2024, 2025)
Q_HI, Q_LO, CS_MIN, TS_WIN, TS_MIN = 0.95, 0.05, 20, 2160, 1080
BETA_WIN, BETA_MIN, STATE_WIN, THR = 720, 360, 720, 0.005196
N_NULL = 4 if TEST else 200
N_FWER = 2 if TEST else 20
SEED = 20261014
N_MIN, T_B, PLANT = 30, 2.0, 0.005
WORKERS = 12
THREADS = 8
BIG = 1 << 20
WORKER_KEYS = ("EH", "CO", "START", "NCO", "ST", "HF", "WK", "NW", "PT", "PO") + tuple(
    "%s%d" % (p, h) for p in ("V", "GP") for h in HOLDS)


def log(m):
    print(m, flush=True)


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def rnd(x, k=3):
    x = float(x)
    return round(x, k) if np.isfinite(x) else None


def lagk(x, k):
    y = np.full_like(x, np.nan)
    y[:, k:] = x[:, :-k]
    return y


def roll(x, k, how, mp=None):
    r = pd.DataFrame(x.T).rolling(k, min_periods=mp or k)
    return getattr(r, how)().to_numpy().T


# --------------------------------------------------------------------------- cell machinery (workers use only this)
_D = {}


def _init(scratch):
    _D.clear()
    for k in WORKER_KEYS:
        _D[k] = np.load(os.path.join(scratch, k + ".npy"))


def offsets(D, rng):
    """One circular offset per coin along its eligible hours (PREREG null.shift)."""
    off = np.zeros(len(D["NCO"]), np.int64)
    for a, m in enumerate(D["NCO"]):
        m = int(m)
        if m >= 1440:
            off[a] = rng.integers(720, m - 720 + 1)
        elif m >= 2:
            off[a] = rng.integers(1, m)
    return off


def shift(P, D, off):
    if off is None:
        return P
    a = D["CO"][P]
    st = D["START"][a]
    return np.sort(st + (P - st + off[a]) % D["NCO"][a])


def dedup(P, D, h):
    """Month filter, then greedy per-coin dedup: after an event at t the next one starts at >= t + h."""
    P = P[D["V%d" % h][P]]
    if P.size == 0:
        return P
    a = D["CO"][P].astype(np.int64)
    key = a * BIG + D["EH"][P]
    nxt = np.searchsorted(key, key + h)
    ok = nxt < key.size
    ok[ok] = (key[nxt[ok]] // BIG) == a[ok]
    nxt = np.where(ok, nxt, -1)
    cur = np.flatnonzero(np.r_[True, a[1:] != a[:-1]])
    keep = []
    while cur.size:
        keep.append(cur)
        cur = nxt[cur]
        cur = cur[cur >= 0]
    return P[np.sort(np.concatenate(keep))]


def tstats(x, ev, D):
    """n, mean, week-clustered t per (state row 0..4, half): row 0 = all."""
    W = int(D["NW"][0])
    key = (D["ST"][ev].astype(np.int64) * 2 + D["HF"][ev]) * W + D["WK"][ev]
    S = np.bincount(key, weights=x, minlength=8 * W).reshape(4, 2, W)
    N = np.bincount(key, minlength=8 * W).reshape(4, 2, W).astype(np.float64)
    S = np.concatenate([S.sum(0)[None], S])
    N = np.concatenate([N.sum(0)[None], N])
    n = N.sum(2)
    with np.errstate(all="ignore"):
        m = S.sum(2) / n
        s = S - N * m[..., None]
        g = (N > 0).sum(2)
        se = np.sqrt(g / (g - 1) * (s * s).sum(2)) / n
        t = np.where((g >= 2) & (se > 0), m / se, np.nan)
    return n, m, t


def run_cells(D, off, keep=False, plant=None):
    shp = (len(SIGS), len(CONDS), len(HOLDS), len(STATES), 2)
    n, m, t = np.zeros(shp), np.full(shp, np.nan), np.full(shp, np.nan)
    evs = {}
    for k in range(len(SIGS)):
        for c in range(len(CONDS)):
            j = k * len(CONDS) + c
            P = shift(D["PT"][D["PO"][j]:D["PO"][j + 1]], D, off)
            for hi, h in enumerate(HOLDS):
                ev = dedup(P, D, h)
                x = D["GP%d" % h][ev]
                if plant is not None and plant[:3] == (k, c, hi):
                    x = x + PLANT * (D["ST"][ev] == plant[3] - 1)
                n[k, c, hi], m[k, c, hi], t[k, c, hi] = tstats(x, ev, D)
                if keep:
                    evs[(k, c, hi)] = ev
    return n, m, t, evs


def null_max(D, r):
    n, _, t, _ = run_cells(D, offsets(D, np.random.default_rng([SEED, r])))
    a = np.abs(t[..., 0])
    ok = (n[..., 0] >= N_MIN) & np.isfinite(a)
    return r, float(a[ok].max()) if ok.any() else float("nan"), float(np.nanmax(a)), t[..., 0].astype(np.float32)


def _null_job(r):
    return null_max(_D, r)


# --------------------------------------------------------------------------- full statistics and gate (main only)
def grp_mean(x, ev, D, cat, ncat):
    key = D["ST"][ev].astype(np.int64) * ncat + cat
    S = np.bincount(key, weights=x, minlength=4 * ncat).reshape(4, ncat)
    N = np.bincount(key, minlength=4 * ncat).reshape(4, ncat).astype(np.float64)
    S, N = np.vstack([S.sum(0), S]), np.vstack([N.sum(0), N])
    with np.errstate(all="ignore"):
        return N, S / N


def full_stats(D, evs, nmt, tn, plant=None):
    n, m, t = nmt
    cells = {}
    for (k, c, hi), ev in evs.items():
        h = HOLDS[hi]
        pl = PLANT * (D["ST"][ev] == plant[3] - 1) if plant is not None and plant[:3] == (k, c, hi) else 0.0
        gp, ep, em = D["GP%d" % h][ev] + pl, D["EP%d" % h][ev] + pl, D["EM%d" % h][ev] - pl
        hf, yr, tr = (D[x][ev].astype(np.int64) for x in ("HF", "YR", "TIER"))
        z = np.zeros(ev.size, np.int64)
        _, EPh = grp_mean(ep, ev, D, hf, 2)
        _, EMh = grp_mean(em, ev, D, hf, 2)
        _, GPy = grp_mean(gp, ev, D, yr, len(YEARS))
        Nt, EPt = grp_mean(ep, ev, D, tr, 2)
        _, EMt = grp_mean(em, ev, D, tr, 2)
        rep = {nm: grp_mean(D["%s%d" % (nm, h)][ev], ev, D, z, 1)[1][:, 0] for nm in ("RAWP", "RAWM", "WINP", "WINM")}
        yrs = list(range(1, len(YEARS))) if SIGS[k] in OI_LS else list(range(len(YEARS)))
        need = 3 if SIGS[k] in OI_LS else 4
        for s in range(len(STATES)):
            mA = m[k, c, hi, s, 0]
            d = 1 if mA > 0 else (-1 if mA < 0 else 0)
            net = EPh[s] if d > 0 else EMh[s]
            nett = EPt[s] if d > 0 else EMt[s]
            ysame = int(sum(1 for y in yrs if d * GPy[s, y] > 0))
            nA, nB, tA, tB = n[k, c, hi, s, 0], n[k, c, hi, s, 1], t[k, c, hi, s, 0], t[k, c, hi, s, 1]
            # gate 6 (ENR1b): per half drop the k most favourable events, the rest must still net > 0
            insel = np.ones(ev.size, bool) if s == 0 else D["ST"][ev] == s - 1
            vd = ep if d > 0 else em
            trim = []
            for half in (0, 1):
                v = np.sort(vd[insel & (hf == half)])
                kk = max(1, int(np.ceil(TRIM * v.size)))
                trim.append((float(v[:v.size - kk].mean()) if v.size > kk else float("nan"), kk))
            gates = [bool(abs(tA) > tn), bool(d * tB > T_B), bool(net[0] > 0 and net[1] > 0), bool(ysame >= need),
                     bool(nA >= N_MIN and nB >= N_MIN), bool(d != 0 and trim[0][0] > 0 and trim[1][0] > 0)]
            cells["%s|%s|%s|H%d" % (SIGS[k], CONDS[c], STATES[s], h)] = dict(
                idx=[k, c, hi, s], d=d,
                A=dict(n=int(nA), t=rnd(tA, 2), g_bp=rnd(1e4 * mA, 2), net_bp=rnd(1e4 * net[0], 2)),
                B=dict(n=int(nB), t=rnd(tB, 2), g_bp=rnd(1e4 * m[k, c, hi, s, 1], 2), net_bp=rnd(1e4 * net[1], 2)),
                years={str(YEARS[y]): rnd(1e4 * d * GPy[s, y], 1) for y in range(len(YEARS))},
                years_same=ysame, years_need="%d/%d" % (need, len(yrs)),
                tiers={"top30": dict(n=int(Nt[s, 0]), net_bp=rnd(1e4 * nett[0], 2)),
                       "31_100": dict(n=int(Nt[s, 1]), net_bp=rnd(1e4 * nett[1], 2))},
                raw_unhedged_cost0_bp=rnd(1e4 * (rep["RAWP"][s] if d > 0 else rep["RAWM"][s]), 2),
                hedged_winsor_cost0_bp_report_only=rnd(1e4 * (rep["WINP"][s] if d > 0 else rep["WINM"][s]), 2),
                trimmed_net_bp=dict(A=rnd(1e4 * trim[0][0], 2), B=rnd(1e4 * trim[1][0], 2),
                                    dropped=[trim[0][1], trim[1][1]]),
                gates=gates, passed=all(gates))
    return cells


def nearest(cells, tn, k=10):
    rest = [(NGATE - sum(v["gates"]), -abs(v["A"]["t"] or 0.0) / tn, key) for key, v in cells.items() if not v["passed"]]
    return [dict(cell=key, n_failed=nf, tA_over_Tnull=round(-r, 3), gates=cells[key]["gates"])
            for nf, r, key in sorted(rest)[:k]]


# --------------------------------------------------------------------------- precompute (main only)
def precompute(V, X):
    t0 = time.time()
    w = X.load(X.GRID)
    X.world(w, feats=False)
    g, ts, univ, sh = w["g"], w["ts"], w["univ"], w["shard"]
    A, T = univ.shape
    import m1_ruler as R
    sh0 = dict(fwd=sh["fwd"], eligible=sh["eligible"], fi=sh["fi"], fr=sh["fr"])
    R.prepare_shard(sh0, 0)                        # the ruler's own no-winsor path
    X.install_layer(sh0, w["dead"])
    if not np.array_equal(sh0["fwd_wx"], sh["fwd_x"], equal_nan=True):
        raise SystemExit("prepare_shard(winsor_pct=0) + layer != world fwd_x")
    fw_main, fw_win = sh0["fwd_wx"], sh["fwd_wx"]
    bi = w["syms"].index("BTCUSDT")
    log("loaded %.0fs: %d coins x %d h, %d delistings" % (time.time() - t0, A, T, len(w["dead"])))
    S = V.handwritten(g, ts)
    S = {k: S[k] for k in SIGS[:8]}
    close, qv = g["trade_close"], g["trade_quote_volume"]
    logc = np.log(close)
    oi = g["open_interest_value_last"]
    loi = np.where(oi > 0, np.log(np.where(oi > 0, oi, 1.0)), np.nan)
    with np.errstate(all="ignore"):
        q24 = roll(qv, 24, "sum")
        S["S9_ret_24h"] = logc - lagk(logc, 24)
        S["S10_qv24_over_720mean"] = q24 / roll(qv, 720, "mean", 360)
        S["S11_taker_share_24h"] = roll(g["taker_buy_quote_volume"], 24, "sum") / np.where(q24 > 0, q24, np.nan)
        S["S12_dlogOI_24h"] = loi - lagk(loi, 24)
    for k in SIGS:
        S[k] = np.where(np.isfinite(S[k]), S[k], np.nan)
    log("signals %.0fs" % (time.time() - t0))

    # beta: trailing 720 h OLS of coin hourly log returns on BTC's, >= 360 joint observations
    lr = np.diff(logc, axis=1, prepend=np.nan)
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
        beta = np.where((nn >= BETA_MIN) & (var > 0), (sxy - sx * sy / nn) / var, np.nan)
    del lr, msk, X0, Y0, nn, sx, sy, sxy, sxx, var

    # BTC state (diag_d1d2.btc_state / labels): codes 0..3 = up_lowvol, up_highvol, down_lowvol, down_highvol
    btc = np.log(close[bi])
    ret30 = pd.Series(btc).diff(STATE_WIN).to_numpy()
    vol30 = pd.Series(np.diff(btc, prepend=np.nan)).rolling(STATE_WIN, min_periods=STATE_WIN // 2).std().to_numpy()
    st_ok = np.isfinite(ret30) & np.isfinite(vol30)
    up, hv = ret30 > 0, vol30 > THR
    state = np.where(up, np.where(hv, 1, 0), np.where(hv, 3, 2)).astype(np.int8)

    i0 = int(ts.get_loc(pd.Timestamp("2021-01-01")))
    fw = fw_main
    nxt_ok = np.zeros((A, T), bool)
    nxt_ok[:, :-1] = np.isfinite(fw[:, 1:])
    E0 = univ & nxt_ok
    E0[:, :i0] = False
    E0[bi] = False
    for a, L in w["dead"].items():
        E0[a, max(L - 1, 0):] = False
    base = int(E0.sum())
    E0 &= np.isfinite(beta)
    no_beta = base - int(E0.sum())
    no_state = int((E0 & ~st_ok[None, :]).sum())
    E0 &= st_ok[None, :]
    fa, ft = np.nonzero(E0)
    D = dict(EH=ft.astype(np.int64), CO=fa.astype(np.int64))
    cnt = np.bincount(fa, minlength=A)
    D["NCO"] = cnt.astype(np.int64)
    D["START"] = np.r_[0, np.cumsum(cnt)[:-1]].astype(np.int64)
    mon = np.asarray(ts.year * 12 + ts.month)
    D["ST"] = state[ft]
    D["HF"] = (np.asarray(ts.month)[ft] % 2 == 0).astype(np.int8)
    wk = np.asarray((ts - pd.Timestamp("2020-10-26")) // pd.Timedelta(hours=168), dtype=np.int64)
    D["WK"] = wk[ft]
    D["NW"] = np.array([int(wk.max()) + 1])
    D["YR"] = (np.asarray(ts.year)[ft] - YEARS[0]).astype(np.int8)
    D["BETA"] = beta[E0]
    for h in HOLDS:
        v = np.zeros(T, bool)
        v[:T - h - 1] = mon[:T - h - 1] == mon[h + 1:]
        D["V%d" % h] = v[ft]
    v30 = pd.DataFrame(qv.T).rolling(720, min_periods=360).sum()
    rk = v30.where(pd.DataFrame(univ.T)).rank(axis=1, ascending=False, method="first").to_numpy().T
    D["TIER"] = (rk[E0] > 30).astype(np.int8)
    del v30, rk
    log("beta/state/eligible %.0fs: eligible %d (no beta %d, no state %d)" % (time.time() - t0, E0.sum(), no_beta, no_state))

    # conditions -> trigger positions in the flat eligible order
    pos = np.full((A, T), -1, np.int64)
    pos[fa, ft] = np.arange(fa.size)
    PT, PO, trig = [], [0], {}
    for k in SIGS:
        s = S[k]
        xm = np.where(univ, s, np.nan)
        r = pd.DataFrame(xm.T).rank(axis=1).to_numpy().T
        nc = np.isfinite(xm).sum(axis=0).astype(np.float64)
        with np.errstate(all="ignore"):
            q = (r - 1.0) / (nc - 1.0)
        q[:, nc < CS_MIN] = np.nan
        ro = pd.DataFrame(s.T).rolling(TS_WIN, min_periods=TS_MIN)
        rr, cc = ro.rank().to_numpy().T, ro.count().to_numpy().T
        with np.errstate(all="ignore"):
            qt = np.where(np.isfinite(s) & (cc >= TS_MIN), (rr - 1.0) / (cc - 1.0), np.nan)
        for cname, cond in zip(CONDS, (q >= Q_HI, q <= Q_LO, qt >= Q_HI, qt <= Q_LO)):
            p = pos[cond & E0]
            PT.append(p)
            PO.append(PO[-1] + p.size)
            trig["%s|%s" % (k, cname)] = int(p.size)
        log("conditions %s %.0fs" % (k, time.time() - t0))
    D["PT"] = np.concatenate(PT)
    D["PO"] = np.array(PO, np.int64)
    del S, pos

    # event log returns on the eligible coin-hours, in coin chunks over threads (numpy releases the GIL)
    F, Fw = np.nan_to_num(fw_main), np.nan_to_num(fw_win)
    FR = np.where(sh["fi"], sh["fr"], 0.0)
    alive = np.ones((A, T), bool)
    for a, L in w["dead"].items():
        alive[a, L:] = False
    b0 = np.nan_to_num(beta)
    Fb, Fwb, FRb = F[bi], Fw[bi], FR[bi]
    names = ("GP", "GM", "WINP", "WINM", "RAWP", "RAWM")

    def chunk(rows):
        acc = {nm: np.zeros((len(rows), T)) for nm in names}
        e0 = E0[rows]
        Fc, Fwc, FRc, alc, bc = F[rows], Fw[rows], FR[rows], alive[rows], b0[rows]
        out = {}
        for k in range(1, max(HOLDS) + 1):
            sl, to = slice(k, None), slice(0, T - k)
            al = alc[:, sl]
            fr_ = FRc[:, sl]
            bb = bc[:, to]
            hed = (Fc[:, sl] - fr_ - bb * (Fb[sl] - FRb[sl])) * al
            hwin = (Fwc[:, sl] - fr_ - bb * (Fwb[sl] - FRb[sl])) * al
            raw = (Fc[:, sl] - fr_) * al
            for nm, v in (("GP", hed), ("GM", -hed), ("WINP", hwin), ("WINM", -hwin), ("RAWP", raw), ("RAWM", -raw)):
                acc[nm][:, to] += np.log1p(np.clip(v, -0.9, None))
            if k == 1:
                x1 = np.zeros((len(rows), T))
                x1[:, to] = hed
                out["X1"] = x1[e0]
            if k in HOLDS:
                for nm in names:
                    out["%s%d" % (nm, k)] = acc[nm][e0]
        return out
    parts = np.array_split(np.arange(A), 32)
    with ThreadPoolExecutor(THREADS) as ex:
        res = list(ex.map(chunk, parts))
    for key in res[0]:
        D[key] = np.concatenate([r[key] for r in res])
    c_rate = R.cost_specs(R.NAV_USD)["flat6"][1]
    if abs(c_rate - 6e-4) > 1e-12:
        raise SystemExit("flat6 rate is not 6 bp: %r" % c_rate)
    c = c_rate * (1.0 + np.abs(D["BETA"]))
    lc = np.log1p(-c)
    for d, gk, ek in ((1, "GP", "EP"), (-1, "GM", "EM")):
        x1 = d * D["X1"]
        adj = np.log1p(np.clip(x1 - c, -0.9, None)) - np.log1p(np.clip(x1, -0.9, None)) + lc
        for h in HOLDS:
            D["%s%d" % (ek, h)] = D["%s%d" % (gk, h)] + adj
    log("event returns %.0fs" % (time.time() - t0))
    Larr = np.full(A, T + 10, np.int64)
    for a, L in w["dead"].items():
        Larr[a] = L
    ctx = dict(ts=ts, syms=w["syms"], sh=sh, fw_main=fw_main, close=g["trade_close"], bi=bi, Larr=Larr, mon=mon,
               T=T, triggers=trig,
               eligible=int(fa.size), excluded_no_beta=no_beta, excluded_no_state=no_state,
               beta_q=[rnd(v, 3) for v in np.quantile(D["BETA"], [0.01, 0.25, 0.5, 0.75, 0.99])],
               seconds=round(time.time() - t0, 1))
    return D, ctx


# --------------------------------------------------------------------------- selftests (main only)
def ab_check(D, evs, ctx):
    T, mon = ctx["T"], ctx["mon"]
    cov = [np.zeros(T, bool), np.zeros(T, bool)]
    same_month = parity = True
    for (k, c, hi), ev in evs.items():
        h = HOLDS[hi]
        t = D["EH"][ev]
        same_month &= bool(np.all(mon[t] == mon[t + h + 1]))
        parity &= bool(np.all(D["HF"][ev] == (mon[t + h + 1] % 2 == 0)))
        for half in (0, 1):
            tt = t[D["HF"][ev] == half]
            dif = np.bincount(tt, minlength=T + h + 3)[:T + 1] - np.bincount(tt + h + 2, minlength=T + h + 3)[:T + 1]
            cov[half] |= np.cumsum(dif)[:T] > 0
    ov = int((cov[0] & cov[1]).sum())
    return dict(same_month=same_month, half_parity=parity, overlap_hours=ov, hours_A=int(cov[0].sum()),
                hours_B=int(cov[1].sum()), ok=bool(same_month and parity and ov == 0))


def ruler_check(D, evs, ctx, R, rng, n_want=20):
    sh, bi, Larr, ts = ctx["sh"], ctx["bi"], ctx["Larr"], ctx["ts"]
    fw = ctx["fw_main"]
    specs = {c: R.cost_specs(R.NAV_USD)[c] for c in ("flat6", "cost0")}
    bad_exit = 0
    for (k, c, hi), ev in evs.items():
        h = HOLDS[hi]
        a, t = D["CO"][ev], D["EH"][ev]
        bad_exit += int((~np.isfinite(fw[a, t + h + 1]) & (Larr[a] > t + h)).sum())
    keys = [k for k, ev in evs.items() if ev.size]
    out, tries = [], 0
    while len(out) < n_want and tries < 5000:
        tries += 1
        key = keys[int(rng.integers(len(keys)))]
        ev = evs[key]
        p = int(ev[int(rng.integers(ev.size))])
        h = HOLDS[key[2]]
        a, t = int(D["CO"][p]), int(D["EH"][p])
        L = int(Larr[a])
        dies = L <= t + h
        if not dies and not np.isfinite(fw[a, t + h + 1]):
            continue
        d = 1 if len(out) % 2 == 0 else -1
        b = float(D["BETA"][p])
        rows = np.arange(t + 1, t + h + 2)
        held = np.zeros((h + 1, 2))
        held[:h, 0], held[:h, 1] = d, -d * b
        held[rows >= L] = 0.0
        cols = [a, bi]
        f = fw[cols][:, rows].T
        fi, fr = sh["fi"][cols][:, rows].T, sh["fr"][cols][:, rows].T
        assert held.shape[1] == f.shape[1] == fi.shape[1] == fr.shape[1], "held / fwd column count differs"
        res, _ = R.settle_multi(held, f, fi, fr, specs)
        tn_ = float(D[("EP" if d > 0 else "EM") + str(h)][p])
        tg_ = float(D[("GP" if d > 0 else "GM") + str(h)][p])
        out.append(dict(cell="%s|%s|H%d" % (SIGS[key[0]], CONDS[key[1]], h), coin=ctx["syms"][a], t=str(ts[t]), d=d,
                        beta=round(b, 3), dies_in_hold=bool(dies), ruler_net_bp=round(1e4 * res["flat6"]["log_nav"], 3),
                        table_net_bp=round(1e4 * tn_, 3),
                        diff_net_bp=float(1e4 * abs(res["flat6"]["log_nav"] - tn_)),
                        diff_cost0_bp=float(1e4 * abs(res["cost0"]["log_nav"] - tg_))))
    mx = max([max(o["diff_net_bp"], o["diff_cost0_bp"]) for o in out], default=float("nan"))
    return dict(events=out, max_diff_bp=mx, nontradable_exit_events=bad_exit,
                ok=bool(len(out) == n_want and mx <= 1.0))


def plant_check(D, tn):
    off = offsets(D, np.random.default_rng(SEED + 999))
    n, m, t, evs = run_cells(D, off, keep=True)
    base = full_stats(D, evs, (n, m, t), tn)
    base_pass = sorted(k for k, v in base.items() if v["passed"])
    best, pick = -1.0, None
    for k in range(len(SIGS)):
        for c in range(len(CONDS)):
            for s in range(1, len(STATES)):
                v = min(n[k, c, 0, s, 0], n[k, c, 0, s, 1])
                if v >= N_MIN and v > best:
                    best, pick = v, (k, c, 0, s)
    if pick is None:
        return dict(ok=False, reason="no h=4 state cell with n >= 30 in both halves")
    n2, m2, t2, _ = run_cells(D, off, plant=pick)
    pl = full_stats(D, evs, (n2, m2, t2), tn, plant=pick)
    name = "%s|%s|%s|H%d" % (SIGS[pick[0]], CONDS[pick[1]], STATES[pick[3]], HOLDS[0])
    parent = "%s|%s|all|H%d" % (SIGS[pick[0]], CONDS[pick[1]], HOLDS[0])
    passed = sorted(k for k, v in pl.items() if v["passed"])
    others = [k for k in passed if k not in (name, parent)]
    return dict(world_seed=SEED + 999, unplanted_pass=base_pass, planted=name, planted_cell=pl[name],
                parent=parent, parent_passed=parent in passed, passed=passed, others_passed=others,
                ok=bool(name in passed and not others))


def data_check(D, evs, ctx, n_top=20):
    """ENR1b change 3: the n_top real events with the largest |unhedged no-winsor log return| over the hold; entry /
    exit closes and the closes around the hold's largest one-hour move, checked against the source parquets."""
    import pyarrow.parquet as pq
    ts, syms, close, fw, T = ctx["ts"], ctx["syms"], ctx["close"], ctx["fw_main"], ctx["T"]
    keys, vals = [], []
    for (k, c, hi), ev in evs.items():
        keys.append(ev * len(HOLDS) + hi)
        vals.append(np.abs(D["RAWP%d" % HOLDS[hi]][ev]))
    keys, vals = np.concatenate(keys), np.concatenate(vals)
    keys, first = np.unique(keys, return_index=True)
    top = keys[np.argsort(-vals[first], kind="stable")[:n_top]]
    rows, need = [], {}
    for q in top:
        p, hi = int(q // len(HOLDS)), int(q % len(HOLDS))
        h = HOLDS[hi]
        a, t = int(D["CO"][p]), int(D["EH"][p])
        hrs = np.arange(t + 1, t + h + 1)
        j = int(hrs[np.argmax(np.abs(np.log1p(np.nan_to_num(fw[a, hrs]))))])
        chk = sorted({t + 1, t + h + 1, j, j + 1, j + 2} & set(range(T)))
        with np.errstate(all="ignore"):
            r1 = float(np.log(close[a, j + 1] / close[a, j]))
            r2 = float(np.log(close[a, j + 2] / close[a, j + 1])) if j + 2 < T else float("nan")
        rows.append(dict(coin=syms[a], a=a, t=str(ts[t]), h=h, raw_log=round(float(D["RAWP%d" % h][p]), 4),
                         max_move_hour=str(ts[j]), r1=round(r1, 4), r2=round(r2, 4) if np.isfinite(r2) else None,
                         spike_revert=bool(abs(r1) > 0.2 and np.isfinite(r2) and abs(r1 + r2) < 0.02), hours=chk))
        need.setdefault(syms[a], set()).update(chk)
    src = {}
    for name, path in SOURCES:
        tb = pq.read_table(path, columns=["symbol", "timestamp", "trade_close"],
                           filters=[("symbol", "in", sorted(need))]).to_pandas()
        tx = pd.to_datetime(tb.timestamp)
        if tx.dt.tz is not None:
            tx = tx.dt.tz_convert("UTC").dt.tz_localize(None)
        for s_, t_, c_ in zip(tb.symbol.astype(str), tx, tb.trade_close.astype(float)):
            src.setdefault((s_, t_), []).append((name, c_))
    bad = 0
    for r in rows:
        pts = []
        for hh in r.pop("hours"):
            gc = float(close[r["a"], hh])
            sv = src.get((r["coin"], ts[hh]), [])
            ok = (not sv and not np.isfinite(gc)) or (bool(sv) and all(
                np.isfinite(v) and np.isfinite(gc) and abs(v - gc) <= 1e-9 * abs(gc) for _, v in sv))
            pts.append(dict(hour=str(ts[hh]), grid=gc, source=[[n_, v] for n_, v in sv], match=bool(ok)))
        r["closes"] = pts
        r["source_match"] = all(x["match"] for x in pts)
        r.pop("a")
        bad += int((not r["source_match"]) or r["spike_revert"])
    return dict(rule="ENR1b_PREREG change_3", events=rows, n_bad=bad, bad=bool(bad))


# --------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    for p_, s_ in ((PREREG, PREREG_SHA16), (PREREG_B, PREREG_B_SHA16)):
        if sha16(p_) != s_:
            raise SystemExit("PREREG sha16 %s != %s (%s)" % (sha16(p_), s_, p_))
    for p_ in (TABLE, DATACHECK):
        if os.path.exists(p_) and not TEST:
            raise SystemExit("%s exists; not overwritten" % p_)
    import m2h_verdict_dense as V      # sets the stage-1 / ROOT paths
    import v4_world as X
    import m1_ruler as R
    code_sha = sha16(os.path.abspath(__file__))
    D, ctx = precompute(V, X)
    os.makedirs(SCRATCH, exist_ok=True)
    for k in WORKER_KEYS:
        np.save(os.path.join(SCRATCH, k + ".npy"), D[k])
    t_pre = time.time() - t0
    log("precompute %.0fs, code %s, test %s" % (t_pre, code_sha, TEST))

    import multiprocessing as mp
    t1 = time.time()
    res = []
    with mp.get_context("spawn").Pool(min(WORKERS, N_NULL), initializer=_init, initargs=(SCRATCH,)) as pool:
        for i, r in enumerate(pool.imap_unordered(_null_job, range(N_NULL))):
            res.append(r)
            if (i + 1) % 20 == 0 or TEST:
                log("null %d/%d %.0fs" % (i + 1, N_NULL, time.time() - t1))
    res.sort(key=lambda r: r[0])
    t_null = time.time() - t1
    maxes = np.array([r[1] for r in res])
    tn = float(np.quantile(maxes, 0.95))
    TN = np.stack([r[3] for r in res]).astype(np.float64)
    log("T_null %.3f (null %.0fs)" % (tn, t_null))

    t2 = time.time()
    off_real = offsets(D, np.random.default_rng(SEED + 555)) if TEST else None
    n, m, t, evs = run_cells(D, off_real, keep=True)
    dc = data_check(D, evs, ctx)
    dc["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(dc, open(DATACHECK, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    log("data check: %d / %d bad (%.0fs)" % (dc["n_bad"], len(dc["events"]), time.time() - t2))
    if dc["bad"] and not TEST:
        log("STOP bad prices before the gate; see %s" % DATACHECK)
        sys.exit(13)
    cells = full_stats(D, evs, (n, m, t), tn)
    t_cells = time.time() - t2
    # report only, added after the TEST run (gate unchanged): each cell's own null t_A distribution, and the
    # realised false-positive rate of the whole five-part gate on extra pure-null worlds
    for v in cells.values():
        k, c, hi, s = v["idx"]
        nt = TN[:, k, c, hi, s]
        nt = nt[np.isfinite(nt)]
        tA = v["A"]["t"]
        v["null_tA"] = dict(mean=rnd(nt.mean(), 2) if nt.size else None, sd=rnd(nt.std(), 2) if nt.size else None,
                            p_abs_ge=rnd(np.mean(np.abs(nt) >= abs(tA)), 3) if nt.size and tA is not None else None)
    t3 = time.time()
    fw_pass = []
    for i in range(N_FWER):
        nw_, mw_, tw_, ew_ = run_cells(D, offsets(D, np.random.default_rng([SEED, 1000 + i])), keep=True)
        fw_pass.append(sorted(k for k, v in full_stats(D, ew_, (nw_, mw_, tw_), tn).items() if v["passed"]))
    fwer = dict(worlds=N_FWER, seeds="[20261014, 1000 + i]", worlds_with_pass=sum(1 for p in fw_pass if p),
                passes=fw_pass, seconds=round(time.time() - t3, 1))
    log("null FWER %d/%d worlds with >= 1 pass (%.0fs)" % (fwer["worlds_with_pass"], N_FWER, fwer["seconds"]))
    st = dict(ab=ab_check(D, evs, ctx), ruler=ruler_check(D, evs, ctx, R, np.random.default_rng(SEED + 20)),
              plant=plant_check(D, tn))
    st["ok"] = bool(st["ab"]["ok"] and st["ruler"]["ok"] and st["plant"]["ok"])
    log("selftest ab %s ruler %s (max %.2e bp) plant %s -> %s" % (st["ab"]["ok"], st["ruler"]["ok"],
        st["ruler"]["max_diff_bp"], st["plant"]["ok"], st["ok"]))
    passed = sorted(k for k, v in cells.items() if v["passed"])
    timing = dict(precompute_s=round(t_pre, 1), null_s=round(t_null, 1), null_workers=min(WORKERS, N_NULL),
                  cells_s=round(t_cells, 1), total_s=round(time.time() - t0, 1))
    if TEST:
        timing["projected_full_s"] = round(t_pre + t_null / max(1, -(-N_NULL // min(WORKERS, N_NULL)))
                                           * -(-200 // WORKERS) + 3 * t_cells + 120, 1)
    rep = dict(schema="ENR1b_TABLE_V1", test=TEST, prereg_sha16=PREREG_SHA16, prereg_b_sha16=PREREG_B_SHA16,
               main_return="fwd_x (no winsor, delisting layer); fwd_wx report only", data_check=dc["n_bad"],
               code_sha16=code_sha,
               grid_sha16=sha16(X.GRID), contract="XS_BASIN_TIME_CONTRACT_V4_20261013", dev_read=False, oos_read=False,
               real_world="null world seed +555 (TEST)" if TEST else "real",
               T_null=round(tn, 4), null_maxes=[round(float(v), 4) for v in maxes],
               null_max_unrestricted=[round(r[2], 4) for r in res],
               n_cells=len(cells), passed=passed, n_passed=len(passed), nearest10=nearest(cells, tn),
               null_fwer_report_only=fwer, selftest=st, timing=timing,
               world=dict(eligible_coin_hours=ctx["eligible"], excluded_no_beta=ctx["excluded_no_beta"],
                          excluded_no_state=ctx["excluded_no_state"], beta_quantiles_1_25_50_75_99=ctx["beta_q"],
                          triggers_before_month_filter_and_dedup=ctx["triggers"]),
               cells=cells, written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(rep, open(TABLE, "w", encoding="utf-8"), indent=0, ensure_ascii=False, default=float)
    log("wrote %s passed %d T_null %.3f %.0fs" % (TABLE, len(passed), tn, time.time() - t0))
    if TEST:
        log("projected full run %.0fs" % timing["projected_full_s"])
    sys.exit(11 if not st["ok"] else (0 if passed else 10))


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
