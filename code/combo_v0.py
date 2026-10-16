"""COMBO_V0 (ADR-0010, 10-13): composite of the 8 h1 handwritten H168 cells on the K10 concentrated
book. CPU only, no GPU, no OOS. One run, three stages; it stops at the first failed gate.

 stage 0  sign convention (TRAIN): an oracle (signal = the realised 168 h forward return) must give
          IC 1 and positive gross on both books; per cell, the 168 h forward rank IC
          (b5_diag_regime_sector's path; its V2 'all' IC is re-derived as a check) is reported
          against the ruler's TRAIN gross (a |t| > 2 cell whose signs disagree is listed, not
          gated). Path checks: h1 port (all 8 cells, full book, net and gross == h1 TRAIN json)
          and a one-cell composite == the raw cell on the K10 and full books. rc 11 on failure.
 stage 1  internal holdout: signs = sign of the 2024 IC (TOP100 & CONC_POOL, forward window inside
          2024), scored on the 2025 TRAIN tiles, m1_ruler flat6, H168, conc K10. Gate: 2025 net > 0
          and > 1-week momentum on the same book and tiles. rc 10 on failure (DEV not opened).
 stage 2  signs refit on all TRAIN, FROZEN_COMBO_V0.json written, then DEV opened once: conc K10
          (primary), full pool, K5 / K20, OKX settlement of the K10 path, per-tile spread.
          Gate: K10 net > 0 and full net > 0.

Composite, per hour: each cell's average rank among the pool's coins with a finite value
(TOP100 & CONC_POOL for the conc books, TOP100 for the full book), standardised to z across those
coins (>= MIN_COINS); composite = mean over available cells of sign_k * z_k, >= MIN_CELLS of 8.
Sensitivity (reported, not gated): w_k = 0.5 * IR_k / sum|IR| + 0.5 * sign_k / 8, IR = IC mean /
IC sd of the fitting period; composite = sum w z / sum |w| over available cells.
Ruler, band and costs unchanged; no no-trade band. IC t uses overlapping 168 h horizons (optimistic).

    python -I combo_v0.py              (COMBO_TEST=1: stage 2 on the last 12 TRAIN tiles, *_TEST files)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"          # CPU only (ADR-0010)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                           # -I drops cwd and PYTHONPATH
import numpy as np
import pandas as pd
import b5_fixed as F
TR, M, D, W, R = F._imports()
import m2h_verdict_dense as V
import b5_verdict as BV
import b5_diag_regime_sector as DG

TEST = os.environ.get("COMBO_TEST") == "1"
SFX = "_TEST" if TEST else ""
OUT_DIR = os.path.join(F.A_ROOT, "COMBO_V0")
TRAIN_OUT = os.path.join(OUT_DIR, "COMBO_V0_TRAIN%s.json" % SFX)
FROZEN = os.path.join(OUT_DIR, "FROZEN_COMBO_V0%s.json" % SFX)
DEV_OUT = os.path.join(OUT_DIR, "COMBO_V0_DEV%s.json" % SFX)
DIAG_V2 = os.path.join(F.A_ROOT, "B5_GATE1", "DIAG_REGIME_SECTOR_V2.json")
CELLS = ("T1_mom_168h", "CTREND_lite_17") + V.T2_CELLS + V.T3_CELLS
MOM = "T1_mom_168h"
E = M.EPISODE
H_IC, STEP, MIN_CELLS, MIN_COINS = 168, 24, 5, 20
K10 = BV.PRIMARY
KK = BV.bk(K10)
CONC = tuple(b for b in BV.BOOKS if b[0] == "conc")
FULL = ("full",)


def log(m):
    print(m, flush=True)


def dump(obj, path):
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)


def sgn(x):
    return 1 if x >= 0 else -1


def ic_stats(sig, fut, mask, hours):
    """Rank IC per signal hour (signal at h-1 -> 168 h forward from h), rows with >= 20 coins."""
    a = np.where(mask[:, hours - 1], sig[:, hours - 1], np.nan).T
    b = np.where(mask[:, hours - 1], fut[:, hours], np.nan).T
    v = D.cs_spearman_rows(a, b)
    v = v[(np.isfinite(a) & np.isfinite(b)).sum(axis=1) >= D.IC_MIN_COINS]
    m = float(v.mean()) if len(v) else float("nan")
    sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    ok = sd > 0
    return dict(ic=round(m, 5), sd=round(sd, 5), ir=round(m / sd, 4) if ok else None,
                t=round(m / (sd / np.sqrt(len(v))), 2) if ok else None, n=int(len(v)))


def zrank(x, mask):
    """(A, T) -> per hour, average rank over mask & finite, standardised across those coins."""
    v = np.where(mask & np.isfinite(x), x, np.nan)
    r = pd.DataFrame(v.T).rank(axis=1).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(r, axis=1, keepdims=True)
        sd = np.nanstd(r, axis=1, keepdims=True)
        z = np.where(sd > 0, (r - mu) / np.where(sd > 0, sd, 1.0), np.nan)
    z[np.isfinite(r).sum(axis=1) < MIN_COINS] = np.nan
    return z.T


def composite(Z, w, min_cells=MIN_CELLS):
    num = np.zeros(Z[0].shape)
    den = np.zeros(Z[0].shape)
    cnt = np.zeros(Z[0].shape, dtype=int)
    for z, wk in zip(Z, w):
        f = np.isfinite(z)
        num += np.where(f, wk * z, 0.0)
        den += np.where(f, abs(wk), 0.0)
        cnt += f
    return np.where(cnt >= min_cells, num / np.where(den > 0, den, 1.0), np.nan)


def tiles_of(sig, starts, univ):
    """h1's handwritten path: TOP100 mask on the absolute grid, row t = signal hour s0 + t - 1."""
    sig = np.where(univ, sig, np.nan)
    out = []
    for s0 in starts:
        L = np.full((sig.shape[0], E), np.nan)
        L[:, M.LAG:] = sig[:, s0:s0 + E - M.LAG]
        out.append(L.T)
    return out


def books_of(sig_c, sig_f, starts, univ, shard, pool, conc=(K10,), keep=False):
    """{book key: {field: per tile}}: sig_c through the conc books, sig_f through the full book."""
    rec = {}
    for sig, books, kh in ((sig_c, conc, keep), (sig_f, (FULL,), False)):
        r = BV.score_sigs(tiles_of(sig, starts, univ), starts, shard, books, pool, keep_held=kh)
        if r is None:
            raise RuntimeError("non-finite tile net")
        rec.update(r)
    return rec


def summ(rec, sel=None):
    return {k: V.summary(v, sel) for k, v in rec.items() if not k.startswith("_")}


def zsets(S, univ, pool):
    """Per-cell z on the conc mask (TOP100 & CONC_POOL) and on TOP100, computed once per world."""
    mc = univ & pool[:, None]
    return [zrank(S[k], mc) for k in CELLS], [zrank(S[k], univ) for k in CELLS]


def score_combo(Zs, w, starts, univ, shard, pool, conc=(K10,), keep=False):
    wl = [w[k] for k in CELLS]
    return books_of(composite(Zs[0], wl), composite(Zs[1], wl), starts, univ, shard, pool, conc, keep)


def weights(ic):
    """Main = sign of IC, equal weight; sensitivity = IR weights shrunk 50 % toward equal weight."""
    sign = {k: sgn(ic[k]["ic"]) for k in CELLS}
    eq = {k: sign[k] / len(CELLS) for k in CELLS}
    ir = {k: float(ic[k]["ir"] or 0.0) for k in CELLS}
    tot = sum(abs(v) for v in ir.values()) or 1.0
    sens = {k: round(0.5 * ir[k] / tot + 0.5 * eq[k], 6) for k in CELLS}
    return sign, eq, sens


def ic_table(S, fut, mask, hours):
    return {k: ic_stats(S[k], fut, mask, hours) for k in CELLS}


def spread(v, starts_ts):
    """Per-tile distribution: is the mean carried by one or two tiles?"""
    v = np.asarray(v, dtype=np.float64)
    o = np.sort(v)[::-1]
    pos = float(v[v > 0].sum())
    return dict(per_tile=[dict(start=str(s), net=round(float(x), 3)) for s, x in zip(starts_ts, v)],
                mean=round(float(v.mean()), 3), median=round(float(np.median(v)), 3),
                t=round(F.tstat(v), 2), n_pos=int((v > 0).sum()), n=int(len(v)),
                mean_drop_best1=round(float(o[1:].mean()), 3), mean_drop_best2=round(float(o[2:].mean()), 3),
                top2_share_of_positive=round(float(o[:2].clip(min=0).sum() / pos), 3) if pos > 0 else None)


def train_world():
    man, arr = F.load_cache()
    syms, ts, grids = M.load_train_grids(TR.PANEL)
    assert list(syms) == list(man["symbols"])
    S = V.handwritten(grids, ts)
    S = {k: S[k] for k in CELLS}
    del grids
    univ = np.array(arr["univ"])
    tw = dict(syms=list(syms), ts=pd.DatetimeIndex(ts), univ=univ, shard=F.make_shard("real", arr),
              pool=W.pool_cols(list(syms), BV.POOL_JSON), S=S, starts=[int(x) for x in man["starts"]])
    assert len(tw["starts"]) == 50
    return tw


# --------------------------------------------------------------------------- stage 0
def stage0(tw, fut, hours_all):
    univ, sh, pool, S, starts = tw["univ"], tw["shard"], tw["pool"], tw["S"], tw["starts"]
    rep = {}
    # oracle: signal at h-1 = the 168 h forward return from h; IC ~ 1 and gross > 0 on both books
    orc = np.full(fut.shape, np.nan)
    orc[:, :-1] = fut[:, 1:]
    ro = books_of(orc, orc, starts[:4], univ, sh, pool)
    o_ic = ic_stats(orc, fut, univ, hours_all)["ic"]
    rep["oracle"] = dict(ic=o_ic, conc_k10_gross=[round(x, 2) for x in ro[KK]["gross"]],
                         full_gross=[round(x, 2) for x in ro["full"]["gross"]])
    oracle_ok = o_ic > 0.99 and min(ro[KK]["gross"]) > 0 and min(ro["full"]["gross"]) > 0
    # port (all 8 cells, full book, 50 tiles) against the h1 TRAIN json
    h1 = json.load(open(V.H1, encoding="utf-8"))["results"]
    raw = {k: books_of(S[k], S[k], starts, univ, sh, pool) for k in CELLS}
    port = {}
    for k in CELLS:
        ref = h1[k + "|TOP100|lag1"]["168"]
        port[k] = max(abs(float(np.mean(raw[k]["full"][f])) - float(np.mean(ref[f]))) for f in ("net", "gross"))
    port_ok = max(port.values()) <= 1e-6
    # the diag's IC path re-derived (TOP100, its hours)
    diag = json.load(open(DIAG_V2, encoding="utf-8"))["regime"]["cells"]
    ic_top = ic_table(S, fut, univ, hours_all)
    ic_conc = ic_table(S, fut, univ & pool[:, None], hours_all)
    rederive = max(abs(float(np.nanmean(DG.ic_series(S[k], fut, univ, hours_all, D))) - diag[k]["all"]["ic"])
                   for k in CELLS)                 # the diag's own unfiltered mean
    rows, mism = {}, []
    for k in CELLS:
        fg, cg = float(np.mean(raw[k]["full"]["gross"])), float(np.mean(raw[k][KK]["gross"]))
        rows[k] = dict(ic_top100=ic_top[k], full_gross_train=round(fg, 3), ic_conc=ic_conc[k],
                       conc_k10_gross_train=round(cg, 3),
                       full_agree=sgn(ic_top[k]["ic"]) == sgn(fg), conc_agree=sgn(ic_conc[k]["ic"]) == sgn(cg))
        if abs(ic_top[k]["t"] or 0) > 2 and not rows[k]["full_agree"]:
            mism.append(k)
    one = {}
    for k, s in ((MOM, 1), ("T3_abs_premium_high_1h", -1)):     # one-cell composite == the raw cell
        Zc, Zf = zrank(S[k], univ & pool[:, None]), zrank(S[k], univ)
        a = books_of(s * Zc, s * Zf, starts[:8], univ, sh, pool)
        b = books_of(s * S[k], s * S[k], starts[:8], univ, sh, pool)
        one[k] = max(abs(x - y) for bk_ in (KK, "full") for x, y in zip(a[bk_]["net"], b[bk_]["net"]))
    one_ok = max(one.values()) <= 1e-9
    rep.update(port_maxdiff=port, port_ok=port_ok, oracle_ok=oracle_ok, diag_ic_rederive_maxdiff=round(rederive, 6),
               rederive_ok=rederive <= 5e-4, one_cell_maxdiff=one, one_cell_ok=one_ok, cells=rows,
               full_sign_mismatch_t_gt_2=mism,
               rule="sign convention: higher signal = long in both books (C.rank_weights / conc_signal) and "
                    "positive rank IC = higher signal earns more; the oracle checks both ends")
    # the convention itself is settled by the oracle (both books, the IC path); a cell whose IC and
    # ruler gross disagree is reported, not gated (IC t is overlapping, rank IC != H168 banded PnL)
    rep["ok"] = bool(oracle_ok and port_ok and rederive <= 5e-4 and one_ok)
    log("S0 oracle %s port %s (max %.2e) rederive %.1e one-cell %s full-sign mismatch %s -> %s"
        % (oracle_ok, port_ok, max(port.values()), rederive, one_ok, mism, rep["ok"]))
    return rep, raw


# --------------------------------------------------------------------------- stage 1
def stage1(tw, fut, raw, Zs):
    univ, sh, pool, S, starts, ts = tw["univ"], tw["shard"], tw["pool"], tw["S"], tw["starts"], tw["ts"]
    T = univ.shape[1]
    i25 = int(ts.searchsorted(pd.Timestamp("2025-01-01")))
    lo = max(M.LOOKBACK, DG.WIN) + 1
    h24 = np.arange(lo, i25 - H_IC + 1, STEP)               # 168 h forward window ends inside 2024
    h25 = np.arange(i25 + 1, T - H_IC, STEP)
    mc = univ & pool[:, None]
    ic24, ic25 = ic_table(S, fut, mc, h24), ic_table(S, fut, mc, h25)
    ic24_top = ic_table(S, fut, univ, h24)
    sign, eq, sens = weights(ic24)
    sel = [i for i, s in enumerate(starts) if ts[s] >= pd.Timestamp("2025-01-01")]
    t25 = [starts[i] for i in sel]
    main_r = score_combo(Zs, eq, t25, univ, sh, pool)
    sens_r = score_combo(Zs, sens, t25, univ, sh, pool)
    main_s, sens_s = summ(main_r), summ(sens_r)
    mom = {b: V.summary(raw[MOM][b], sel) for b in (KK, "full")}
    bar = mom[KK]["net"]
    ok = bool(main_s[KK]["net"] > 0 and main_s[KK]["net"] > bar)
    rep = dict(fit="2024 rank IC, TOP100 & CONC_POOL, %d signal hours" % len(h24), n_tiles_2025=len(t25),
               sign_2024=sign, weights_main=eq, weights_sens=sens,
               ic_2024=ic24, ic_2025=ic25,
               sign_same_2024_2025={k: sgn(ic24[k]["ic"]) == sgn(ic25[k]["ic"]) for k in CELLS},
               sign_2024_top100_differs=[k for k in CELLS if sgn(ic24_top[k]["ic"]) != sign[k]],
               combo_2025=main_s, combo_2025_sens=sens_s, mom1w_2025=mom,
               combo_2025_k10_tiles=spread(main_r[KK]["net"], [ts[s] for s in t25]),
               cells_2025_raw={k: {b: V.summary(raw[k][b], sel)["net"] for b in (KK, "full")} for k in CELLS},
               gate=dict(rule="2025 conc K10 net > 0 and > 1-week momentum (same book, same tiles)",
                         combo=main_s[KK]["net"], mom1w=bar, passed=ok))
    log("S1 2025 conc K10 net %.2f (mom %.2f) full %.2f turn %.3f beta %.3f | sens K10 %.2f -> %s"
        % (main_s[KK]["net"], bar, main_s["full"]["net"], main_s[KK]["turn"], main_s[KK]["beta"],
           sens_s[KK]["net"], ok))
    return rep


# --------------------------------------------------------------------------- stage 2 freeze
def freeze(tw, fut, hours_all, Zs, code_sha):
    univ, sh, pool, S, starts, ts = tw["univ"], tw["shard"], tw["pool"], tw["S"], tw["starts"], tw["ts"]
    ic_all = ic_table(S, fut, univ & pool[:, None], hours_all)
    sign, eq, sens = weights(ic_all)
    rec = score_combo(Zs, eq, starts, univ, sh, pool, conc=CONC)
    yrs = np.array([ts[s].year for s in starts])
    fz = dict(schema="FROZEN_COMBO_V0", test=TEST, cells=list(CELLS), sign=sign, weights_main=eq,
              weights_sens=sens, ic_train_conc=ic_all,
              composite="per hour: average rank -> z over TOP100 & CONC_POOL (conc books) or TOP100 (full), "
                        ">= %d coins; mean of sign*z over available cells, >= %d of %d" % (MIN_COINS, MIN_CELLS, len(CELLS)),
              hold_h=F.HOLD, ruler="m1_ruler flat6 / cost0, band %s, lag %d" % (R.BAND, M.LAG),
              code_sha16=code_sha,
              conc_pool_sha16=hashlib.sha256(open(BV.POOL_JSON, "rb").read()).hexdigest()[:16],
              train50_in_sample=summ(rec), train2024=summ(rec, list(np.where(yrs == 2024)[0])),
              train2025=summ(rec, list(np.where(yrs == 2025)[0])), dev_read=False, oos_read=False)
    blob = json.dumps(dict(cells=fz["cells"], sign=sign, w=eq, ws=sens), sort_keys=True).encode()
    fz["sha256"] = hashlib.sha256(blob).hexdigest()
    fz["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dump(fz, FROZEN)
    log("FROZEN %s sign %s TRAIN50 K10 net %.2f" % (fz["sha256"][:12], sign, fz["train50_in_sample"][KK]["net"]))
    return fz


# --------------------------------------------------------------------------- stage 2 DEV
def dev_world():
    """b5_verdict's DEV loading: DEV2026 panel, 12 tiles from 2026-01-01, CONC pool, OKX shard."""
    z = np.load(V.DEV_NPZ, allow_pickle=False)
    hours = pd.DatetimeIndex(z["hours"].astype("datetime64[ns]"))
    if hours.max() >= M.OOS_FIRST:
        raise SystemExit("DEV panel reaches OOS")
    dsyms = [str(x) for x in z["symbols"]]
    dgrids = {c: z[c] for c in M.PANEL_COLS}
    jan1 = int((pd.Timestamp("2026-01-01") - hours[0]) / pd.Timedelta(hours=1))
    w = M.build_world(dgrids)
    starts = M.tile_starts(w["feats"].shape[2], first=jan1)[:12]
    if len(starts) != 12 or hours[starts[-1] + E - 1] >= M.OOS_FIRST:
        raise SystemExit("DEV tiling wrong: %d tiles" % len(starts))
    S = V.handwritten(dgrids, hours)
    osh, okx_info = BV.okx_shard(dsyms, hours)
    dw = dict(syms=dsyms, ts=hours, univ=w["univ"], shard=w["shard"], pool=W.pool_cols(dsyms, BV.POOL_JSON),
              S={k: S[k] for k in CELLS}, starts=[int(x) for x in starts])
    return dw, osh, okx_info


def stage2(fz, dw, osh, okx_info):
    univ, sh, pool, S, starts, ts = dw["univ"], dw["shard"], dw["pool"], dw["S"], dw["starts"], dw["ts"]
    Zs = zsets(S, univ, pool)
    rec = score_combo(Zs, fz["weights_main"], starts, univ, sh, pool, conc=CONC, keep=True)
    sens = score_combo(Zs, fz["weights_sens"], starts, univ, sh, pool)
    raw = {k: books_of(S[k], S[k], starts, univ, sh, pool) for k in CELLS}
    main_s = summ(rec)
    st = [ts[s] for s in starts]
    ok = bool(main_s[KK]["net"] > 0 and main_s["full"]["net"] > 0)
    live = [float((univ[:, s0 + t] & pool).sum()) for s0 in starts for t in range(0, E, 24)]
    out = dict(schema="COMBO_V0_DEV", test=TEST, frozen_sha256=fz["sha256"], n_tiles=len(starts),
               span=[str(st[0]), str(ts[starts[-1] + E - 1])], conc_pool_live_mean=round(float(np.mean(live)), 1),
               conc_pool_live_min=int(min(live)), combo=main_s, combo_sens=summ(sens), mom1w=summ(raw[MOM]),
               cells_raw={k: {b: V.summary(raw[k][b])["net"] for b in (KK, "full")} for k in CELLS},
               tiles={b: spread(rec[b]["net"], st) for b in (KK, "full")},
               okx=okx_info, okx_conc_k10=BV.okx_settle(rec["_held"], starts, osh) if osh is not None else None,
               binance_resettle_maxdiff_bpd=BV.binance_resettle_check(rec["_held"], starts, sh, rec),
               gate=dict(rule="DEV conc K10 net > 0 and full-pool net > 0", conc_k10=main_s[KK]["net"],
                         full=main_s["full"]["net"], passed=ok),
               oos_read=False)
    o = out["okx_conc_k10"]
    log("S2 DEV K10 net %.2f (t %.2f, %d/%d tiles > 0, drop best 2 %.2f) full %.2f | K5 %.2f K20 %.2f | "
        "OKX %s | sens K10 %.2f | mom %.2f -> %s"
        % (main_s[KK]["net"], out["tiles"][KK]["t"], out["tiles"][KK]["n_pos"], len(starts),
           out["tiles"][KK]["mean_drop_best2"], main_s["full"]["net"], main_s["conc_k5"]["net"],
           main_s["conc_k20"]["net"], o["win"]["net"] if o else None, out["combo_sens"][KK]["net"],
           out["mom1w"][KK]["net"], ok))
    return out


# --------------------------------------------------------------------------- main
def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    for p in (DEV_OUT, FROZEN):                    # DEV is read once: no rerun after the freeze
        if os.path.exists(p) and not TEST:
            raise SystemExit("%s exists; not overwritten" % p)
    code_sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
    tw = train_world()
    T = tw["univ"].shape[1]
    hours_all = np.arange(max(M.LOOKBACK, DG.WIN) + 1, T - H_IC, STEP)
    fut = DG.fwd_cum(tw["shard"]["fwd_w"], H_IC)
    log("TRAIN loaded %.0fs, code %s, test %s" % (time.time() - t0, code_sha, TEST))
    s0, raw = stage0(tw, fut, hours_all)
    rep = dict(schema="COMBO_V0_TRAIN", test=TEST, code_sha16=code_sha, stage0=s0, dev_read=False, oos_read=False)
    if not s0["ok"]:
        dump(rep, TRAIN_OUT)
        log("STOP stage 0 failed")
        sys.exit(11)
    Zs = zsets(tw["S"], tw["univ"], tw["pool"])
    rep["stage1"] = stage1(tw, fut, raw, Zs)
    dump(rep, TRAIN_OUT)
    if not rep["stage1"]["gate"]["passed"] and not TEST:
        log("STOP stage 1 gate failed; DEV not opened")
        sys.exit(10)
    fz = freeze(tw, fut, hours_all, Zs, code_sha)
    del Zs, raw, fut
    assert os.path.exists(FROZEN)
    if TEST:                                       # stand-in: last 12 TRAIN tiles, no OKX
        dw, osh, okx_info = dict(tw, starts=tw["starts"][-12:]), None, None
    else:
        dw, osh, okx_info = dev_world()
    out = stage2(fz, dw, osh, okx_info)
    out["seconds"] = round(time.time() - t0, 1)
    out["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dump(out, DEV_OUT)
    log("wrote %s %.0fs" % (DEV_OUT, time.time() - t0))
    sys.exit(0 if out["gate"]["passed"] else 12)


if __name__ == "__main__":
    main()

