"""COMBO_V1 (order V4_COMBO_V1_CR1 s2.2; contract V4). CPU only, no OOS. combo_v0.py is not modified;
COMBO_V0 has a receipt and is not rerun.

Conventions (written before any number is read; PREREG below):
  data      V4 grid (v4_world): point-in-time TOP100 incl. later delistings; delisting layer on; ruler
            m1_ruler flat6, H168, band 0.5, lag 1. Both books on TOP100: K10 conc = top/bottom 10 of
            TOP100, full = rank book over TOP100. (The OKX 205 pool is only a DEV column.)
  cells     the 8 h1 handwritten cells (m2h_verdict_dense.handwritten, unchanged).
  signs     year Y in 2022..2025: sign of the mean rank IC (signal h-1 -> 168 h forward from h, every
            24 h, TOP100, rows >= 20 coins) over signal hours 2021-01-01 .. Y-01-01 - 169 h (the forward
            window ends before Y). Coverage = valid IC rows / sample hours in that window; < 50 % ->
            weight 0 for year Y.
  composite per hour, per active cell: average rank over TOP100 & finite -> z (>= 20 coins);
            composite = sum w z / sum |w| over finite cells, needs >= min(5, n_active) cells.
            Sensitivity (reported, not gated): w = 0.5 IR/sum|IR| + 0.5 sign/n_active.
  tiles     back-to-back 336 h tiles from 2022-01-01 (year = tile-start year).
  gate      mean K10 net over all 2022-25 tiles > 0 AND > T1_mom_168h with a fixed + sign on the same
            book and tiles. Full pool, K5/K20 and per-year are reported only.
  extra     2024 IC per cell on V4 TOP100 vs the same on the 2026-06 survivors only (the coins of the old
            TRAIN panel), vs V0's 2024 IC: how much of 2024's negative momentum is survivorship.
  pass  ->  signs refit on 2021-01..2025-12 (forward window ends <= 2025-12-31), FROZEN_COMBO_V1.json,
            DEV read once: K10, full, K5/K20, OKX column (K10 path settled on OKX), per tile.
            DEV gate: K10 net > 0 and full net > 0.  fail -> rc 10, DEV not opened.

    python -I combo_v1.py     (COMBO_TEST=1: 2 tiles per year, "DEV" = last 4 TRAIN tiles, *_TEST files)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m2h_verdict_dense as V           # handwritten cells; sets the stage-1 / ROOT paths
import numpy as np
import pandas as pd
import m2h_common as M
import m2h_dense_common as D
import v4_world as X

PREREG = dict(p_pass_train_gate=0.25, written="2026-10-14 02:40 (77o)",
              why="V0 died on 2024->2025 sign flips of 4/8 cells; expanding windows from 2021 keep "
                  "momentum's long-run + sign if 2021-23 agree, but 2022 is fit on the 2021 bull alone "
                  "and OI/LS are off until 2023; the 8-cell mean was ~0 on V3 TRAIN.")
TEST = os.environ.get("COMBO_TEST") == "1"
SFX = "_TEST" if TEST else ""
OUT_DIR = os.path.join(X.ROOT, "artifacts", "m2h_v1", "COMBO_V1")
TRAIN_OUT = os.path.join(OUT_DIR, "COMBO_V1_TRAIN%s.json" % SFX)
FROZEN = os.path.join(OUT_DIR, "FROZEN_COMBO_V1%s.json" % SFX)
DEV_OUT = os.path.join(OUT_DIR, "COMBO_V1_DEV%s.json" % SFX)
V0_JSON = os.path.join(X.ROOT, "artifacts", "m2h_v1", "COMBO_V0", "COMBO_V0_TRAIN.json")
OLD_TRAIN = os.path.join(X.ROOT, r"runtime\train2024_2025_panel_v1\TRAIN2024JAN_2025DEC_HOURLY54_TARGET_LONG_V1.parquet")
OKX_NPZ = os.path.join(HERE, "OKX_DEV_1H_V1.npz")
CELLS = ("T1_mom_168h", "CTREND_lite_17") + V.T2_CELLS + V.T3_CELLS
MOM = "T1_mom_168h"
H_IC, STEP, MIN_CELLS, COV_MIN = 168, 24, 5, 0.5
YEARS = (2022, 2023, 2024, 2025)


def log(m):
    print(m, flush=True)


def dump(obj, path):
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)


def fwd_cum(fw, h=H_IC):
    lr = np.log1p(np.nan_to_num(fw, nan=0.0))
    cs = np.concatenate([np.zeros((lr.shape[0], 1)), np.cumsum(lr, axis=1)], axis=1)
    out = np.full(lr.shape, np.nan)
    T = lr.shape[1]
    out[:, :T - h + 1] = cs[:, h:] - cs[:, :T - h + 1]
    return out


def ic_rows(sig, fut, mask, hours):
    """Per sample hour: rank IC (signal at h-1, forward from h); NaN where < 20 coins."""
    a = np.where(mask[:, hours - 1], sig[:, hours - 1], np.nan).T
    b = np.where(mask[:, hours - 1], fut[:, hours], np.nan).T
    v = D.cs_spearman_rows(a, b)
    ok = (np.isfinite(a) & np.isfinite(b)).sum(axis=1) >= D.IC_MIN_COINS
    return np.where(ok, v, np.nan)


def stat(v):
    n0 = len(v)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return dict(ic=None, ir=None, t=None, n=int(len(v)), cov=round(len(v) / max(n0, 1), 4))
    m, sd = float(v.mean()), float(v.std(ddof=1))
    return dict(ic=round(m, 5), ir=round(m / sd, 4) if sd > 0 else None,
                t=round(m / (sd / np.sqrt(len(v))), 2) if sd > 0 else None, n=int(len(v)),
                cov=round(len(v) / max(n0, 1), 4))


def sgn(x):
    return 1 if x >= 0 else -1


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def zsets(S, univ):
    """Per-cell average-rank z over TOP100 & finite (one pool for every book, contract s3)."""
    return {k: X.zrank_rows(np.where(univ & np.isfinite(S[k]), S[k], np.nan).T).T for k in CELLS}


def composite(Z, w):
    act = [k for k in CELLS if w.get(k, 0)]
    need = min(MIN_CELLS, len(act))
    shp = Z[CELLS[0]].shape
    num, den, cnt = np.zeros(shp), np.zeros(shp), np.zeros(shp, dtype=int)
    for k in act:
        f = np.isfinite(Z[k])
        num += np.where(f, w[k] * Z[k], 0.0)
        den += np.where(f, abs(w[k]), 0.0)
        cnt += f
    if not act:
        return np.full(shp, np.nan)
    return np.where(cnt >= need, num / np.where(den > 0, den, 1.0), np.nan)


def score(sig, sh, univ, starts, which=X.BOOKS, keep=False):
    """Every tile through v4_world (rank_weights -> held_path -> delisting layer -> settle_multi)."""
    sigm = np.where(univ, sig, np.nan)
    rec, helds = {}, []
    for s0 in starts:
        r = X.books(X.tile_rows(sigm, s0), sh, s0, which, keep=keep)
        for b, v in r.items():
            d = rec.setdefault(b, {f: [] for f in ("net", "gross", "turn", "beta")})
            for f in d:
                d[f].append(float(v[f]))
            if "_held" in v:
                helds.append(v["_held"])
    for b in rec:
        if not np.all(np.isfinite(rec[b]["net"])):
            raise RuntimeError("non-finite tile net in %s" % b)
    return rec, helds


def summ(rec, sel=None):
    out = {}
    for b, v in rec.items():
        idx = list(range(len(v["net"]))) if sel is None else list(sel)
        out[b] = {f: (round(float(np.mean([v[f][i] for i in idx])), 4) if idx else None) for f in v}
        out[b]["n"] = len(idx)
    return out


def spread(v, st):
    v = np.asarray(v, dtype=np.float64)
    o = np.sort(v)[::-1]
    pos = float(v[v > 0].sum())
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return dict(per_tile=[dict(start=str(s), net=round(float(x), 3)) for s, x in zip(st, v)],
                mean=round(float(v.mean()), 3), median=round(float(np.median(v)), 3),
                t=round(float(v.mean() / (sd / np.sqrt(len(v)))), 2) if sd > 0 else None,
                n_pos=int((v > 0).sum()), n=int(len(v)),
                mean_drop_best2=round(float(o[2:].mean()), 3) if len(v) > 2 else None,
                top2_share_of_positive=round(float(o[:2].clip(min=0).sum() / pos), 3) if pos > 0 else None)


def fit_hours(ts, lo, hi):
    """Sample signal hours h (signal at h-1 >= lo, forward window h..h+167 ends before hi), every 24 h."""
    i0 = int(ts.searchsorted(lo)) + 1
    i1 = int(ts.searchsorted(hi)) - H_IC - 1
    return np.arange(i0, i1 + 1, STEP)


def ic_table(S, fut, mask, hours):
    return {k: stat(ic_rows(S[k], fut, mask, hours)) for k in CELLS}


def weights(ics):
    """Main: sign of the fitting-window IC, equal weight over active cells (coverage >= 50 %).
    Sensitivity: 0.5 IR / sum|IR| + 0.5 sign / n_active."""
    act = {k: ics[k]["ic"] is not None and ics[k]["cov"] >= COV_MIN for k in CELLS}
    sign = {k: sgn(ics[k]["ic"]) if act[k] else 0 for k in CELLS}
    n = max(sum(act.values()), 1)
    eq = {k: sign[k] / n for k in CELLS}
    ir = {k: float(ics[k]["ir"] or 0.0) if act[k] else 0.0 for k in CELLS}
    tot = sum(abs(v) for v in ir.values()) or 1.0
    sens = {k: round(0.5 * ir[k] / tot + 0.5 * eq[k], 6) for k in CELLS}
    return sign, eq, sens


def stage0(w, fut, starts):
    """Delisting layer selftest (incl. the negative test) and the sign-convention oracle."""
    t = time.time()
    st = X.selftest(w)
    sh, univ, ts = w["shard"], w["univ"], w["ts"]
    orc = np.full(fut.shape, np.nan)
    orc[:, :-1] = fut[:, 1:]
    hrs = fit_hours(ts, pd.Timestamp("2022-01-01"), pd.Timestamp("2022-07-01"))
    o_ic = stat(ic_rows(orc, fut, univ, hrs))["ic"]
    ro, _ = score(orc, sh, univ, starts[:4], which=(X.K10, ("full",)))
    oracle_ok = bool(o_ic > 0.99 and min(ro["conc_k10"]["gross"]) > 0 and min(ro["full"]["gross"]) > 0)
    rep = dict(layer_selftest=st, oracle=dict(ic=o_ic, k10_gross=ro["conc_k10"]["gross"],
                                              full_gross=ro["full"]["gross"], ok=oracle_ok),
               ok=bool(st["ok"] and oracle_ok), seconds=round(time.time() - t, 1))
    log("S0 layer eq_sw %.1e eq_clean %.1e negative %s (%d cases) oracle ic %.3f %s -> %s"
        % (st["eq_score_window"], st["eq_layer_clean"], st["negative_ok"], len(st["negative"]), o_ic,
           oracle_ok, rep["ok"]))
    return rep

def fill_years(Z, wy, starts, yrs, shape):
    """One (A, T) signal: the columns a tile reads (s0 - LAG .. s0 + E - LAG - 1) get its start year's
    composite. Tiles are back to back, so every column belongs to one tile."""
    sig = np.full(shape, np.nan)
    for s0, y in zip(starts, yrs):
        a, b = s0 - M.LAG, s0 + X.E - M.LAG
        sig[:, a:b] = composite({k: Z[k][:, a:b] for k in CELLS}, wy[y])
    return sig


def stage1(w, fut, S, Z):
    ts, univ, sh = w["ts"], w["univ"], w["shard"]
    starts = list(X.seq_tiles(ts, pd.Timestamp("2022-01-01")))
    if TEST:
        starts = [s for y in YEARS for s in [x for x in starts if ts[x].year == y][:2]]
    yrs = [int(ts[s].year) for s in starts]
    fits, w_main, w_sens = {}, {}, {}
    for y in YEARS:
        hrs = fit_hours(ts, X.TRAIN_FIRST, pd.Timestamp("%d-01-01" % y))
        ics = ic_table(S, fut, univ, hrs)
        sign, eq, sens = weights(ics)
        w_main[y], w_sens[y] = eq, sens
        fits[y] = dict(window="2021-01-01 .. %d-01-01 (forward window ends before)" % y, sample_hours=len(hrs),
                       ic=ics, sign=sign, active=[k for k in CELLS if sign[k]], w_main=eq, w_sens=sens)
        log("fit %d: %d hours, signs %s" % (y, len(hrs), {k[:10]: sign[k] for k in CELLS}))
    t = time.time()
    rec, _ = score(fill_years(Z, w_main, starts, yrs, univ.shape), sh, univ, starts)
    rec_s, _ = score(fill_years(Z, w_sens, starts, yrs, univ.shape), sh, univ, starts, which=(X.K10, ("full",)))
    rec_m, _ = score(S[MOM], sh, univ, starts)
    raw = {k: score(S[k], sh, univ, starts, which=(X.K10, ("full",)))[0] for k in CELLS}
    log("scored %d tiles x (main, sens, mom, 8 raw) %.0fs" % (len(starts), time.time() - t))
    by_year = {y: [i for i, v in enumerate(yrs) if v == y] for y in YEARS}
    main_s, mom_s = summ(rec), summ(rec_m)
    ok = bool(main_s["conc_k10"]["net"] > 0 and main_s["conc_k10"]["net"] > mom_s["conc_k10"]["net"])
    st = [ts[s] for s in starts]
    flips = {k: [fits[y]["sign"][k] for y in YEARS] for k in CELLS}
    rep = dict(n_tiles=len(starts), span=[str(st[0]), str(ts[starts[-1] + X.E - 1])], fits=fits,
               sign_path=flips, combo=main_s, combo_sens=summ(rec_s), mom1w_fixed_plus=mom_s,
               by_year={y: dict(combo=summ(rec, by_year[y]), mom1w=summ(rec_m, by_year[y]),
                                combo_sens=summ(rec_s, by_year[y])) for y in YEARS},
               cells_raw={k: {b: summ(raw[k])[b]["net"] for b in ("conc_k10", "full")} for k in CELLS},
               cells_raw_by_year={k: {y: summ(raw[k], by_year[y])["conc_k10"]["net"] for y in YEARS} for k in CELLS},
               tiles_k10=spread(rec["conc_k10"]["net"], st), tiles_full=spread(rec["full"]["net"], st),
               gate=dict(rule="2022-25 mean conc K10 net > 0 and > fixed + sign T1_mom_168h (same book, tiles)",
                         combo=main_s["conc_k10"]["net"], mom1w=mom_s["conc_k10"]["net"], passed=ok))
    log("S1 2022-25 K10 net %.2f (mom %.2f) full %.2f turn %.3f beta %.3f | sens K10 %.2f | per year %s -> %s"
        % (main_s["conc_k10"]["net"], mom_s["conc_k10"]["net"], main_s["full"]["net"], main_s["conc_k10"]["turn"],
           main_s["conc_k10"]["beta"], rep["combo_sens"]["conc_k10"]["net"],
           {y: rep["by_year"][y]["combo"]["conc_k10"]["net"] for y in YEARS}, ok))
    return rep


def extra_2024(w, fut, S):
    """2024 IC per cell: V4 TOP100 (incl. later delistings) vs the 2026-06 survivors only vs V0."""
    import pyarrow.parquet as pq
    ts, univ = w["ts"], w["univ"]
    surv = set(pq.read_table(OLD_TRAIN, columns=["symbol"]).column(0).unique().to_pylist())
    sm = np.asarray([s in surv for s in w["syms"]])
    hrs = fit_hours(ts, pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01"))
    i24 = slice(int(ts.searchsorted(pd.Timestamp("2024-01-01"))), int(ts.searchsorted(pd.Timestamp("2025-01-01"))))
    v4, sv = ic_table(S, fut, univ, hrs), ic_table(S, fut, univ & sm[:, None], hrs)
    v0 = {}
    if os.path.exists(V0_JSON):
        v0 = json.load(open(V0_JSON, encoding="utf-8")).get("stage1", {}).get("ic_2024", {})
    nonsurv = float((univ[:, i24] & ~sm[:, None]).sum(axis=0).mean())
    return dict(sample_hours=len(hrs), top100_nonsurvivors_mean_2024=round(nonsurv, 1),
                rule="V0 = TOP100 & CONC_POOL on the old TRAIN; survivors = coins of the old TRAIN panel",
                cells={k: dict(v4_top100=v4[k]["ic"], survivors_only=sv[k]["ic"],
                               v0=(v0.get(k) or {}).get("ic"), v4_t=v4[k]["t"]) for k in CELLS})

def freeze(w, fut, S, Z, code_sha):
    ts, univ, sh = w["ts"], w["univ"], w["shard"]
    hrs = fit_hours(ts, X.TRAIN_FIRST, X.DEV_FIRST)
    ics = ic_table(S, fut, univ, hrs)
    sign, eq, sens = weights(ics)
    starts = list(X.seq_tiles(ts, pd.Timestamp("2022-01-01")))
    if TEST:
        starts = starts[:8]
    rec, _ = score(composite(Z, eq), sh, univ, starts, which=(X.K10, ("full",)))
    fz = dict(schema="FROZEN_COMBO_V1", contract="XS_BASIN_TIME_CONTRACT_V4_20261013", test=TEST,
              cells=list(CELLS), fit="rank IC, TOP100, signal hours 2021-01-01 .. 2025-12-31 (forward window "
              "ends before 2026-01-01), %d sample hours" % len(hrs), ic=ics, sign=sign, weights_main=eq,
              weights_sens=sens, active=[k for k in CELLS if sign[k]],
              composite="per hour: average rank -> z over TOP100 & finite (>= %d coins); sum w z / sum |w| over "
                        "finite active cells, >= min(%d, n_active)" % (X.MIN_COINS, MIN_CELLS),
              hold_h=X.HOLD, ruler="m1_ruler flat6 / cost0, band %s, lag %d, delisting layer v4_world" % (
                  X.R.BAND, M.LAG), code_sha16=code_sha, grid_sha16=sha16(X.GRID),
              train_2022_25_in_sample=summ(rec), dev_read=False, oos_read=False)
    blob = json.dumps(dict(cells=fz["cells"], sign=sign, w=eq, ws=sens), sort_keys=True).encode()
    fz["sha256"] = hashlib.sha256(blob).hexdigest()
    fz["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dump(fz, FROZEN)
    log("FROZEN %s sign %s in-sample K10 %.2f" % (fz["sha256"][:12], sign, fz["train_2022_25_in_sample"]["conc_k10"]["net"]))
    return fz


def dev_world():
    w = X.load(X.DEVZ)
    X.world(w, feats=False)
    ts = w["ts"]
    starts = list(X.seq_tiles(ts, X.DEV_FIRST))[:12]
    if len(starts) != 12 or ts[starts[-1] + X.E - 1] >= M.OOS_FIRST:
        raise SystemExit("DEV tiling wrong: %d tiles" % len(starts))
    S = V.handwritten(w["g"], ts)
    w["S"] = {k: S[k] for k in CELLS}
    osh, info = X.okx_shard(w["syms"], ts, w["dead"], OKX_NPZ)
    del w["g"]
    return w, starts, osh, info


def stage2(fz, w, starts, osh, info):
    ts, univ, sh, S = w["ts"], w["univ"], w["shard"], w["S"]
    Z = zsets(S, univ)
    rec, helds = score(composite(Z, fz["weights_main"]), sh, univ, starts, keep=True)
    sens, _ = score(composite(Z, fz["weights_sens"]), sh, univ, starts, which=(X.K10, ("full",)))
    mom, _ = score(S[MOM], sh, univ, starts, which=(X.K10, ("full",)))
    raw = {k: score(S[k], sh, univ, starts, which=(X.K10, ("full",)))[0] for k in CELLS}
    main_s = summ(rec)
    st = [ts[s] for s in starts]
    ok = bool(main_s["conc_k10"]["net"] > 0 and main_s["full"]["net"] > 0)
    dead_in = sum(1 for L in w["dead"].values() if starts[0] <= L < starts[-1] + X.E)
    out = dict(schema="COMBO_V1_DEV", test=TEST, frozen_sha256=fz["sha256"], n_tiles=len(starts),
               span=[str(st[0]), str(ts[starts[-1] + X.E - 1])], delistings_inside=dead_in,
               combo=main_s, combo_sens=summ(sens), mom1w=summ(mom),
               cells_raw={k: {b: summ(raw[k])[b]["net"] for b in ("conc_k10", "full")} for k in CELLS},
               tiles={b: spread(rec[b]["net"], st) for b in ("conc_k10", "full")},
               okx=info, okx_conc_k10=X.okx_settle(helds, starts, osh) if osh is not None else None,
               gate=dict(rule="DEV conc K10 net > 0 and full net > 0", conc_k10=main_s["conc_k10"]["net"],
                         full=main_s["full"]["net"], passed=ok), oos_read=False)
    o = out["okx_conc_k10"]
    log("S2 DEV K10 %.2f (t %s, %d/%d > 0) full %.2f | K5 %.2f K20 %.2f | OKX %s | sens %.2f | mom %.2f -> %s"
        % (main_s["conc_k10"]["net"], out["tiles"]["conc_k10"]["t"], out["tiles"]["conc_k10"]["n_pos"], len(starts),
           main_s["full"]["net"], main_s["conc_k5"]["net"], main_s["conc_k20"]["net"], o["net"] if o else None,
           out["combo_sens"]["conc_k10"]["net"], out["mom1w"]["conc_k10"]["net"], ok))
    return out


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    for p in (DEV_OUT, FROZEN, TRAIN_OUT):            # one run; DEV is read once
        if os.path.exists(p) and not TEST:
            raise SystemExit("%s exists; not overwritten" % p)
    code_sha = sha16(os.path.abspath(__file__))
    w = X.load(X.GRID)
    X.world(w, feats=False)
    S = V.handwritten(w["g"], w["ts"])
    S = {k: S[k] for k in CELLS}
    del w["g"]
    fut = fwd_cum(w["shard"]["fwd_wx"])
    log("V4 TRAIN loaded %.0fs: %d coins x %d h, %d delistings, code %s, test %s"
        % (time.time() - t0, len(w["syms"]), len(w["ts"]), len(w["dead"]), code_sha, TEST))
    rep = dict(schema="COMBO_V1_TRAIN", contract="XS_BASIN_TIME_CONTRACT_V4_20261013", test=TEST,
               code_sha16=code_sha, grid_sha16=sha16(X.GRID), prereg=PREREG, conventions=__doc__,
               dev_read=False, oos_read=False)
    starts_all = list(X.seq_tiles(w["ts"], pd.Timestamp("2022-01-01")))
    rep["stage0"] = stage0(w, fut, starts_all)
    if not rep["stage0"]["ok"]:
        dump(rep, TRAIN_OUT)
        log("STOP stage 0 failed")
        sys.exit(11)
    Z = zsets(S, w["univ"])
    rep["stage1"] = stage1(w, fut, S, Z)
    rep["extra_2024"] = extra_2024(w, fut, S)
    rep["seconds"] = round(time.time() - t0, 1)
    dump(rep, TRAIN_OUT)
    if not rep["stage1"]["gate"]["passed"] and not TEST:
        log("STOP stage 1 gate failed; DEV not opened")
        sys.exit(10)
    fz = freeze(w, fut, S, Z, code_sha)
    del Z, fut
    if TEST:                                        # stand-in DEV: last 4 TRAIN tiles, no OKX
        dw, dst, osh, info = dict(w, S=S), starts_all[-4:], None, None
    else:
        del w, S
        dw, dst, osh, info = dev_world()
    out = stage2(fz, dw, dst, osh, info)
    out["seconds"] = round(time.time() - t0, 1)
    out["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    dump(out, DEV_OUT)
    log("wrote %s %.0fs" % (DEV_OUT, time.time() - t0))
    sys.exit(0 if out["gate"]["passed"] else 12)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()

