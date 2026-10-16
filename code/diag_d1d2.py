"""D1 / D2 regime diagnostics (QuantMind 10-14 order with the revised D1; not gating, TRAIN only, CPU).
New file; b5_diag_regime_sector (V2) is the definition source and is not modified.

 regime  V2's 2x2 exactly: BTC 720 h log return > 0 x BTC 720 h realised vol > 0.005196 (V2's TRAIN
         median, recomputed bit-exact on the old TRAIN, never refit); 168 h forward rank IC every 24 h,
         TOP100, lag 1 (V2's ic_series). Segment = run of consecutive 24 h samples in one state.
         t = mean / sd of a segment block bootstrap (segments resampled with replacement per state).
 D1      old TRAIN (2024-25): per cell x state x year IC, segments, samples; sign stability across years;
         for the 4 cells COMBO_V0 flipped, actual yearly IC vs the IC implied by that year's state mix
         (pooled per-state ICs weighted by the year's state shares).
 H1/H2   pre-registered (PREREG below, written to PREREG_H1H2.json before any number is computed),
         tested on the V4 grid, signal hours 2021-01-01 .. t + 168 h < 2024-01-01.
 D2      old TRAIN: per cell, cross-sectional OLS of the cell's rank z on the coin's trailing 720 h beta
         (vs BTC, and vs the TOP100 equal-weight market), residual -> IC per state; plus the realised
         beta of each cell's K10 book per state (v4_world.book_sig K10 -> held_of -> ruler ffill,
         hourly gross book return on BTC / EW market returns, OLS per state, 50 tiles).
No DEV, no OOS, no state label on any DEV hour. Output ROOT\\artifacts\\m2h_v1\\DIAG_D1D2\\, refused if present.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import b5_fixed as F

OUT_DIR = os.path.join(F.A_ROOT, "DIAG_D1D2")
OUT = os.path.join(OUT_DIR, "DIAG_D1D2.json")
PREREG_OUT = os.path.join(OUT_DIR, "PREREG_H1H2.json")
V2_JSON = os.path.join(F.A_ROOT, "B5_GATE1", "DIAG_REGIME_SECTOR_V2.json")
WIN, STEP, H_IC = 720, 24, 168
THR_V2 = 0.005196
B = 10000
SEED = 20261014
CELLS = ("T1_mom_168h", "CTREND_lite_17", "T2_neg_funding_7d", "T2_neg_retail_LS_24h", "T2_neg_dlogOI_7d",
         "T3_neg_premium_24h", "T3_neg_mark_index_24h", "T3_abs_premium_high_1h")
FLIP4 = ("T1_mom_168h", "CTREND_lite_17", "T2_neg_dlogOI_7d", "T3_neg_mark_index_24h")
STATES = ("up_lowvol", "up_highvol", "down_lowvol", "down_highvol")
YEARS = (2024, 2025)
PREREG = dict(
    source="QuantMind 训练特征构筑 session 10-14 (D1 revision); fixed before any 2021-23 state number exists",
    H1=dict(cell="T2_neg_funding_7d", state="up_highvol", absval=True,
            claim="|mean IC| in up_highvol < |mean IC| of the other three states pooled"),
    H2=dict(cell="T1_mom_168h", state="down_highvol", absval=False,
            claim="mean IC in down_highvol < mean IC of the other three states pooled"),
    span="V4 grid (fwd_wx, delisting layer), TOP100, lag 1, signal hours 2021-01-01 00:00 .. t + 168 h < 2024-01-01",
    threshold="0.005196 = V2 TRAIN median, recomputed exactly on the old TRAIN; state rule, window, cells as V2",
    test="one-sided segment block bootstrap, B 10000, seed 20261015: d = f(m_state) - f(m_rest), f = abs for H1, "
         "identity for H2; rest = the three other states, each resampled separately then pooled; "
         "p = P*(d* >= 0); pass if p < 0.05",
    untestable="any of the four states with < 4 segments in the span -> 测不了, not 没过",
    excluded="OI / LS cells are not tested (2021 about 10 % present)",
    disclosure="COMBO_V1 had already computed 2021-23 per-year cell ICs on V4 (not split by state) before this",
    use="pass -> fixed state switch (state weight x 0) in the V4 weight layer, COMBO_V2 only, nothing tuned; "
        "both fail -> regime line closed, only D2 beta neutral kept; no retry with other definitions")


def log(m):
    print(m, flush=True)


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def fwd_cum(fw, h=H_IC):
    lr = np.log1p(np.nan_to_num(fw, nan=0.0))
    cs = np.concatenate([np.zeros((lr.shape[0], 1)), np.cumsum(lr, axis=1)], axis=1)
    T = lr.shape[1]
    out = np.full(lr.shape, np.nan)
    out[:, :T - h + 1] = cs[:, h:] - cs[:, :T - h + 1]
    return out


def ic_series(sig, fut, mask, hours, D):
    """V2 verbatim: rank IC per signal hour t (signal at t-1 -> forward return from t)."""
    a = np.where(mask[:, hours - 1], sig[:, hours - 1], np.nan).T
    b = np.where(mask[:, hours - 1], fut[:, hours], np.nan).T
    return D.cs_spearman_rows(a, b)


def btc_state(close_btc):
    btc = np.log(close_btc)
    r1 = np.diff(btc, prepend=np.nan)
    ret30 = pd.Series(btc).diff(WIN).to_numpy()
    vol30 = pd.Series(r1).rolling(WIN, min_periods=WIN // 2).std().to_numpy()
    return ret30, vol30


def labels(ret30, vol30, idx, thr):
    """State index (into STATES) at absolute hours idx; NaN compares False, as in V2."""
    up = ret30[idx] > 0
    hiv = vol30[idx] > thr
    lab = np.where(up, np.where(hiv, 1, 0), np.where(hiv, 3, 2))
    return np.where(np.isfinite(ret30[idx]) & np.isfinite(vol30[idx]), lab, -1)   # -1 = no state (warm-up)


def segments(lab, brk=None):
    new = np.r_[True, lab[1:] != lab[:-1]]
    if brk is not None:
        new |= np.r_[True, brk[1:] != brk[:-1]]
    return np.cumsum(new) - 1


def seg_sums(ic, seg, sel):
    v, s = ic[sel], seg[sel]
    ok = np.isfinite(v)
    if not ok.any():
        return np.zeros(0), np.zeros(0)
    _, inv = np.unique(s[ok], return_inverse=True)
    return np.bincount(inv, weights=v[ok]), np.bincount(inv).astype(float)


def boot_mean(groups, rng):
    num, den = np.zeros(B), np.zeros(B)
    for su, cn in groups:
        if len(su) == 0:
            continue
        d = rng.integers(0, len(su), size=(B, len(su)))
        num += su[d].sum(axis=1)
        den += cn[d].sum(axis=1)
    return num / den


def cell_stat(ic, seg, sel, rng):
    su, cn = seg_sums(ic, seg, sel)
    out = dict(ic=None, t=None, n_seg=int(len(su)), n_samples=int(cn.sum()))
    if len(su) == 0:
        return out
    m = float(su.sum() / cn.sum())
    out["ic"] = round(m, 4)
    if len(su) >= 2:
        sd = float(boot_mean([(su, cn)], rng).std())
        out["t"] = round(m / sd, 2) if sd > 0 else None
    return out
def stats_by(ic, lab, yr, rng):
    """D1: pooled / per state / per state x year; segments split at year ends for the yearly cells."""
    seg_all, seg_y = segments(lab), segments(lab, yr)
    out = dict(all=cell_stat(ic, seg_all, np.ones(len(ic), bool), rng),
               by_year={str(y): cell_stat(ic, seg_y, yr == y, rng) for y in YEARS}, states={})
    for k, s in enumerate(STATES):
        st = dict(pooled=cell_stat(ic, seg_all, lab == k, rng),
                  by_year={str(y): cell_stat(ic, seg_y, (lab == k) & (yr == y), rng) for y in YEARS})
        ys = [st["by_year"][str(y)]["ic"] for y in YEARS]
        st["sign_stable"] = None if None in ys else bool(np.sign(ys[0]) == np.sign(ys[1]))
        out["states"][s] = st
    return out


def mix_explain(row, shares):
    """Actual yearly IC vs the IC implied by the year's state mix at the pooled per-state ICs."""
    pooled = {s: row["states"][s]["pooled"]["ic"] or 0.0 for s in STATES}
    imp = {y: sum(shares[y][s] * pooled[s] for s in STATES) for y in shares}
    act = {y: row["by_year"][y]["ic"] for y in shares}
    within = sum(shares["2025"][s] * ((row["states"][s]["by_year"]["2025"]["ic"] or 0.0)
                                      - (row["states"][s]["by_year"]["2024"]["ic"] or 0.0)) for s in STATES)
    d_act, d_imp = act["2025"] - act["2024"], imp["2025"] - imp["2024"]
    return dict(actual={y: round(v, 4) for y, v in act.items()}, implied_by_mix={y: round(v, 4) for y, v in imp.items()},
                change_actual=round(d_act, 4), change_from_mix=round(d_imp, 4), change_within_states=round(within, 4),
                frac_explained_by_mix=round(d_imp / d_act, 3) if d_act else None)


def roll_beta(lr, x, idx):
    """(A, T) coin hourly log returns on factor x (T,), OLS beta over [h - 719, h] for h in idx; >= 360 obs."""
    out = np.full((lr.shape[0], len(idx)), np.nan)
    for j, h in enumerate(idx):
        Y, xx = lr[:, h - WIN + 1:h + 1], x[h - WIN + 1:h + 1]
        m = np.isfinite(Y) & np.isfinite(xx)[None, :]
        n = m.sum(axis=1)
        xm, ym = np.where(m, xx[None, :], 0.0), np.where(m, Y, 0.0)
        nn = np.maximum(n, 1)
        mx, my = xm.sum(axis=1) / nn, ym.sum(axis=1) / nn
        cov = (xm * ym).sum(axis=1) / nn - mx * my
        var = (xm * xm).sum(axis=1) / nn - mx * mx
        ok = (n >= WIN // 2) & (var > 0)
        out[ok, j] = cov[ok] / var[ok]
    return out


def resid_rows(z, b, min_coins=20):
    """Per row OLS (intercept) of z on b over jointly finite coins -> residual, slope; thin rows NaN."""
    m = np.isfinite(z) & np.isfinite(b)
    cnt = np.maximum(m.sum(axis=1), 1)
    zm = np.where(m, z, 0.0).sum(axis=1) / cnt
    bm = np.where(m, b, 0.0).sum(axis=1) / cnt
    dz = np.where(m, z - zm[:, None], 0.0)
    db = np.where(m, b - bm[:, None], 0.0)
    sxx = (db * db).sum(axis=1)
    slope = np.where(sxx > 0, (dz * db).sum(axis=1) / np.where(sxx > 0, sxx, 1.0), np.nan)
    r = np.where(m, dz - slope[:, None] * db, np.nan)
    thin = m.sum(axis=1) < min_coins
    r[thin] = np.nan
    slope[thin] = np.nan
    return r, slope


def h_test(ic, lab, rng, k_state, absval):
    """Pre-registered one-sided segment block bootstrap: f(m_state) < f(m_rest)."""
    seg = segments(lab)
    grp = {k: seg_sums(ic, seg, lab == k) for k in range(4)}
    per = {STATES[k]: dict(ic=round(float(su.sum() / cn.sum()), 4) if len(su) else None, n_seg=int(len(su)),
                           n_samples=int(cn.sum()), n_hours=int(cn.sum()) * STEP) for k, (su, cn) in grp.items()}
    out = dict(per_state=per)
    if any(len(grp[k][0]) < 4 for k in range(4)):
        out["result"] = "测不了"
        return out
    f = np.abs if absval else (lambda v: v)
    rest = [grp[k] for k in range(4) if k != k_state]
    ms = grp[k_state][0].sum() / grp[k_state][1].sum()
    mr = sum(g[0].sum() for g in rest) / sum(g[1].sum() for g in rest)
    dstar = f(boot_mean([grp[k_state]], rng)) - f(boot_mean(rest, rng))
    p = float(np.mean(dstar >= 0))
    out.update(m_state=round(float(ms), 4), m_rest=round(float(mr), 4), d=round(float(f(ms) - f(mr)), 4),
               p_one_sided=round(p, 4), result="过" if p < 0.05 else "没过")
    return out
def k10_beta(S, univ, sh, starts, lab_of, x_btc, x_ew, X, R, E):
    """Realised beta of each cell's K10 book per state: hourly ruler gross vs BTC / TOP100 EW returns."""
    out = {}
    for k in CELLS:
        sigm = np.where(univ, S[k], np.nan)
        rec = dict(book=[], btc=[], ew=[], st=[])
        for s0 in starts:
            bsig = X.book_sig(X.tile_rows(sigm, s0), sh, s0, X.K10)
            held = X.held_of(bsig, sh, s0, layer=False)
            fw = sh["fwd_w"][:, s0:s0 + E].T
            rec["book"].append(np.nansum(held * np.nan_to_num(fw), axis=1))
            rec["btc"].append(x_btc[s0:s0 + E])
            rec["ew"].append(x_ew[s0:s0 + E])
            rec["st"].append(lab_of(np.arange(s0, s0 + E) - 1))
        v = {f: np.concatenate(a) for f, a in rec.items()}
        row = {}
        for j, s in enumerate(STATES + ("all",)):
            m = (v["st"] == j) if s != "all" else np.ones(len(v["st"]), bool)
            row[s] = dict(hours=int(m.sum()))
            for f in ("btc", "ew"):
                ok = m & np.isfinite(v[f]) & np.isfinite(v["book"])
                xx, yy = v[f][ok], v["book"][ok]
                vv = float(xx.var())
                row[s]["beta_" + f] = round(float(np.cov(yy, xx, bias=True)[0, 1] / vv), 4) if vv > 0 else None
        out[k] = row
    return out


def main():
    t0 = time.time()
    for p in (OUT, PREREG_OUT):
        if os.path.exists(p):
            raise SystemExit("%s exists; not overwritten" % p)
    os.makedirs(OUT_DIR, exist_ok=True)
    pre = dict(PREREG, written=time.strftime("%Y-%m-%dT%H:%M:%S%z"), code_sha16=sha16(os.path.abspath(__file__)))
    json.dump(pre, open(PREREG_OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log("prereg written %s" % sha16(PREREG_OUT))
    TR, M, D, W, R = F._imports()
    import m2h_verdict_dense as V
    import v4_world as X
    E = M.EPISODE
    rng = np.random.default_rng(SEED)
    rep = dict(schema="DIAG_D1D2_V1", prereg_sha16=sha16(PREREG_OUT), dev_read=False, oos_read=False,
               dev_state_labels=False, gating=False)

    # ------------------------------------------------------------ old TRAIN (2024-25): D1 table, D2
    man, arr = F.load_cache()
    syms, ts, grids = M.load_train_grids(TR.PANEL)
    assert list(syms) == list(man["symbols"])
    univ = np.array(arr["univ"])
    sh = F.make_shard("real", arr)
    T = univ.shape[1]
    close = grids["trade_close"]
    bi = list(syms).index("BTCUSDT")
    ret30, vol30 = btc_state(close[bi])
    thr = float(np.nanmedian(vol30))
    if round(thr, 6) != THR_V2:
        raise SystemExit("V2 threshold not reproduced: %.8f" % thr)
    hours = np.arange(max(M.LOOKBACK, WIN) + 1, T - H_IC, STEP)
    lab = labels(ret30, vol30, hours - 1, thr)
    yr = np.asarray(ts[hours].year)
    S = V.handwritten(grids, ts)
    fut = fwd_cum(sh["fwd_w"], H_IC)
    ics = {k: ic_series(S[k], fut, univ, hours, D) for k in CELLS}
    v2chk = None
    if os.path.exists(V2_JSON):
        v2 = json.load(open(V2_JSON, encoding="utf-8"))["regime"]["cells"]
        v2chk = max(abs(round(float(np.nanmean(ics[k][lab == j])), 4) - v2[k][s]["ic"])
                    for k in CELLS for j, s in enumerate(STATES))
    shares = {str(y): {s: round(float(np.mean(lab[yr == y] == j)), 4) for j, s in enumerate(STATES)} for y in YEARS}
    d1 = {k: stats_by(ics[k], lab, yr, rng) for k in CELLS}
    rep["d1"] = dict(span="old TRAIN 2024-01..2025-12 (V3 panel, 489 coins)", threshold=thr, v2_ic_max_diff=v2chk,
                     samples=int(len(hours)), state_share_by_year=shares,
                     state_hours_by_year={str(y): {s: int(((lab == j) & (yr == y)).sum()) * STEP
                                                   for j, s in enumerate(STATES)} for y in YEARS},
                     segments_by_year={str(y): {s: int(len(seg_sums(np.ones(len(lab)), segments(lab, yr), (lab == j) & (yr == y))[0]))
                                                for j, s in enumerate(STATES)} for y in YEARS},
                     cells=d1, flip4_mix={k: mix_explain(d1[k], shares) for k in FLIP4},
                     t_rule="segment block bootstrap B %d" % B)
    log("D1 done %.0fs v2 diff %s" % (time.time() - t0, v2chk))
    lr = np.diff(np.log(close), axis=1, prepend=np.nan)
    x_btc = lr[bi]
    with np.errstate(all="ignore"):
        x_ew = np.nanmean(np.where(univ, lr, np.nan), axis=0)
    seg_all = segments(lab)
    d2 = {}
    for fac, x in (("btc", x_btc), ("ew", x_ew)):
        beta = roll_beta(lr, x, hours - 1).T
        fb = np.where(univ[:, hours - 1].T, fut[:, hours].T, np.nan)
        for k in CELLS:
            z = X.zrank_rows(np.where(univ[:, hours - 1] & np.isfinite(S[k][:, hours - 1]), S[k][:, hours - 1], np.nan).T)
            res, slope = resid_rows(z, beta)
            icr = D.cs_spearman_rows(res, fb)
            icr = np.where(np.isfinite(res).sum(axis=1) >= D.IC_MIN_COINS, icr, np.nan)
            row = d2.setdefault(k, {})
            row[fac] = {s: dict(cell_stat(icr, seg_all, lab == j, rng),
                                ic_raw=d1[k]["states"][s]["pooled"]["ic"],
                                slope_on_beta=round(float(np.nanmean(slope[lab == j])), 4))
                        for j, s in enumerate(STATES)}
            row[fac]["all"] = dict(cell_stat(icr, seg_all, np.ones(len(lab), bool), rng), ic_raw=d1[k]["all"]["ic"])
    starts = M.tile_starts(T)
    rep["d2"] = dict(rule="per sample hour OLS of the cell's TOP100 rank z on trailing 720 h beta (>= 360 obs), "
                          "residual 168 h forward rank IC", cells=d2,
                     k10_realised_beta=k10_beta(S, univ, sh, starts, lambda i: labels(ret30, vol30, i, thr),
                                                np.r_[x_btc[1:], np.nan], np.r_[x_ew[1:], np.nan], X, R, E))
    log("D2 done %.0fs" % (time.time() - t0))
    del S, grids, sh, fut, arr

    # ------------------------------------------------------------ V4 2021-23: pre-registered H1 / H2
    w = X.load(X.GRID)
    X.world(w, feats=False)
    ts4 = w["ts"]
    S4 = V.handwritten(w["g"], ts4)
    b4 = w["syms"].index("BTCUSDT")
    r4, v4 = btc_state(w["g"]["trade_close"][b4])
    del w["g"]
    i0, i1 = int(ts4.get_loc(pd.Timestamp("2021-01-01"))), int(ts4.get_loc(pd.Timestamp("2024-01-01")))
    h4 = np.arange(i0, i1 - H_IC, STEP)
    lab4 = labels(r4, v4, h4 - 1, thr)
    fut4 = fwd_cum(w["shard"]["fwd_wx"])
    rng_h = np.random.default_rng(20261015)
    hres = {}
    for key in ("H1", "H2"):
        p = PREREG[key]
        ic4 = ic_series(S4[p["cell"]], fut4, w["univ"], h4, D)
        hres[key] = dict(cell=p["cell"], state=p["state"], claim=p["claim"],
                         **h_test(ic4, lab4, rng_h, STATES.index(p["state"]), p["absval"]))
        log("%s %s" % (key, hres[key].get("result")))
    rep["h1h2"] = dict(span="V4 %s .. %s (signal hours, +168 h < 2024-01-01)" % (ts4[h4[0]], ts4[h4[-1]]),
                       samples=int(len(h4)), state_share={s: round(float(np.mean(lab4 == j)), 4)
                                                          for j, s in enumerate(STATES)},
                       tests=hres, use=PREREG["use"])
    rep["d3"] = "not triggered: COMBO_V1 failed its TRAIN gate, DEV was not read, no DEV state label computed"
    rep["seconds"] = round(time.time() - t0, 1)
    rep["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    rep["code_sha16"] = sha16(os.path.abspath(__file__))
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    log("DIAG_D1D2 done %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
