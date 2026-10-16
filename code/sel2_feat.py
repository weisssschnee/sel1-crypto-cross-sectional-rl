"""SEL2 step 1b: intra-hour statistics computed on the 1-minute bars, then one value per V4 hour (XS_SEL1 10-16 afternoon).

Step 1 (sel2_minbar.py) summed or took the last bar, which collapses the hour and loses its internal structure; the
VM cannot rebuild it from hourly rows. Here each hour's 60 minute bars give one published statistic per hour.

Minute prep, inside one hour only: close is forward-filled from the hour's first traded minute (a minute with no
aggtrade = unchanged price), r_k = log c_k - log c_{k-1}, k = 1..59 (59 returns, none across the hour boundary).
An hour needs >= MIN_LIVE traded minutes, else every statistic is NaN.

  name            definition (on the hour's 59 returns r, flow q = signed_aggressor_notional, notional n)   source
  rv              sqrt(sum r^2)                                       Andersen, Bollerslev, Diebold & Labys (2003) Econometrica
  rskew           sqrt(N) sum r^3 / RV^1.5                            Amaya, Christoffersen, Jacobs & Vasquez (2015) JFE
  rkurt           N sum r^4 / RV^2                                    Amaya et al. (2015) JFE
  rs_up / rs_dn   sum r^2 1{r>0} / sum r^2 1{r<0}                     Barndorff-Nielsen, Kinnebrock & Shephard (2010)
  rsj             (rs_up - rs_dn) / RV                                Patton & Sheppard (2015) REStat
  jump_share      max(RV - BPV, 0) / RV, BPV = (pi/2) N/(N-1) sum|r_k||r_k-1|   Barndorff-Nielsen & Shephard (2004, 2006)
  amihud          mean over traded minutes of |r_k| / n_k             Amihud (2002) JFM
  kyle_lambda     OLS slope of r_k on q_k (same minute)               Kyle (1985); Hasbrouck (2009) JF
  flow_ac1        lag-1 autocorrelation of q over the hour's minutes  Lillo & Farmer (2004)
  ret_ac1         lag-1 autocorrelation of r                          Lo & MacKinlay (1988)
  vr5             sum of squared non-overlapping 5-min returns / sum r^2 (r_1..r_55)   Lo & MacKinlay (1988) RFS
  roll_spread     2 sqrt(-cov(r_k, r_k-1)), 0 if cov >= 0             Roll (1984) JF
  r_first45       log c_44 - log c_0                                  Gao, Han, Li & Zhou (2018) JFE
  r_last15        log c_59 - log c_44                                 Gao, Han, Li & Zhou (2018) JFE
  vol_last15      sum n_45..59 / sum n_0..59                          Admati & Pfleiderer (1988) RFS
  maxdd           min_k c_k / max_{j<=k} c_j - 1                      Magdon-Ismail & Atiya (2004)

Causality as step 1: V4 row h uses minutes h:00..h:59 only (labelled by open; h:59 closes at the hour's close).
Dedup as step 1: per feature the hourly max |rho| (Pearson, and Spearman because these are heavy-tailed) against
every V4 terminal, raw panel column, taker_buy_share and every step-1 column (and its log); > THR -> dropped.
TRAIN months only (2024-01..2025-12); DEV / OOS shards are not opened. numpy only, single thread.

    python -I sel2_feat.py [--out OUT.npz] [--selftest]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _p in (r"D:\ChengboRemote\runtime\m1_stage1", r"D:\riskpg_stateful_v5s_governor_nextgen_beem\src",
           r"D:\riskpg_stateful_v5s_governor_nextgen_beem\runtime"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import m1_common as C

OUT_ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\XS_SEL1"
SHARD_DIR = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\runtime\m1_staging\m1_shards"
SHARD_MONTHS = ["2024-%02d" % m for m in range(1, 13)] + ["2025-%02d" % m for m in range(1, 13)]
COLS = tuple(C.SOURCE_COLUMNS)
FEATS = ("rv", "rskew", "rkurt", "rs_up", "rs_dn", "rsj", "jump_share", "amihud", "kyle_lambda", "flow_ac1",
         "ret_ac1", "vr5", "roll_spread", "r_first45", "r_last15", "vol_last15", "maxdd")
MIN_LIVE = 30
THR = 0.95


def log(m):
    print(m, flush=True)


def _ffill_hour(c):
    """(..., 60) close with NaN for untraded minutes -> forward-filled inside the hour (leading NaNs stay)."""
    fin = np.isfinite(c)
    idx = np.where(fin, np.arange(c.shape[-1]), 0)
    np.maximum.accumulate(idx, axis=-1, out=idx)
    out = np.take_along_axis(c, idx, axis=-1)
    seen = np.maximum.accumulate(fin, axis=-1)
    return np.where(seen, out, np.nan)


def _ac1(x):
    """Lag-1 autocorrelation along the last axis, NaN-aware (pairs where both ends are finite)."""
    a, b = x[..., 1:], x[..., :-1]
    m = np.isfinite(a) & np.isfinite(b)
    n = m.sum(-1)
    a0, b0 = np.where(m, a, 0.0), np.where(m, b, 0.0)
    ma, mb = a0.sum(-1) / np.maximum(n, 1), b0.sum(-1) / np.maximum(n, 1)
    da, db = np.where(m, a - ma[..., None], 0.0), np.where(m, b - mb[..., None], 0.0)
    den = np.sqrt((da * da).sum(-1) * (db * db).sum(-1))
    return np.where((n >= 10) & (den > 0), (da * db).sum(-1) / np.where(den > 0, den, 1.0), np.nan)


def hourly(feats):
    """(A, F, n_min) minute bars, F = COLS -> (A, len(FEATS), n_min // 60) float32, + (A, H) covered.

    Hour k is built from minutes [60k, 60k + 60) and nothing else (reshape; no window crosses an hour)."""
    A, F, n = feats.shape
    if F != len(COLS) or n % 60:
        raise ValueError("feats %s vs %d cols / whole hours" % (feats.shape, len(COLS)))
    H = n // 60
    x = feats.reshape(A, F, H, 60).astype(np.float64)
    raw_c = x[:, COLS.index("close_price")]
    traded = np.isfinite(raw_c) & (raw_c > 0)
    c = _ffill_hour(np.where(traded, raw_c, np.nan))
    lc = np.log(c)
    r = lc[..., 1:] - lc[..., :-1]                               # (A, H, 59); NaN only before the first trade
    r0 = np.nan_to_num(r)
    nt = np.nan_to_num(x[:, COLS.index("notional")])
    q = np.where(traded, x[:, COLS.index("signed_aggressor_notional")], np.nan)
    ok = traded.sum(-1) >= MIN_LIVE
    N = r.shape[-1]
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        s2 = (r0 ** 2).sum(-1)
        pos = s2 > 0
        d2 = np.where(pos, s2, np.nan)
        out["rv"] = np.sqrt(s2)
        out["rskew"] = np.sqrt(N) * (r0 ** 3).sum(-1) / d2 ** 1.5
        out["rkurt"] = N * (r0 ** 4).sum(-1) / d2 ** 2
        out["rs_up"] = np.where(r0 > 0, r0 ** 2, 0.0).sum(-1)
        out["rs_dn"] = np.where(r0 < 0, r0 ** 2, 0.0).sum(-1)
        out["rsj"] = (out["rs_up"] - out["rs_dn"]) / d2
        a = np.abs(r0)
        bpv = (np.pi / 2) * N / (N - 1) * (a[..., 1:] * a[..., :-1]).sum(-1)
        out["jump_share"] = np.maximum(s2 - bpv, 0.0) / d2
        m_am = traded[..., 1:] & (nt[..., 1:] > 0) & np.isfinite(r)
        out["amihud"] = np.where(m_am, a / np.where(m_am, nt[..., 1:], 1.0), 0.0).sum(-1) / np.where(
            m_am.sum(-1) > 0, m_am.sum(-1), np.nan)
        qk = q[..., 1:]
        m_k = np.isfinite(qk) & np.isfinite(r)
        nk = m_k.sum(-1)
        qm = np.where(m_k, qk, 0.0).sum(-1) / np.maximum(nk, 1)
        rm = np.where(m_k, r, 0.0).sum(-1) / np.maximum(nk, 1)
        dq = np.where(m_k, qk - qm[..., None], 0.0)
        dr = np.where(m_k, r - rm[..., None], 0.0)
        vq = (dq * dq).sum(-1)
        out["kyle_lambda"] = np.where((nk >= 10) & (vq > 0), (dq * dr).sum(-1) / np.where(vq > 0, vq, 1.0), np.nan)
        out["flow_ac1"] = _ac1(np.where(np.isfinite(x[:, COLS.index("close_price")]), np.nan_to_num(q), 0.0))
        out["ret_ac1"] = _ac1(r)
        r55 = r0[..., :55]
        s55 = (r55 ** 2).sum(-1)
        out["vr5"] = (r55.reshape(A, H, 11, 5).sum(-1) ** 2).sum(-1) / np.where(s55 > 0, s55, np.nan)
        u, v = r0[..., 1:], r0[..., :-1]
        cov = ((u - u.mean(-1, keepdims=True)) * (v - v.mean(-1, keepdims=True))).sum(-1) / (N - 2)
        out["roll_spread"] = np.where(cov < 0, 2.0 * np.sqrt(np.abs(cov)), 0.0)
        out["r_first45"] = lc[..., 44] - lc[..., 0]
        out["r_last15"] = lc[..., 59] - lc[..., 44]
        tn = nt.sum(-1)
        out["vol_last15"] = nt[..., 45:].sum(-1) / np.where(tn > 0, tn, np.nan)
        out["maxdd"] = np.nanmin(c / np.fmax.accumulate(c, axis=-1) - 1.0, axis=-1)
    res = np.stack([np.where(ok, out[f], np.nan) for f in FEATS], axis=1)
    return res.astype(np.float32), ok


def _ref_hour(m):
    """Scalar loop reference for one hour, m = (F, 60) minute bars; mirrors the docstring table line by line."""
    cl = m[COLS.index("close_price")].astype(np.float64)
    c, last = [], np.nan
    for v in cl:
        last = v if np.isfinite(v) and v > 0 else last
        c.append(last)
    c = np.array(c)
    r = np.diff(np.log(c))
    r0 = np.nan_to_num(r)
    N = len(r)
    s2 = sum(v * v for v in r0)
    n = np.nan_to_num(m[COLS.index("notional")].astype(np.float64))
    q = m[COLS.index("signed_aggressor_notional")].astype(np.float64)
    tr = np.isfinite(cl)
    up = sum(v * v for v in r0 if v > 0)
    dn = sum(v * v for v in r0 if v < 0)
    bpv = np.pi / 2 * N / (N - 1) * sum(abs(r0[k]) * abs(r0[k - 1]) for k in range(1, N))
    am = [abs(r[k - 1]) / n[k] for k in range(1, 60) if tr[k] and n[k] > 0 and np.isfinite(r[k - 1])]
    kk = [(q[k], r[k - 1]) for k in range(1, 60) if tr[k] and np.isfinite(q[k]) and np.isfinite(r[k - 1])]
    qq, rr = np.array(kk).T
    fl = np.where(tr, np.nan_to_num(q), 0.0)
    ra = r[np.isfinite(r)]
    r5 = [sum(r0[5 * j:5 * j + 5]) for j in range(11)]
    cv = np.cov(r0[1:], r0[:-1])[0, 1]
    return dict(rv=np.sqrt(s2), rskew=np.sqrt(N) * sum(v ** 3 for v in r0) / s2 ** 1.5,
                rkurt=N * sum(v ** 4 for v in r0) / s2 ** 2, rs_up=up, rs_dn=dn, rsj=(up - dn) / s2,
                jump_share=max(s2 - bpv, 0.0) / s2, amihud=np.mean(am),
                kyle_lambda=np.polyfit(qq, rr, 1)[0], flow_ac1=np.corrcoef(fl[1:], fl[:-1])[0, 1],
                ret_ac1=np.corrcoef(ra[1:], ra[:-1])[0, 1],
                vr5=sum(v * v for v in r5) / sum(v * v for v in r0[:55]),
                roll_spread=2 * np.sqrt(-cv) if cv < 0 else 0.0,
                r_first45=np.log(c[44]) - np.log(c[0]), r_last15=np.log(c[59]) - np.log(c[44]),
                vol_last15=n[45:].sum() / n.sum(),
                maxdd=min(c[k] / max(c[j] for j in range(k + 1) if np.isfinite(c[j])) - 1
                          for k in range(60) if np.isfinite(c[k])))


def selftest():
    """Hand-made 2 coins x 3 hours: vectorised hourly() against the scalar loop _ref_hour on every hour; hour 0
    bit-identical when every minute from 60 on changes; minute 59 reaches hour 0 and nothing else (no ffill or
    return crosses the hour boundary)."""
    rng = np.random.default_rng(11)
    A, n = 2, 180
    x = rng.uniform(1.0, 9.0, (A, len(COLS), n))
    ci = COLS.index("close_price")
    x[:, ci] = 100.0 * np.exp(np.cumsum(rng.normal(0, 2e-3, (A, n)), axis=1))
    x[:, COLS.index("signed_aggressor_notional")] = rng.normal(0, 5.0, (A, n))
    gaps = rng.random((A, n)) < 0.15                            # untraded minutes: the whole bar is absent
    gaps[:, [0, 60, 120]] = False
    gaps[1, 120:123] = True                                     # coin 1, hour 2 opens untraded (leading NaN)
    x[np.repeat(gaps[:, None, :], len(COLS), 1)] = np.nan
    x = x.astype(np.float32)
    out, ok = hourly(x)
    assert ok.all()
    worst = 0.0
    for a in range(A):
        for h in range(3):
            ref = _ref_hour(x[a, :, 60 * h:60 * h + 60])
            for k, f in enumerate(FEATS):
                v, w = float(out[a, k, h]), float(ref[f])
                if not np.isfinite(w):
                    assert not np.isfinite(v), (a, h, f, v, w)
                    continue
                err = abs(v - w) / max(1e-12, abs(w))
                assert err < 1e-4, (a, h, f, v, w)
                worst = max(worst, err)
    y = x.copy()
    y[:, :, 60:] = rng.uniform(100.0, 900.0, (A, len(COLS), 120))
    o2, _ = hourly(y)
    assert np.array_equal(o2[:, :, 0], out[:, :, 0], equal_nan=True), "hour 0 moved when minutes >= 60 changed"
    assert not np.array_equal(o2[:, :, 1], out[:, :, 1], equal_nan=True)
    z = x.copy()
    z[:, :, 59] *= 1.05
    o3, _ = hourly(z)
    assert not np.array_equal(o3[:, :, 0], out[:, :, 0], equal_nan=True), "minute 59 should reach hour 0"
    assert np.array_equal(o3[:, :, 1:], out[:, :, 1:], equal_nan=True), "minute 59 leaked into hour 1"
    log("SEL2_FEAT selftest PASS: %d feats x 6 hours vs scalar reference (max rel err %.1e), hour 0 sealed from "
        "minutes >= 60, minute 59 reaches hour 0 only" % (len(FEATS), worst))


def _rank(v):
    return pd.Series(v).rank(method="average").to_numpy()


def _pearson(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100:
        return float("nan")
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT_ROOT, "SEL2_FEAT_FEATURES.npz"))
    ap.add_argument("--step1", default=os.path.join(OUT_ROOT, "SEL2_MINBAR_FEATURES.npz"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    import m2h_common as M
    import v4_world as X
    t0 = time.time()
    w = X.load(X.GRID)
    syms, ts, g = w["syms"], w["ts"], w["g"]
    si = {s: i for i, s in enumerate(syms)}
    A, T = len(syms), len(ts)
    feats = np.full((A, len(FEATS), T), np.nan, dtype=np.float32)
    covered = np.zeros((A, T), dtype=bool)
    ci = COLS.index("close_price")
    for mon in SHARD_MONTHS:
        z = np.load(os.path.join(SHARD_DIR, "m1_shard_%s.npz" % mon), allow_pickle=False)
        fm, cl = z["feats"], z["close"]
        if fm.shape[1] != len(COLS):
            raise SystemExit("%s: shard has %d cols, m1_common.SOURCE_COLUMNS %d" % (mon, fm.shape[1], len(COLS)))
        same = np.isfinite(cl)
        if not np.array_equal(fm[:, ci][same], cl[same].astype(np.float32)):
            raise SystemExit("%s: shard col %d is not close_price" % (mon, ci))
        hh, cv = hourly(fm)
        h0 = int(ts.get_loc(pd.Timestamp(mon + "-01")))
        H = min(hh.shape[2], T - h0)
        for i, s in enumerate(str(v) for v in z["symbols"]):
            j = si.get(s)
            if j is not None:
                feats[j, :, h0:h0 + H] = hh[i, :, :H]
                covered[j, h0:h0 + H] = cv[i, :H]
        del z, fm, cl, hh
        log("  %s %.0fs" % (mon, time.time() - t0))
    # ---- dedup on the covered coin-hours: V4 terminals + raw panel + taker_buy_share + step-1 columns ----
    cm = covered.ravel()
    ref = {}
    bw = M.build_world(g)
    for k, t in enumerate(M.TERMINALS):
        ref["v4:" + t] = bw["feats"][:, k].ravel()[cm]
    del bw
    for c in M.PANEL_COLS:
        ref["panel:" + c] = g[c].ravel()[cm]
    qv = g["trade_quote_volume"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ref["v4:taker_buy_share"] = np.where(qv > 0, g["taker_buy_quote_volume"] / qv, np.nan).ravel()[cm]
    z1 = np.load(a.step1, allow_pickle=False)
    if [str(s) for s in z1["symbols"]] != syms or len(z1["hours"]) != T:
        raise SystemExit("step-1 npz is not on the V4 grid")
    for k, c in enumerate(str(v) for v in z1["feature_names"]):
        v = z1["feats"][:, k].ravel()[cm].astype(np.float64)
        ref["step1:" + c] = v
        if np.nanmin(v) > 0:
            ref["step1:log " + c] = np.log(v)
    del z1
    rref = {k: _rank(v) for k, v in ref.items()}
    live = np.isfinite(g["trade_close"])
    univ = w["univ"] & live
    rows, kept, kept_r = {}, {}, {}
    for k, f in enumerate(FEATS):
        v = feats[:, k].ravel()[cm].astype(np.float64)
        rv_ = _rank(v)
        best = (0.0, None, None)
        for nm in ref:
            for how, rho in (("pearson", _pearson(v, ref[nm])), ("spearman", _pearson(rv_, rref[nm]))):
                if np.isfinite(rho) and abs(rho) > best[0]:
                    best = (abs(rho), nm, how)
        bestk = (0.0, None)
        for nm in kept:                                          # also against the step-1b features kept so far
            rho = max(abs(_pearson(v, kept[nm])), abs(_pearson(rv_, kept_r[nm])))
            if np.isfinite(rho) and rho > bestk[0]:
                bestk = (rho, nm)
        fin = np.isfinite(feats[:, k])
        drop = best[0] > THR or bestk[0] > THR
        rows[f] = dict(max_abs_rho_existing=round(best[0], 4), vs=best[1], how=best[2],
                       max_abs_rho_new=round(bestk[0], 4), vs_new=bestk[1],
                       coverage_of_covered=round(float(fin.sum() / covered.sum()), 4),
                       coverage_top100=round(float((fin & univ).sum() / univ.sum()), 4), keep=not drop)
        if not drop:
            kept[f], kept_r[f] = v, rv_
        log("  %-12s keep %-5s cov %.3f top100 %.3f | max|rho| existing %.3f %s (%s) | new %.3f %s"
            % (f, not drop, rows[f]["coverage_of_covered"], rows[f]["coverage_top100"], best[0], best[1], best[2],
               bestk[0], bestk[1]))
    keep = [f for f in FEATS if rows[f]["keep"]]
    np.savez_compressed(a.out, feats=feats, minbar_covered=covered, feature_names=np.array(FEATS),
                        keep=np.array(keep), hours=ts.values.astype("datetime64[ns]"), symbols=np.array(syms))
    rep = dict(out=a.out, shape=list(feats.shape), feats=list(FEATS), keep=keep,
               dropped=[f for f in FEATS if f not in keep], threshold=THR, min_live_minutes=MIN_LIVE,
               per_feature=rows, covered_coin_hours=int(covered.sum()), months=SHARD_MONTHS,
               coverage_vs_v4_coin_hours=float(covered.sum() / live.sum()),
               coverage_vs_v4_top100_coin_hours=float((covered & univ).sum() / univ.sum()),
               seconds=round(time.time() - t0, 1), dev_read=False, oos_read=False)
    json.dump(rep, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    log("SEL2_FEAT %s keep %d/%d drop %s | cov all %.4f top100 %.4f | %.0fs"
        % (feats.shape, len(keep), len(FEATS), rep["dropped"], rep["coverage_vs_v4_coin_hours"],
           rep["coverage_vs_v4_top100_coin_hours"], rep["seconds"]))


if __name__ == "__main__":
    main()
