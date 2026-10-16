"""BASIN5 parallel diagnostics (merged plan 10-13, not gating, TRAIN only, CPU). One file, two parts.

 regime  the handwritten H168 cells (m2h_verdict_dense.handwritten, TOP100, lag 1) scored per hour by
         the 168 h forward rank IC (D.cs_spearman_rows on fwd_w cumulated, as D.ic_tile), every 24 h,
         grouped by a 2x2 regime fixed before looking: BTC 720 h trailing log return > 0 or not x BTC
         720 h trailing realised vol (hourly log returns) above or below its TRAIN median. Reported
         per cell: mean IC, t (24 h steps, overlapping horizon so t is optimistic), n hours, and
         whether the sign flips against the all-hours mean.
 sector  rolling return-correlation clusters, past data only: every 720 h (from hour 1440), the
         TOP100 coins' trailing 720 h hourly log-return correlation, average-linkage hierarchical
         clustering on 1 - corr cut at 0.5, clusters of >= 4 coins. Lead-lag test inside a cluster:
         leader = the top-third coins by the cluster's previous 24 h return, laggards = the rest;
         signal for laggards = cluster mean 24 h return minus own 24 h return (catch-up), scored as
         the 24 h forward rank IC among the laggards, pooled over clusters per hour, every 24 h.
         The null is the same score with cluster labels shuffled within each refit (seed fixed).
Writes ROOT\\artifacts\\m2h_v1\\B5_GATE1\\DIAG_REGIME_SECTOR.json. No DEV, no OOS.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b5_fixed as F

OUT = os.path.join(F.A_ROOT, "B5_GATE1", "DIAG_REGIME_SECTOR%s.json" % ("_V2" if os.environ.get("B5_DIAG_V2") == "1" else ""))
WIN = 720
STEP = 24
H_IC = 168
SEED = 20261113
RESID = os.environ.get("B5_DIAG_V2") == "1"     # v2: market-residual returns, 10 clusters (maxclust)
K_CL = 10


def log(m):
    print(m, flush=True)


def fwd_cum(fwd_w, h):
    """(A, T): log sum of the next h hourly (winsorised) returns starting at t (t .. t+h-1)."""
    lr = np.log1p(np.nan_to_num(fwd_w, nan=0.0))
    cs = np.concatenate([np.zeros((lr.shape[0], 1)), np.cumsum(lr, axis=1)], axis=1)
    T = lr.shape[1]
    out = np.full(lr.shape, np.nan)
    out[:, :T - h + 1] = cs[:, h:] - cs[:, :T - h + 1]
    return out


def ic_series(sig, fut, mask, hours, D):
    """Rank IC per signal hour t (signal at t-1 -> forward return from t, as lag 1)."""
    a = np.where(mask[:, hours - 1], sig[:, hours - 1], np.nan).T
    b = np.where(mask[:, hours - 1], fut[:, hours], np.nan).T
    return D.cs_spearman_rows(a, b)


def tstat(v):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return float(v.mean() / (sd / np.sqrt(len(v)))) if sd > 0 else float("nan")


def main():
    t_all = time.time()
    TR, M, D, W, R = F._imports()
    import m2h_verdict_dense as V
    man, arr = F.load_cache()
    syms, ts, grids = M.load_train_grids(TR.PANEL)
    assert list(syms) == list(man["symbols"])
    univ = np.array(arr["univ"])
    sh = F.make_shard("real", arr)
    T = univ.shape[1]
    rep = dict(schema="B5_DIAG_REGIME_SECTOR_V1", train_only=True, dev_read=False, oos_read=False)

    # ---------------------------------------------------------------- regime
    close = grids["trade_close"]
    bi = list(syms).index("BTCUSDT")
    btc = np.log(close[bi])
    r1 = np.diff(btc, prepend=np.nan)
    ret30 = pd.Series(btc).diff(WIN).to_numpy()
    vol30 = pd.Series(r1).rolling(WIN, min_periods=WIN // 2).std().to_numpy()
    med = float(np.nanmedian(vol30))
    hours = np.arange(max(M.LOOKBACK, WIN) + 1, T - H_IC, STEP)
    up = ret30[hours - 1] > 0
    hiv = vol30[hours - 1] > med
    reg = {"up_lowvol": up & ~hiv, "up_highvol": up & hiv, "down_lowvol": ~up & ~hiv, "down_highvol": ~up & hiv}
    S = V.handwritten(grids, ts)
    fut = fwd_cum(sh["fwd_w"], H_IC)
    cells = {}
    for name in ("T1_mom_168h", "CTREND_lite_17") + V.T2_CELLS + V.T3_CELLS:
        ic = ic_series(S[name], fut, univ, hours, D)
        allm = float(np.nanmean(ic))
        row = dict(all=dict(ic=round(allm, 4), t=round(tstat(ic), 2), n=int(len(ic))))
        for k, m in reg.items():
            v = ic[m]
            mm = float(np.nanmean(v)) if len(v) else float("nan")
            row[k] = dict(ic=round(mm, 4), t=round(tstat(v), 2), n=int(m.sum()),
                          sign_flip=bool(np.isfinite(mm) and np.sign(mm) != np.sign(allm)))
        cells[name] = row
        log("regime %s all %.4f | %s" % (name, allm, {k: row[k]["ic"] for k in reg}))
    rep["regime"] = dict(rule="BTC 720 h log return > 0 x BTC 720 h realised vol > TRAIN median (%.6f); "
                              "168 h forward rank IC every 24 h, TOP100, lag 1; t uses overlapping "
                              "horizons, so it overstates" % med,
                         hours_per_regime={k: int(m.sum()) for k, m in reg.items()}, cells=cells)
    del S

    # ---------------------------------------------------------------- sector
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    lr = np.diff(np.log(close), axis=1, prepend=np.nan)
    r24 = pd.DataFrame(np.log(close).T).diff(24).to_numpy().T
    fut24 = fwd_cum(sh["fwd_w"], 24)
    rng = np.random.default_rng(SEED)
    ics, ics_null, n_cl, sizes = [], [], [], []
    for t0 in range(2 * WIN, T - 24, WIN):
        live = univ[:, t0 - 1] & np.isfinite(lr[:, t0 - WIN:t0]).mean(axis=1).__ge__(0.9)
        idx = np.flatnonzero(live)
        if len(idx) < 20:
            continue
        Y = lr[idx, t0 - WIN:t0]
        if RESID:                              # v2: market factor out first (v1 collapsed to ~1 cluster)
            mk = np.nanmean(Y, axis=0)
            Yc = np.where(np.isfinite(Y), Y, np.nan)
            b = np.nansum((Yc - np.nanmean(Yc, axis=1, keepdims=True)) * (mk - mk.mean()), axis=1) / np.nansum((mk - mk.mean()) ** 2)
            Y = Yc - b[:, None] * mk[None, :]
        X = pd.DataFrame(Y.T)
        C = X.corr(min_periods=WIN // 2).to_numpy()
        C = np.nan_to_num(C, nan=0.0)
        np.fill_diagonal(C, 1.0)
        Dm = np.clip(1.0 - C, 0.0, 2.0)
        Z = linkage(squareform(Dm, checks=False), "average")
        lab = fcluster(Z, K_CL, criterion="maxclust") if RESID else fcluster(Z, 0.5, criterion="distance")
        labs = {"real": lab, "null": rng.permutation(lab)}
        uniq, cnt = np.unique(lab, return_counts=True)
        n_cl.append(int((cnt >= 4).sum()))
        sizes += [int(c) for c in cnt if c >= 4]
        for t in range(t0, min(t0 + WIN, T - 24), STEP):
            for kind, lb in labs.items():
                sig_all, fut_all = [], []
                for c in np.unique(lb):
                    mem = idx[lb == c]
                    if len(mem) < 4:
                        continue
                    rr = r24[mem, t - 1]
                    ok = np.isfinite(rr) & univ[mem, t - 1]
                    if ok.sum() < 4:
                        continue
                    m2, rr2 = mem[ok], rr[ok]
                    cut = np.quantile(rr2, 2 / 3)
                    lag = rr2 < cut
                    sig_all += list(np.mean(rr2) - rr2[lag])
                    fut_all += list(fut24[m2[lag], t])
                if len(sig_all) >= 20:
                    v = D.cs_spearman_rows(np.asarray([sig_all]), np.asarray([fut_all]))[0]
                    (ics if kind == "real" else ics_null).append(float(v))
    rep["sector"] = dict(version="v2 market-residual, maxclust %d" % K_CL if RESID else "v1",
                         rule="720 h rolling return-corr clusters (avg linkage, cut 1-corr 0.5, >= 4 coins), "
                              "refit every 720 h on past data; laggard catch-up IC (24 h fwd) pooled per hour "
                              "every 24 h; null = labels shuffled within each refit",
                         refits=len(n_cl), clusters_per_refit_mean=round(float(np.mean(n_cl)), 2) if n_cl else None,
                         cluster_size_median=int(np.median(sizes)) if sizes else None,
                         ic_mean=round(float(np.mean(ics)), 4) if ics else None, ic_t=round(tstat(ics), 2),
                         n_hours=len(ics), null_ic_mean=round(float(np.mean(ics_null)), 4) if ics_null else None,
                         null_ic_t=round(tstat(ics_null), 2))
    log("sector real IC %s t %s | null %s t %s" % (rep["sector"]["ic_mean"], rep["sector"]["ic_t"],
                                                   rep["sector"]["null_ic_mean"], rep["sector"]["null_ic_t"]))
    rep["seconds"] = round(time.time() - t_all, 1)
    rep["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log("DIAG done %.0fs" % (time.time() - t_all))


if __name__ == "__main__":
    main()
