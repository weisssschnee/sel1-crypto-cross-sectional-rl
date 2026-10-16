"""SEL2 step 2: sector clustering from data, no external labels (XS_SEL1 10-16 morning supplement).

Every month-start, cluster TOP100 into k groups using trailing 720h β-hedged residual returns, then hold that
grouping for the next month. The grouping is a (A, T) int8 grid; coins outside TOP100 or months without enough
history are assigned group -1 (no group). No CoinGecko or external taxonomy: those are today's labels and leak
future information. Causality: month M's grouping uses only data before M (β hedge trailing 720h, correlation on
the residuals, hierarchical clustering).

Three VM operators for the sector dimension:
  GRP_DEMEAN(x)  group-mean centered x
  GRP_RANK(x)    within-group rank of x (ties average, as in sel1_common.rank_weights)
  GRP_MEAN(x)    group mean of x (sector momentum or rotation)

Self-check:
  - grouping at month M uses only hours < M
  - hand-calculated 3-hour group rank matches vectorised output
  - per-month group-size distribution and ARI (adjusted Rand index) between adjacent months

    python -I sel2_sector.py [--out OUT.npz] [--k K] [--selftest]
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def adjusted_rand_score(a, b):
    """Hubert-Arabie ARI, same formula as sklearn.metrics.adjusted_rand_score (m1cu128 has no sklearn)."""
    _, ia = np.unique(np.asarray(a), return_inverse=True)
    _, ib = np.unique(np.asarray(b), return_inverse=True)
    n = len(ia)
    if n < 2:
        return 1.0
    ct = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(ct, (ia, ib), 1)
    c2 = lambda x: (x * (x - 1) / 2.0).sum()
    s_ij, s_a, s_b = c2(ct), c2(ct.sum(axis=1)), c2(ct.sum(axis=0))
    exp = s_a * s_b / (n * (n - 1) / 2.0)
    mx = (s_a + s_b) / 2.0
    return 1.0 if mx == exp else float((s_ij - exp) / (mx - exp))


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT_ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\XS_SEL1"
LOOKBACK = 720
TRAIN_MONTHS = ["%d-%02d-01" % (y, m) for y in [2024, 2025] for m in range(1, 13)]


def log(m):
    print(m, flush=True)


MIN_OBS = 360                                   # a coin needs half the 720h window to be clustered


def _beta_hedge_residuals(ret, mkt, eligible):
    """(W, A) window returns and (W,) market -> (W, A) residuals y - β·m, one β per coin over the window.

    Only eligible, finite hours enter β and the residuals; everything else is NaN."""
    ok = eligible & np.isfinite(ret) & np.isfinite(mkt)[:, None]
    n = ok.sum(axis=0)
    m = np.where(ok, mkt[:, None], 0.0)
    y = np.where(ok, ret, 0.0)
    nn = np.maximum(n, 1)
    mm, my = m.sum(axis=0) / nn, y.sum(axis=0) / nn
    cov = (m * y).sum(axis=0) / nn - mm * my
    var = (m * m).sum(axis=0) / nn - mm * mm
    beta = cov / (var + 1e-12)
    return np.where(ok & (n >= MIN_OBS)[None, :], ret - beta[None, :] * mkt[:, None], np.nan)


def _corr_matrix(res):
    """(W, A) residuals -> (A, A) pairwise Pearson ρ over hours where both are finite (NaN if < 100)."""
    f = np.isfinite(res).astype(np.float64)
    x = np.where(f > 0, res, 0.0)
    n = f.T @ f
    sx = x.T @ f                                  # sx[i, j] = Σ x_i over hours where j is also finite
    sxx = (x * x).T @ f
    sxy = x.T @ x
    with np.errstate(invalid="ignore", divide="ignore"):
        num = n * sxy - sx * sx.T
        den = np.sqrt((n * sxx - sx * sx) * (n * sxx.T - sx.T * sx.T))
        c = num / den
    return np.where(n >= 100, c, np.nan)


def cluster_one_month(ret, mkt, eligible, univ_mask, k):
    """Cluster TOP100 coins in univ_mask using trailing 720h β-hedged residuals -> group assignments (A,) int8.

    Coins not in univ_mask or without enough data get -1. Hierarchical complete linkage on 1 − |ρ|."""
    A = ret.shape[1]
    grp = np.full(A, -1, dtype=np.int8)
    res = _beta_hedge_residuals(ret[-LOOKBACK:], mkt[-LOOKBACK:], eligible[-LOOKBACK:])
    c = _corr_matrix(res)
    cands = np.where(univ_mask & np.isfinite(res).any(axis=0))[0]
    # a coin whose ρ with any other candidate is undefined (too little overlap) is left ungrouped
    while len(cands) >= k and not np.isfinite(c[np.ix_(cands, cands)]).all():
        bad = (~np.isfinite(c[np.ix_(cands, cands)])).sum(axis=1)
        cands = np.delete(cands, int(np.argmax(bad)))
    if len(cands) < k:
        return grp
    sub = c[np.ix_(cands, cands)]
    dist = 1.0 - np.abs(sub)
    np.fill_diagonal(dist, 0.0)
    try:
        z = linkage(squareform(dist, checks=False), method="complete")
        labels = fcluster(z, k, criterion="maxclust")
    except Exception:
        return grp
    grp[cands] = labels.astype(np.int8)
    return grp


def build_sectors(g, univ, ts, k):
    """V4 grid -> (A, T) int8 sector assignments, updated at month-start using trailing 720h."""
    close = g["trade_close"]
    A, T = close.shape
    ret = np.full((T, A), np.nan)
    ret[1:] = (close[:, 1:] / close[:, :-1] - 1.0).T
    mkt = np.nanmean(np.where(univ.T, ret, np.nan), axis=1)
    el = np.isfinite(ret) & univ.T
    sectors = np.full((A, T), -1, dtype=np.int8)
    month_starts = pd.DatetimeIndex([pd.Timestamp(m) for m in TRAIN_MONTHS])
    ts = pd.DatetimeIndex(ts)
    month_starts_idx = []
    ari_list = []
    prev_grp = None
    for i, ms in enumerate(month_starts):
        t0 = ts.get_loc(ms) if ms in ts else None
        if t0 is None or t0 < LOOKBACK:
            continue
        month_starts_idx.append(t0)
        t1 = ts.get_loc(month_starts[i + 1]) if i + 1 < len(month_starts) else T
        grp = cluster_one_month(ret[:t0], mkt[:t0], el[:t0], univ[:, t0], k)
        sectors[:, t0:t1] = grp[:, None]
        if prev_grp is not None:
            both = (grp >= 0) & (prev_grp >= 0)
            if both.sum() > 1:
                ari_list.append((ms.strftime("%Y-%m"), adjusted_rand_score(grp[both], prev_grp[both])))
        log("  %s: %d groups, sizes %s, ARI vs prev %s" % (
            ms.strftime("%Y-%m"), len(set(grp[grp >= 0])),
            dict(zip(*np.unique(grp[grp >= 0], return_counts=True))),
            "%.3f" % ari_list[-1][1] if ari_list and ari_list[-1][0] == ms.strftime("%Y-%m") else "—"))
        prev_grp = grp
    return sectors, ari_list


def grp_demean(x, g, eligible):
    """(T, A) x and (A, T) group assignments -> (T, A) group-mean centered x."""
    x = x.T
    out = np.full_like(x, np.nan)
    for t in range(x.shape[0]):
        groups = g[:, t]
        el = eligible[:, t]
        for gid in np.unique(groups[groups >= 0]):
            m = (groups == gid) & el
            if m.sum() == 0:
                continue
            gx = x[t, m]
            mn = np.nanmean(gx)
            out[t, m] = gx - mn
    return out.T


def grp_rank(x, g, eligible):
    """(T, A) x and (T, A) group assignments -> (T, A) within-group rank (ties average)."""
    T, A = x.shape
    out = np.full((T, A), np.nan)
    for t in range(T):
        groups = g[t]
        el = eligible[t]
        for gid in np.unique(groups[groups >= 0]):
            m = (groups == gid) & el
            if m.sum() == 0:
                continue
            gx = x[t, m]
            fin = np.isfinite(gx)
            if fin.sum() == 0:
                continue
            ranks = pd.Series(gx).rank(method="average").to_numpy()
            out[t, m] = ranks
    return out


def grp_mean(x, g, eligible):
    """(T, A) x and (A, T) group assignments -> (T, A) group mean (same value for all members)."""
    x = x.T
    out = np.full_like(x, np.nan)
    for t in range(x.shape[0]):
        groups = g[:, t]
        el = eligible[:, t]
        for gid in np.unique(groups[groups >= 0]):
            m = (groups == gid) & el
            if m.sum() == 0:
                continue
            gx = x[t, m]
            mn = np.nanmean(gx)
            out[t, m] = mn
    return out.T


def selftest():
    """Hand-made 5 coins x 3 hours, 2 groups: check grp_rank against manual calculation."""
    A, T = 5, 3
    x = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0], [np.nan, 6.0, 7.0], [8.0, np.nan, 9.0], [10.0, 11.0, 12.0]])
    g = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int8).T
    el = np.isfinite(x.T)
    # hour 0: g0 = {0:1, 1:3}, g1 = {2:nan, 3:8, 4:10} -> ranks g0 {0:1, 1:2}, g1 {3:1, 4:2}
    # hour 1: g0 = {0:2, 3:nan}, g1 = {1:4, 2:6, 4:11} -> g0 {0:1}, g1 {1:1, 2:2, 4:3}
    # hour 2: g0 = {2:7, 3:9}, g1 = {0:nan, 1:5, 4:12} -> g0 {2:1, 3:2}, g1 {1:1, 4:2}
    expected = np.array([[1.0, 1.0, np.nan], [2.0, 1.0, 1.0], [np.nan, 2.0, 1.0], [1.0, np.nan, 2.0], [2.0, 3.0, 2.0]])
    out = grp_rank(x.T, g, el)
    assert np.allclose(out, expected.T, equal_nan=True), (out.T, expected)
    # vectorised β-residual and pairwise ρ against a per-coin / per-pair loop, with holes
    rng = np.random.default_rng(5)
    W, A2 = LOOKBACK, 7
    mk = rng.normal(0, 0.01, W)
    r = mk[:, None] * rng.uniform(0.5, 1.5, A2) + rng.normal(0, 0.01, (W, A2))
    r[rng.random((W, A2)) < 0.2] = np.nan
    r[:500, 6] = np.nan                                   # coin 6 under MIN_OBS -> all NaN
    e = rng.random((W, A2)) > 0.1
    res = _beta_hedge_residuals(r, mk, e)
    for a in range(A2):
        ok = e[:, a] & np.isfinite(r[:, a])
        if ok.sum() < MIN_OBS:
            assert not np.isfinite(res[:, a]).any()
            continue
        y, x = r[ok, a], mk[ok]
        b = ((x - x.mean()) * (y - y.mean())).mean() / (x.var() + 1e-12)
        assert np.allclose(res[ok, a], y - b * x) and not np.isfinite(res[~ok, a]).any()
    c = _corr_matrix(res)
    for i in range(A2):
        for j in range(A2):
            ok = np.isfinite(res[:, i]) & np.isfinite(res[:, j])
            if ok.sum() < 100:
                assert not np.isfinite(c[i, j])
            else:
                assert abs(c[i, j] - np.corrcoef(res[ok, i], res[ok, j])[0, 1]) < 1e-9, (i, j)
    log("SEL2_SECTOR selftest PASS: 3-hour group rank matches hand calculation; "
        "vectorised β-residual and pairwise ρ match the loop")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT_ROOT, "SEL2_SECTOR.npz"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    for _p in (r"D:\ChengboRemote\runtime\m1_stage1", r"D:\riskpg_stateful_v5s_governor_nextgen_beem\src",
               r"D:\riskpg_stateful_v5s_governor_nextgen_beem\runtime"):
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import v4_world as X
    t0 = time.time()
    w = X.load(X.GRID)
    g, univ = w["g"], w["univ"]
    sectors, ari = build_sectors(g, univ, w["ts"], a.k)
    np.savez_compressed(a.out, sectors=sectors, k=a.k, train_months=TRAIN_MONTHS, lookback=LOOKBACK,
                        symbols=w["syms"], hours=np.asarray(w["ts"]).astype("datetime64[ns]"))
    rep = dict(out=a.out, shape=list(sectors.shape), k=a.k, lookback=LOOKBACK, train_months=TRAIN_MONTHS,
               ari_adjacent_months=ari, seconds=round(time.time() - t0, 1), dev_read=False, oos_read=False)
    json.dump(rep, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    log("SEL2_SECTOR %s k=%d | %d months | %.0fs" % (sectors.shape, a.k, len(TRAIN_MONTHS), rep["seconds"]))


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
