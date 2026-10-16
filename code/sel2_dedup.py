"""SEL2 step 1b: drop minute columns that restate a V4 hourly terminal (peer ruling 10-16 13:20).

For each SEL2_MINBAR column (raw, and log for positive columns; plus buy_notional/notional) the hourly Pearson
correlation against every V4 terminal the VM sees (M.build_world feats) and every raw V4 panel column, on the
coin-hours the minute shards cover. max |rho| > 0.99 -> dropped (it is already in V4). Also reports the minute
coverage as a share of V4 coin-hours (all, and inside the panel TOP100). TRAIN grid only; no DEV, no OOS.

    python -I sel2_dedup.py [--feats SEL2_MINBAR_FEATURES.npz]
-> SEL2_MINBAR_KEEP.json next to the npz, one line.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _p in (r"D:\ChengboRemote\runtime\m1_stage1", r"D:\riskpg_stateful_v5s_governor_nextgen_beem\src",
           r"D:\riskpg_stateful_v5s_governor_nextgen_beem\runtime"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import m2h_common as M
import v4_world as X

THR = 0.99
OUT_ROOT = os.path.join(X.ROOT, "artifacts", "m2h_v1", "XS_SEL1")


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    x, y = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default=os.path.join(OUT_ROOT, "SEL2_MINBAR_FEATURES.npz"))
    a = ap.parse_args()
    z = np.load(a.feats, allow_pickle=False)
    cols = [str(c) for c in z["feature_names"]]
    cov = z["minbar_covered"]
    w = X.load(X.GRID)
    if [str(s) for s in z["symbols"]] != w["syms"] or len(z["hours"]) != len(w["ts"]):
        raise SystemExit("SEL2 npz is not on the V4 grid")
    cm = cov.ravel()
    bw = M.build_world(w["g"])
    v4 = {"term:" + t: bw["feats"][:, k].ravel()[cm] for k, t in enumerate(M.TERMINALS)}
    v4.update({"panel:" + c: w["g"][c].ravel()[cm] for c in M.PANEL_COLS})
    qv = w["g"]["trade_quote_volume"]
    with np.errstate(invalid="ignore", divide="ignore"):
        v4["derived:taker_buy_share"] = np.where(qv > 0, w["g"]["taker_buy_quote_volume"] / qv, np.nan).ravel()[cm]
    del bw
    f = z["feats"]
    cand = {}
    for k, c in enumerate(cols):
        v = f[:, k].ravel()[cm].astype(np.float64)
        cand[c] = v
        if np.nanmin(v) > 0:
            cand["log " + c] = np.log(v)
    n = f[:, cols.index("notional")].ravel()[cm].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        cand["buy_notional/notional"] = f[:, cols.index("buy_notional")].ravel()[cm] / np.where(n > 0, n, np.nan)
    rep = {}
    for c, v in cand.items():
        rs = {k: corr(v, u) for k, u in v4.items()}
        best = max(rs, key=lambda k: abs(rs[k]) if np.isfinite(rs[k]) else -1)
        rep[c] = dict(max_abs_rho=abs(rs[best]), v4=best, rho=rs[best])
    base = lambda c: c[4:] if c.startswith("log ") else c
    dropped = sorted({c for c in cols if any(rep[k]["max_abs_rho"] > THR for k in rep if base(k) == c)})
    keep = [c for c in cols if c not in dropped]
    live = np.isfinite(w["g"]["trade_close"])
    univ = w["univ"] & live
    out = dict(threshold=THR, keep=keep, dropped=dropped, per_column=rep,
               coverage_vs_v4_coin_hours=float(cov.sum() / live.sum()),
               coverage_vs_v4_top100_coin_hours=float((cov & univ).sum() / univ.sum()),
               covered_coin_hours=int(cov.sum()), dev_read=False, oos_read=False)
    p = os.path.join(os.path.dirname(a.feats), "SEL2_MINBAR_KEEP.json")
    json.dump(out, open(p, "w"), indent=1)
    print("SEL2 keep %d %s | drop %s | cov all %.4f top100 %.4f | %s"
          % (len(keep), keep, ["%s(%s %.4f)" % (c, max((rep[k] for k in rep if base(k) == c), key=lambda r: r["max_abs_rho"])["v4"],
                                               max(rep[k]["max_abs_rho"] for k in rep if base(k) == c)) for c in dropped],
             out["coverage_vs_v4_coin_hours"], out["coverage_vs_v4_top100_coin_hours"], p), flush=True)


if __name__ == "__main__":
    main()
