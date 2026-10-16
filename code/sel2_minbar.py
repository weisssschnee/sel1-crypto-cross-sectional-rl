"""SEL2 step 1: the m1 minute shards' own columns, aggregated to the V4 hourly grid (XS_SEL1 10-16, peer ruling).

No new features. Columns are m1_common.SOURCE_COLUMNS (the 16 aggtrades terminals the shards were built with;
funding_level is not in the shards), read from m1_common, not written by hand. Hour rule, mechanical only:
  flow   sum                          notional, quantity, agg_trade_count, buy/sell_notional, signed_aggressor_notional
  ratio  notional-weighted mean       price_std, price_range_bps, close_to_open_bps, volume_imbalance,
                                      buy_sell_notional_ratio, large_notional_ratio_100k_plus, avg_agg_trade_notional
  price  the hour's last minute bar   close_price, vwap
  max    max                          max_trade_notional
Causality: V4 row h is labelled by the hour's open (row h's trade_close is the close at h+1h, first held at h+1).
Minute bars are labelled by their open, so row h uses minutes h:00..h:59 only -- every bar that has closed by the
hour's close and none after it. The selftest proves it on a hand-made 3-hour case; main also checks the label
alignment on real data (minute close of h:59 vs V4 trade_close[h], offsets -1/0/+1).
TRAIN months only (2024-01..2025-12); DEV / OOS shards are not opened.

    python -I sel2_minbar.py [--out OUT.npz] [--selftest]
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
# -I drops PYTHONPATH: m1_common lives in the stage-1 dir and pulls okx_alpha_discovery from ROOT/src.
for _p in (r"D:\ChengboRemote\runtime\m1_stage1", r"D:\riskpg_stateful_v5s_governor_nextgen_beem\src",
           r"D:\riskpg_stateful_v5s_governor_nextgen_beem\runtime"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import m1_common as C
import v4_world as X

OUT_ROOT = os.path.join(X.ROOT, "artifacts", "m2h_v1", "XS_SEL1")
SHARD_DIR = os.path.join(X.ROOT, "runtime", "m1_staging", "m1_shards")
SHARD_MONTHS = ["2024-%02d" % m for m in range(1, 13)] + ["2025-%02d" % m for m in range(1, 13)]
COLS = tuple(C.SOURCE_COLUMNS)
FLOW = ("notional", "quantity", "agg_trade_count", "buy_notional", "sell_notional", "signed_aggressor_notional")
PRICE = ("close_price", "vwap")
MAXC = ("max_trade_notional",)
RULE = {c: ("sum" if c in FLOW else "last" if c in PRICE else "max" if c in MAXC else "wmean") for c in COLS}
# Columns that restate V4's taker_buy_share (taker buy quote / quote); only their hourly correlation is reported.
TAKER_DUP = ("buy_notional", "sell_notional", "signed_aggressor_notional", "volume_imbalance", "buy_sell_notional_ratio")


def log(m):
    print(m, flush=True)


def hourly(feats):
    """(A, F, n_min) minute bars, F = COLS -> (A, F, n_min // 60) hours by RULE, + (A, H) covered.

    Hour k is built from minutes [60k, 60k + 60) and nothing else (pure reshape, no rolling window)."""
    A, F, n = feats.shape
    if F != len(COLS) or n % 60:
        raise ValueError("feats %s vs %d cols / whole hours" % (feats.shape, len(COLS)))
    H = n // 60
    x = feats.reshape(A, F, H, 60).astype(np.float64)
    fin = np.isfinite(x)
    w = x[:, COLS.index("notional")]                           # (A, H, 60) weights
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    out = np.full((A, F, H), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for k, c in enumerate(COLS):
            v, f = x[:, k], fin[:, k]
            any_f = f.any(axis=2)
            r = RULE[c]
            if r == "sum":
                o = np.where(f, v, 0.0).sum(axis=2)
            elif r == "max":
                o = np.where(f, v, -np.inf).max(axis=2)
            elif r == "last":
                j = 59 - np.argmax(f[:, :, ::-1], axis=2)       # last finite minute of the hour
                o = np.take_along_axis(v, j[:, :, None], axis=2)[:, :, 0]
            else:
                ww = np.where(f, w, 0.0)
                sw = ww.sum(axis=2)
                o = np.where(sw > 0, (ww * np.where(f, v, 0.0)).sum(axis=2) / np.where(sw > 0, sw, 1.0), np.nan)
            out[:, k] = np.where(any_f, o, np.nan)
    covered = fin[:, COLS.index("close_price")].any(axis=2)
    return out.astype(np.float32), covered


def selftest():
    """Hand-made 3 hours, 1 coin: each rule against a hand calculation; hour 0 bit-identical when every minute
    from 60 on is changed; minute 59 does reach hour 0 (it closes at the hour's close), minute 60 does not."""
    rng = np.random.default_rng(3)
    x = rng.uniform(1.0, 9.0, (1, len(COLS), 180)).astype(np.float32)
    x[0, :, 7] = np.nan                                         # a missing minute inside hour 0
    out, cov = hourly(x)
    h0 = x[0, :, :60].astype(np.float64)
    f = np.isfinite(h0)
    w = np.where(f[COLS.index("notional")], h0[COLS.index("notional")], 0.0)
    for k, c in enumerate(COLS):
        v = h0[k][f[k]]
        hand = dict(sum=v.sum(), max=v.max(), last=h0[k][59],
                    wmean=(w[f[k]] * v).sum() / w[f[k]].sum())[RULE[c]]
        assert abs(float(out[0, k, 0]) - np.float32(hand)) <= 1e-6 * max(1.0, abs(hand)), (c, out[0, k, 0], hand)
    y = x.copy()
    y[0, :, 60:] = rng.uniform(100.0, 900.0, (len(COLS), 120))
    o2, _ = hourly(y)
    assert np.array_equal(o2[:, :, 0], out[:, :, 0]), "hour 0 moved when minutes >= 60 changed"
    assert not np.array_equal(o2[:, :, 1], out[:, :, 1])
    z = x.copy()
    z[0, :, 59] += 50.0
    o3, _ = hourly(z)
    assert not np.array_equal(o3[:, :, 0], out[:, :, 0]), "minute 59 should reach hour 0"
    assert np.array_equal(o3[:, :, 1:], out[:, :, 1:])
    assert cov.all()
    print("SEL2_MINBAR selftest PASS: %d cols x 4 rules vs hand, hour 0 sealed from minutes >= 60, rules %s"
          % (len(COLS), {r: sum(1 for c in COLS if RULE[c] == r) for r in ("sum", "wmean", "last", "max")}), flush=True)


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 2 else float("nan"), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(OUT_ROOT, "SEL2_MINBAR_FEATURES.npz"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest()
    if a.selftest:
        return
    t0 = time.time()
    w = X.load(X.GRID)
    syms, ts, g = w["syms"], w["ts"], w["g"]
    si = {s: i for i, s in enumerate(syms)}
    A, T = len(syms), len(ts)
    feats = np.full((A, len(COLS), T), np.nan, dtype=np.float32)
    covered = np.zeros((A, T), dtype=bool)
    align = {-1: [], 0: [], 1: []}
    ci = COLS.index("close_price")
    for mon in SHARD_MONTHS:
        z = np.load(os.path.join(SHARD_DIR, "m1_shard_%s.npz" % mon), allow_pickle=False)
        if z["feats"].shape[1] != len(COLS):
            raise SystemExit("%s: shard has %d cols, m1_common.SOURCE_COLUMNS %d" % (mon, z["feats"].shape[1], len(COLS)))
        fm = z["feats"]
        cl = z["close"]
        same = np.isfinite(cl)
        if not np.array_equal(fm[:, ci][same], cl[same].astype(np.float32)):
            raise SystemExit("%s: shard col %d is not close_price; column order differs from SOURCE_COLUMNS" % (mon, ci))
        hh, cv = hourly(fm)
        h0 = int(ts.get_loc(pd.Timestamp(mon + "-01")))
        H = min(hh.shape[2], T - h0)
        for i, s in enumerate(str(v) for v in z["symbols"]):
            j = si.get(s)
            if j is None:
                continue
            feats[j, :, h0:h0 + H] = hh[i, :, :H]
            covered[j, h0:h0 + H] = cv[i, :H]
            vc = g["trade_close"][j]
            for off in align:                                   # minute close at h:59 vs V4 trade_close[h + off]
                lo, hi = max(0, -off), min(H, T - h0 - off)
                p, q = hh[i, ci, lo:hi].astype(np.float64), vc[h0 + lo + off:h0 + hi + off]
                m = np.isfinite(p) & np.isfinite(q) & (q > 0)
                align[off].append(np.abs(p[m] / q[m] - 1.0))
        log("  %s: %d syms, %d h, %.0fs" % (mon, len(z["symbols"]), H, time.time() - t0))
    med = {off: float(np.median(np.concatenate(v))) if v else float("nan") for off, v in align.items()}
    ok_align = med[0] < 1e-6 and med[0] < min(med[-1], med[1])
    qv, tb = g["trade_quote_volume"], g["taker_buy_quote_volume"]
    with np.errstate(invalid="ignore", divide="ignore"):
        tbs = np.where(qv > 0, tb / qv, np.nan)
        n = feats[:, COLS.index("notional")].astype(np.float64)
        rows = {c: feats[:, COLS.index(c)].astype(np.float64) for c in TAKER_DUP}
        rows["buy_notional/notional"] = rows["buy_notional"] / np.where(n > 0, n, np.nan)
    cm = covered.ravel()
    dup = {c: _corr(v.ravel()[cm], tbs.ravel()[cm]) for c, v in rows.items()}
    qv_c = _corr(np.log(np.where(n > 0, n, np.nan)).ravel()[cm], np.log(np.where(qv > 0, qv, np.nan)).ravel()[cm])
    np.savez_compressed(a.out, feats=feats, minbar_covered=covered, feature_names=np.array(COLS),
                        rules=np.array([RULE[c] for c in COLS]), hours=ts.values.astype("datetime64[ns]"),
                        symbols=np.array(syms))
    rep = dict(out=a.out, shape=list(feats.shape), cols=list(COLS), rules=RULE, months=SHARD_MONTHS,
               covered_coin_hours=int(covered.sum()), coins_covered=int(covered.any(axis=1).sum()),
               align_median_abs_rel={str(k): v for k, v in med.items()}, align_ok=bool(ok_align),
               corr_vs_v4_taker_buy_share={c: dict(r=v[0], n=v[1]) for c, v in dup.items()},
               corr_log_notional_vs_log_quote_volume=dict(r=qv_c[0], n=qv_c[1]),
               seconds=round(time.time() - t0, 1), dev_read=False, oos_read=False)
    json.dump(rep, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    log("SEL2_MINBAR %s cov %d coin-h on %d coins | align med |rel| -1:%.2e 0:%.2e +1:%.2e ok %s | "
        "corr taker_buy_share: %s | log notional~qv %.3f | %.0fs"
        % (feats.shape, rep["covered_coin_hours"], rep["coins_covered"], med[-1], med[0], med[1], ok_align,
           ", ".join("%s %.3f" % (c, v[0]) for c, v in dup.items()), qv_c[0], rep["seconds"]))
    if not ok_align:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
