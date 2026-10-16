"""XS_SEL1 OKX settlement grid (order: "有候选过门 ... 用本机的 OKX 原生 1h K 线和资金费，重新结算 2024–25 和 DEV").
Copy of b5_okx_dev_grid.build widened to TRAIN + DEV; b5_okx_dev_grid is not modified.

Partitions source_date_utc8 20240101..20260701 only; every row at/after 2026-07-01T00:00Z is dropped right after each
file is read. Rows are bar OPEN hours (UTC naive) 2023-12-31T16:00 .. 2026-06-30T23:00 (the first 8 rows are the
20240101 partition's lead-in). close / funding_rate conventions are b5_okx_dev_grid's (funding at floor_hour(F),
same-hour events summed). Symbols = CONC_POOL (Binance name -> OKX swap); not in the pool = not OKX tradable.
    py -I sel1_okx_grid.py
-> OKX_SEL1_1H_V1.npz + .json in b5_scratch.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CARRIER = r"G:\AlphaFactory_CryptoData\derived\okx_native_crypto_long_v1_20260917"
POOL = r"G:\Chengbo\runtime\basin3_h1\b5_scratch\CONC_POOL.json"
OUT = r"G:\Chengbo\runtime\basin3_h1\b5_scratch\OKX_SEL1_1H_V1.npz"
RECEIPT = r"G:\Chengbo\runtime\basin3_h1\b5_scratch\OKX_SEL1_1H_V1.json"
P_LO, P_HI = 20240101, 20260701
OOS_FIRST = pd.Timestamp("2026-07-01T00:00:00", tz="UTC")
H0 = pd.Timestamp("2023-12-31T16:00:00", tz="UTC")
H1 = pd.Timestamp("2026-06-30T23:00:00", tz="UTC")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def partitions(kind):
    out = []
    for d in sorted(os.listdir(os.path.join(CARRIER, kind))):
        k = int(d.split("=")[1])
        if P_LO <= k <= P_HI:
            out.append((k, d))
    assert out and max(k for k, _ in out) <= P_HI
    return out


def load(kind, cols, tcol, keep):
    """Each partition file on its own; truncate at 2026-07-01 at once, then pool symbols only."""
    frames, n_in, n_drop, opened = [], 0, 0, []
    for k, d in partitions(kind):
        for f in sorted(glob.glob(os.path.join(CARRIER, kind, d, "*.parquet"))):
            t = pq.ParquetFile(f).read(columns=cols).to_pandas()
            n_in += len(t)
            oos = t[tcol] >= OOS_FIRST
            n_drop += int(oos.sum())
            t = t[~oos]
            frames.append(t[t.venue_symbol.isin(keep)])
            opened.append(k)
    return pd.concat(frames, ignore_index=True), n_in, n_drop, opened


def main():
    pool = json.load(open(POOL, encoding="utf-8"))["pool"]
    syms = sorted(pool)
    okx = [pool[s]["okx"] for s in syms]
    assert len(set(okx)) == len(okx)
    oi = pd.Index(okx)
    hours = pd.date_range(H0, H1, freq="h")
    A, T = len(syms), len(hours)

    bars, n1, d1, op1 = load("1h", ["timestamp", "venue_symbol", "close", "observed_bar_present"], "timestamp", oi)
    dup = bars.duplicated(["venue_symbol", "timestamp"], keep=False)
    if dup.any():
        assert (bars[dup].groupby(["venue_symbol", "timestamp"]).close.nunique() == 1).all(), "conflicting bars"
    bars = bars.drop_duplicates(["venue_symbol", "timestamp"])
    assert not (bars.timestamp != bars.timestamp.dt.floor("h")).any(), "bars off the hourly grid"
    bars = bars[bars.observed_bar_present & np.isfinite(bars.close)]
    ti = ((bars.timestamp - H0) / pd.Timedelta(hours=1)).to_numpy().astype(int)
    ok = (ti >= 0) & (ti < T)
    close = np.full((A, T), np.nan)
    close[oi.get_indexer(bars.venue_symbol)[ok], ti[ok]] = bars.close.to_numpy()[ok]

    fe, n2, d2, op2 = load("funding_events", ["venue_symbol", "funding_time", "funding_rate"], "funding_time", oi)
    fe = fe[np.isfinite(fe.funding_rate)].drop_duplicates(["venue_symbol", "funding_time"])
    fe = fe.assign(fh=fe.funding_time.dt.floor("h"))
    tj = ((fe.fh - H0) / pd.Timedelta(hours=1)).to_numpy().astype(int)
    okf = (tj >= 0) & (tj < T)
    fe, tj = fe[okf], tj[okf]
    acc = np.zeros((A, T)); cnt = np.zeros((A, T), dtype=np.int32)
    ai = oi.get_indexer(fe.venue_symbol)
    np.add.at(acc, (ai, tj), fe.funding_rate.to_numpy())
    np.add.at(cnt, (ai, tj), 1)
    funding = np.where(cnt > 0, acc, np.nan)

    hours_np = hours.tz_convert(None).to_numpy().astype("datetime64[ns]")
    np.savez_compressed(OUT, symbols=np.array(syms, dtype="U20"), okx=np.array(okx, dtype="U24"),
                        hours=hours_np, close=close, funding_rate=funding)
    yr = hours.year.to_numpy()
    cov = {str(y): round(float(np.isfinite(close[:, yr == y]).mean()), 4) for y in (2024, 2025, 2026)}
    rec = dict(schema="SEL1_OKX_1H_V1", builder=os.path.abspath(__file__), builder_sha256=sha256(__file__),
               carrier=CARRIER, pool=POOL, pool_sha256=sha256(POOL), npz=OUT, npz_sha256=sha256(OUT), shape=[A, T],
               hours=[str(hours_np[0]), str(hours_np[-1])], coverage_by_year=cov,
               conventions="b5_okx_dev_grid (bar open hours; funding at floor_hour(F), same-hour summed)",
               red_line=dict(oos_read=False, oos_first=str(OOS_FIRST), partitions_1h=[min(op1), max(op1), len(op1)],
                             partitions_funding=[min(op2), max(op2), len(op2)], rows_loaded_1h=n1,
                             rows_dropped_ge_2026_07_01_1h=d1, rows_loaded_funding=n2,
                             rows_dropped_ge_2026_07_01_funding=d2))
    json.dump(rec, open(RECEIPT, "w", encoding="utf-8"), indent=1)
    print("npz", rec["npz_sha256"][:16], "shape", A, T, "cov", cov, "dropped", d1, d2)


if __name__ == "__main__":
    main()
