"""DEV 2026-01..06 hourly panel (BASIN3 handoff section V). CPU only, no GPU, no scoring.

Builds one hourly grid by month, with the TRAIN 2025-12 tail prepended so every symbol has at
least 672 h of lookback from 2026-01-01. Adjacent month shards overlap (each starts around the
15th of the previous month); the overlap is checked cell by cell and the mismatched-cell count is
reported. DEV is never scored here -- structure only. OOS (2026-07+) is never opened.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem"
DEV_DIR = os.path.join(ROOT, r"branches\glm_dev2026_cache_v3\cache_v3")
TRAIN = os.path.join(ROOT, r"runtime\train2024_2025_panel_v1\TRAIN2024JAN_2025DEC_HOURLY54_TARGET_LONG_V1.parquet")
OUT = os.path.join(ROOT, "artifacts", "m2h_v1", "DEV2026_HOURLY_PANEL_V1.npz")

MONTHS = ["DEV2026_M01", "DEV2026_M02", "DEV2026_M03", "DEV2026_M04", "DEV2026_M05", "DEV2026_M06"]
# the four price/volume fields, funding, and the T2/T3 fields the handoff asks to reconcile
CHECK = ["trade_close", "trade_high", "trade_low", "trade_quote_volume",
         "funding_rate", "open_interest_value_last", "global_long_short_account_ratio_last",
         "taker_buy_quote_volume", "premium_close_bps", "mark_index_basis_bps", "premium_high"]
OOS_FIRST = pd.Timestamp("2026-07-01")


def dev_path(m):
    return os.path.join(DEV_DIR, f"{m}_FEATURE_ROWS.parquet")


def probe():
    t = pq.ParquetFile(dev_path(MONTHS[0]))
    print("DEV cols:", t.schema.to_arrow_schema().names)
    md = t.metadata
    print("rows:", md.num_rows, "row_groups:", md.num_row_groups)
    tb = pq.read_table(dev_path(MONTHS[0]), columns=["symbol", "timestamp"])
    d = tb.to_pandas()
    print("M01 span:", d.timestamp.min(), "->", d.timestamp.max(), "symbols:", d.symbol.nunique())
    for m in MONTHS:
        dd = pq.read_table(dev_path(m), columns=["timestamp"]).to_pandas()
        print("%s %s -> %s rows=%d" % (m, dd.timestamp.min(), dd.timestamp.max(), len(dd)))


def load_train_tail():
    """TRAIN rows from 2025-06-01: covers the 672 h VM lookback and the 720 h universe at
    2026-01-01, and gives CTREND-lite its 200-day volume SMA (h1 had it warm on TRAIN)."""
    d = pq.read_table(TRAIN, columns=["symbol", "timestamp"] + CHECK).to_pandas()
    d = d[d.timestamp >= pd.Timestamp("2025-06-01")].reset_index(drop=True)
    print("TRAIN tail rows %d span %s -> %s" % (len(d), d.timestamp.min(), d.timestamp.max()))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    if args.probe:
        probe()
        return
    if not args.build:
        raise SystemExit("use --probe or --build")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    frames = [load_train_tail()]
    for m in MONTHS:
        p = dev_path(m)
        d = pq.read_table(p, columns=["symbol", "timestamp"] + CHECK).to_pandas()
        if d.timestamp.max() >= OOS_FIRST:
            raise SystemExit(f"{m} reaches {d.timestamp.max()}, inside the sealed OOS window")
        print("%s rows %d span %s -> %s symbols %d"
              % (m, len(d), d.timestamp.min(), d.timestamp.max(), d.symbol.nunique()))
        frames.append(d)

    syms = sorted(set().union(*[set(f.symbol) for f in frames]))
    ts = pd.date_range(min(f.timestamp.min() for f in frames),
                       max(f.timestamp.max() for f in frames), freq="h")
    A, T = len(syms), len(ts)
    print("grid %d symbols x %d hours (%s -> %s)" % (A, T, ts[0], ts[-1]))

    grids = {c: np.full((A, T), np.nan) for c in CHECK}
    seen = np.zeros((A, T), dtype=np.int8)     # how many shards wrote each cell
    mismatch = {c: 0 for c in CHECK}
    compared = {c: 0 for c in CHECK}
    si_idx = pd.Index(syms)

    for f in frames:
        si = si_idx.get_indexer(f.symbol)
        ti = ((f.timestamp - ts[0]) / pd.Timedelta(hours=1)).to_numpy()
        if not np.all(ti == np.round(ti)):
            raise SystemExit("a shard is off the hourly grid")
        ti = ti.astype(int)
        ok = (si >= 0) & (ti >= 0) & (ti < T)
        if not ok.all():
            print("dropped %d off-grid rows" % int((~ok).sum()))
        si, ti = si[ok], ti[ok]
        wrote = seen[si, ti] > 0            # only cells a previous shard already wrote are comparable
        for c in CHECK:
            v = f[c].to_numpy(dtype=np.float64)[ok]
            g = grids[c]
            prev = g[si, ti]
            both = np.isfinite(prev) & np.isfinite(v)
            # bit-equality on the overlapping cells: NaN==NaN counts as agree via isfinite gate
            diff = wrote & both & (prev != v)
            nan_gap = wrote & (np.isfinite(prev) != np.isfinite(v))
            mismatch[c] += int(diff.sum()) + int(nan_gap.sum())
            compared[c] += int(wrote.sum())
            g[si, ti] = np.where(np.isfinite(v), v, np.where(np.isfinite(prev), prev, v))
        seen[si, ti] += 1

    jan1 = int((pd.Timestamp("2026-01-01") - ts[0]) / pd.Timedelta(hours=1))
    print("OVERLAP compared=%s" % {c: int(compared[c]) for c in CHECK})
    print("OVERLAP mismatch=%s" % {c: int(mismatch[c]) for c in CHECK})
    print("OVERLAP total_mismatch=%d" % sum(mismatch.values()))
    # lookback coverage from 2026-01-01: finite close hours inside [Jan1-672h, Jan1)
    close = grids["trade_close"]
    warm = np.isfinite(close[:, max(0, jan1 - 672):jan1]).sum(1)
    print("lookback at 2026-01-01 (finite hours of last 672): min=%d median=%d n_lt672=%d"
          % (int(warm.min()), int(np.median(warm)), int((warm < 672).sum())))

    fr = grids["funding_rate"]
    for lab, lo_, hi_ in (("TRAIN tail", 0, jan1), ("DEV", jan1, T)):
        print("funding finite frac %s: %.4f" % (lab, float(np.isfinite(fr[:, lo_:hi_]).mean())))
    fwd = np.full_like(close, np.nan)
    contig = np.zeros(close.shape, dtype=bool)
    contig[:, :-1] = np.isfinite(close[:, 1:]) & np.isfinite(close[:, :-1])
    fwd[:, :-1] = np.where(contig[:, :-1], close[:, 1:] / close[:, :-1] - 1.0, np.nan)
    np.savez_compressed(OUT, symbols=np.array(syms), hours=ts.to_numpy().astype("datetime64[h]"),
                        **{c: grids[c] for c in CHECK}, fwd=fwd)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
