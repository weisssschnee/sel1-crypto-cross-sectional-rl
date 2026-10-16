"""Schema / span probe of the three V4 input panels (footer + symbol/timestamp only). No OOS."""
import sys
import pyarrow.parquet as pq
import pandas as pd

R = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\runtime"
P = {"BF3": R + r"\train2021_2023_panel_v1\BACKFILL2020NOV_2023DEC_HOURLY11_LONG_V3.parquet",
     "TR": R + r"\train2024_2025_panel_v1\TRAIN2024JAN_2025DEC_HOURLY54_TARGET_LONG_V1.parquet",
     "SM2": R + r"\xs_seam2024_2026_panel_v1\SEAM2024JAN_2026JUN_HOURLY11_LONG_V2.parquet"}
for k, p in P.items():
    f = pq.ParquetFile(p)
    names = f.schema.to_arrow_schema().names
    t = pq.read_table(p, columns=["symbol", "timestamp"]).to_pandas()
    print(k, f.metadata.num_rows, t.symbol.nunique(), str(t.timestamp.min()), str(t.timestamp.max()),
          str(t.timestamp.dtype))
    print(" cols", names if len(names) < 30 else names[:8] + ["...%d" % len(names)])
    sys.stdout.flush()
