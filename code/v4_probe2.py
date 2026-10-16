"""Symbol overlaps of the V4 inputs + DEV V1 npz layout (keys / shapes / hour span). No OOS."""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

R = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\runtime"
P = {"BF3": R + r"\train2021_2023_panel_v1\BACKFILL2020NOV_2023DEC_HOURLY11_LONG_V3.parquet",
     "TR": R + r"\train2024_2025_panel_v1\TRAIN2024JAN_2025DEC_HOURLY54_TARGET_LONG_V1.parquet",
     "SM2": R + r"\xs_seam2024_2026_panel_v1\SEAM2024JAN_2026JUN_HOURLY11_LONG_V2.parquet"}
DEV = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\DEV2026_HOURLY_PANEL_V1.npz"
S = {k: set(pq.read_table(p, columns=["symbol"]).to_pandas().symbol.unique()) for k, p in P.items()}
z = np.load(DEV, allow_pickle=False)
ds = set(str(x) for x in z["symbols"])
h = pd.DatetimeIndex(z["hours"].astype("datetime64[ns]"))
print("keys", sorted(z.files))
print("dev", len(ds), str(h[0]), str(h[-1]), len(h), z["trade_close"].shape)
print("BF3&TR", len(S["BF3"] & S["TR"]), "BF3&SM2", len(S["BF3"] & S["SM2"]), "TR&SM2", len(S["TR"] & S["SM2"]),
      "union", len(S["BF3"] | S["TR"] | S["SM2"]))
print("DEV&TR", len(ds & S["TR"]), "DEV&SM2", len(ds & S["SM2"]), "DEV-TR", sorted(ds - S["TR"])[:8],
      "TR-DEV", len(S["TR"] - ds))
t = pq.read_table(P["SM2"], columns=["symbol", "timestamp"]).to_pandas()
t26 = t[t.timestamp >= "2026-01-01"]
print("SM2 2026 syms", t26.symbol.nunique(), "SM2 alive 2025-11..12", t[(t.timestamp >= "2025-11-01") &
      (t.timestamp < "2026-01-01")].symbol.nunique())
print("TR cols", [c for c in pq.ParquetFile(P["TR"]).schema.to_arrow_schema().names][:20])
