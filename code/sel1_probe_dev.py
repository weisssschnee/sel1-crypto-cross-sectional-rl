"""Probe (read only, no DEV values): V4 DEV / TRAIN grid hour ranges and symbol lists."""
import numpy as np
import pandas as pd

P = "D:\\riskpg_stateful_v5s_governor_nextgen_beem\\runtime\\xs_v4_panel_v1\\"
for n in ("V4_DEV2026_HOURLY_PANEL.npz", "V4_TRAIN_GRID.npz"):
    z = np.load(P + n, allow_pickle=False)
    t = pd.DatetimeIndex(z["hours"].astype("datetime64[ns]"))
    print(n, t.min(), t.max(), len(t), len(z["symbols"]), sorted(z.files)[:20])
a = list(np.load(P + "V4_DEV2026_HOURLY_PANEL.npz")["symbols"])
b = list(np.load(P + "V4_TRAIN_GRID.npz")["symbols"])
print("same_syms", a == b, len(set(a) & set(b)))
