import numpy as np, pandas as pd, glob, os
R = r"D:\riskpg_stateful_v5s_governor_nextgen_beem"
for p in glob.glob(r"D:\ChengboRemote\runtime\m1_stage1\m1_build_shard*.py") + glob.glob(R + r"\runtime\m1_build_shard*.py"):
    src = open(p, encoding="utf-8").read(); i = src.find("TERMINALS")
    print(p, os.path.getsize(p)); 
    for ln in src.splitlines():
        if "TERMINALS" in ln or "feats" in ln and "=" in ln: print("  ", ln.strip()[:150])
z = np.load(R + r"\runtime\xs_v4_panel_v1\V4_TRAIN_GRID.npz", allow_pickle=False)
hrs = pd.DatetimeIndex(z["hours"].astype("datetime64[ns]")); syms = [str(s) for s in z["symbols"]]
c = z["trade_close"]; b = syms.index("BTCUSDT"); print("grid", c.shape, hrs[0], hrs[-1], z.files)
m = np.load(R + r"\runtime\m1_staging\m1_shards\m1_shard_2024-03.npz", allow_pickle=False)
ms = [str(s) for s in m["symbols"]]; mb = ms.index("BTCUSDT"); mc = m["close"][mb]
t0 = pd.Timestamp("2024-03-01")
for h in (5, 6, 100):
    T = hrs.get_loc(t0 + pd.Timedelta(hours=h))
    print("row", hrs[T], "grid close %.2f" % c[b, T], "min[h*60+59] %.2f" % mc[h*60+59], "min[h*60-1] %.2f" % mc[h*60-1], "min[h*60] %.2f" % mc[h*60])
print("feat0 vs close", np.nanmax(np.abs(m["feats"][mb, 0] - mc)))
print("missing syms in grid", [s for s in ms if s not in syms])
