"""Debug (read-only): one single-coin event through sel1_common vs the ENR1b two-column settlement."""
import os, sys, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, torch, pandas as pd
import sel1_common as S, cr1_common as Q, m1_ruler as R
warnings.simplefilter("ignore", RuntimeWarning)
vm = Q.make_vm()
ctx = S.build("real", torch.device("cuda"), vm, log=lambda m: None)
sh, bi = ctx["shard"], ctx["bi"]
a = ctx["syms"].index("XTZUSDT"); t = int(ctx["ts"].get_loc(pd.Timestamp("2021-10-08 11:00"))); h = 72; d = 1.0
m = [m for m in ctx["months"] if m["m0"] <= t < m["m1"]][0]
des = np.zeros((m["m1"] - m["m0"], len(ctx["syms"]))); des[t + 1 - m["m0"]:t + 1 + h - m["m0"], a] = d
(held, cols), = S.held_months([des], h)
r = S.settle_window(held, cols, ctx, m["m0"], m["m1"], keep=True)
H = r["held"]; nz = np.flatnonzero(H[:, 0])
print("rows", nz.min() + m["m0"] - t, nz.max() + m["m0"] - t, len(nz), "coin", set(H[nz, 0]), "hedge", set(np.round(H[nz, 1], 6)), "beta_t", round(float(ctx["beta"][a, t]), 6))
rows = np.arange(t + 1, t + h + 2)
specs = {c: R.cost_specs(R.NAV_USD)[c] for c in ("flat6", "cost0")}
hv = np.zeros((h + 1, 2)); hv[:h, 0], hv[:h, 1] = d, -d * ctx["beta"][a, t]
res, p = R.settle_multi(hv, sh["fwd_wx"][[a, bi]][:, rows].T, sh["fi"][[a, bi]][:, rows].T, sh["fr"][[a, bi]][:, rows].T, specs)
res2, p2 = R.settle_multi(H, r["fw"], r["fi"], r["fr"], specs)
print("ref", res["cost0"]["log_nav"], p["turnover"], "win", r["cost0"], res2["cost0"]["log_nav"], p2["turnover"])
sub = H[t + 1 - m["m0"]:t + h + 2 - m["m0"]]
print("held equal on event rows", bool(np.array_equal(sub, hv)), float(np.abs(sub - hv).max()))
print("fw equal", bool(np.array_equal(r["fw"][t + 1 - m["m0"]:t + h + 2 - m["m0"]], sh["fwd_wx"][[a, bi]][:, rows].T, equal_nan=True)))
print("held outside nonzero", int((np.abs(np.delete(H, np.arange(t + 1 - m["m0"], t + h + 2 - m["m0"]), axis=0)) > 0).sum()))
