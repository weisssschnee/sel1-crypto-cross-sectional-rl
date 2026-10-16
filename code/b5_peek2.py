import json
import numpy as np
A = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1"
z = np.load(A + r"\BASIN5_20261012\smoke\B5_REWARD_SCORES.npz", allow_pickle=True)
for k, run in (("p8", "RGP_P8_V2"), ("null", "RGP_NULL_V2")):
    first = {}
    for l in open(A + "\\" + run + r"\RGP_LOG.jsonl", encoding="utf-8"):
        if l.endswith("\n"):
            r = json.loads(l)
            if r["fam"] == "T1" and r["expr"] not in first:
                first[r["expr"]] = r
    g = z[k + "_gross"].mean(1)
    lg = [first[str(e)]["gross"] for e in z[k + "_expr"]]
    print(k, "re", np.round(g, 4).tolist(), "log", np.round(lg, 4).tolist(), "ntok", z[k + "_n_tokens"].tolist())
