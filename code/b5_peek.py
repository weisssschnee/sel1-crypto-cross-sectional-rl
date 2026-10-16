import json, sys
import numpy as np
d = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\BASIN5_20261012\smoke"
j = json.load(open(d + r"\REWARD_PICK.json", encoding="utf-8"))
print("pick", j["pick"], "pools", {k: (v["scored"], v["gross_matches_log"]) for k, v in j["pools"].items()})
print("target", j["target"]["corr_s"], j["target"]["n_tokens"], j["target"]["gross_bpd"])
for x, t in j["table"].items():
    print(x, t["target_rank"], t["of"], t["target_score"], t["null_best"], t["null_beats_target"])
z = np.load(d + r"\B5_REWARD_SCORES.npz", allow_pickle=True)
g = z["p8_gross"].mean(1)
print("p8 gross_mean", np.round(g, 3).tolist())
