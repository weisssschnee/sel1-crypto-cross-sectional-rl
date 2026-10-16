import json
a = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1"
c = json.load(open(a + r"\CR1_PC_P8\HOF.json", encoding="utf-8"))
e = c["entries"]
print(len(e), sorted(e[0].keys()))
b = json.load(open(a + r"\B5_DBG_REAL\HOF_RANKED.json", encoding="utf-8"))
print({k: len(v) for k, v in b["families"].items()}, sorted(b["families"]["T2"][0].keys()))
print([(x.get("h"), x.get("basin")) for x in b["families"]["T2"][:5]])
