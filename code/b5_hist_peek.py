"""Read-only peek: per family, distinct valid formulas in a run's FORMULAS.jsonl ranked by mean logged
gross_t (the TRAIN reward on the drawn tiles); prints top 5 with logged corr_s, and where corr>0.7 sits.

    python b5_hist_peek.py RUN_NAME
"""
import collections
import json
import os
import sys

ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem"
d = os.path.join(ROOT, "artifacts", "m2h_v1", sys.argv[1])
agg = collections.defaultdict(lambda: dict(gt=[], cs=[], u0=None))
for line in open(os.path.join(d, "FORMULAS.jsonl"), encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    if r.get("invalid_reason") is not None or r.get("gross_t") is None:
        continue
    a = agg[(r["basin"], r["expr"])]
    a["gt"].append(r["gross_t"])
    if r.get("corr_s") is not None:
        a["cs"].append(r["corr_s"])
    a["u0"] = r["update"] if a["u0"] is None else a["u0"]
for fam in ("T1", "T2", "T3"):
    xs = [(sum(v["gt"]) / len(v["gt"]), (sum(v["cs"]) / len(v["cs"])) if v["cs"] else None, len(v["gt"]), v["u0"], e)
          for (f, e), v in agg.items() if f == fam]
    xs.sort(key=lambda x: -x[0])
    hi = [i + 1 for i, x in enumerate(xs) if x[1] is not None and x[1] > 0.7]
    print("%s distinct %d corr>0.7 n %d ranks %s" % (fam, len(xs), len(hi), hi[:12]))
    for x in xs[:5]:
        print("  gt %.3f corr %s n %d u%d %s" % (x[0], x[1] if x[1] is None else round(x[1], 3), x[2], x[3], x[4][:80]))
