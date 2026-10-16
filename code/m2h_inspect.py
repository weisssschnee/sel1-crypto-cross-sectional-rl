"""One-line-per-run summary of a BASIN3 run directory (keeps remote output short)."""
import collections
import json
import os
import sys

d = sys.argv[1]
tail = int(sys.argv[2]) if len(sys.argv) > 2 else 3
print("files:", sorted(os.listdir(d)))
inv = collections.Counter()
n = 0
p = os.path.join(d, "FORMULAS.jsonl")
if os.path.exists(p):
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        n += 1
        if r.get("invalid_reason"):
            inv[str(r["invalid_reason"])[:40]] += 1
    print("formulas %d invalid %d %s" % (n, sum(inv.values()), dict(inv)))
p = os.path.join(d, "BATCH_LOG.jsonl")
if os.path.exists(p):
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    for r in rows[-tail:]:
        fam = " ".join("%s:n%d adv%s" % (f.get("basin"), f.get("elite_count", -1),
                                         None if f.get("elite_advantage_mean") is None
                                         else round(f["elite_advantage_mean"], 4))
                       for f in r.get("per_family", []))
        print("upd %3d %6.1fs valid %3d pos%% %5.1f std %.3f | %s | pg %s gn %s"
              % (r["update"], r.get("seconds", -1), r["n_valid"],
                 (r.get("reward_positive_fraction") or 0) * 100,
                 r.get("reward_valid_std", 0), fam,
                 None if r.get("pg_loss") is None else round(r["pg_loss"], 5),
                 None if r.get("gradient_norm") is None else round(r["gradient_norm"], 3)))
    print("updates logged:", len(rows))
for f in ("TRAIN_RECEIPT_V1.json", "STOPPED.json", "HEARTBEAT.json"):
    p = os.path.join(d, f)
    if os.path.exists(p):
        print(f, json.dumps(json.load(open(p)))[:300])
