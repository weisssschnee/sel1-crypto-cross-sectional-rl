"""CR1 smoke gate: receipt present, every update n_valid >= 20, no skipped / non-finite step, finite loss,
pool selftest bit-equal. Prints the mean step times (the speed-up check). rc 0 pass, 5 fail.
    python -I cr1_smoke_gate.py RUN --updates 16
"""
import argparse
import json
import math
import os
import sys

A_ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1"

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--updates", type=int, required=True)
a = ap.parse_args()
d = os.path.join(A_ROOT, a.run)
bad = []
if not os.path.exists(os.path.join(d, "TRAIN_RECEIPT_V1.json")):
    bad.append("no receipt")
man = json.load(open(os.path.join(d, "LAUNCH_MANIFEST.json"), encoding="utf-8"))
if not man["pool_selftest"]["ok"]:
    bad.append("pool selftest")
rows = [json.loads(x) for x in open(os.path.join(d, "BATCH_LOG.jsonl"), encoding="utf-8") if x.strip()]
if len(rows) != a.updates:
    bad.append("rows %d != %d" % (len(rows), a.updates))
for r in rows:
    if r["n_valid"] < 20 or r.get("skipped") or r.get("skipped_nonfinite_update"):
        bad.append("u%d valid %d skipped %s" % (r["update"], r["n_valid"], r.get("skipped")))
    if r.get("pg_loss") is None or not math.isfinite(r["pg_loss"]):
        bad.append("u%d loss %s" % (r["update"], r.get("pg_loss")))
keys = rows[0]["times"].keys() if rows else []
tm = {k: round(sum(r["times"][k] for r in rows[1:]) / max(len(rows) - 1, 1), 2) for k in keys}
sec = round(sum(r["seconds"] for r in rows[1:]) / max(len(rows) - 1, 1), 2)
uniq = round(sum(r["n_unique"] for r in rows) / max(len(rows), 1), 1)
json.dump(dict(run=a.run, ok=not bad, bad=bad[:20], mean_seconds=sec, mean_times=tm, mean_unique=uniq,
               vram_gib=rows[-1].get("vram_gib") if rows else None),
          open(os.path.join(d, "SMOKE_GATE.json"), "w"), indent=1)
print("SMOKE GATE %s: %.1fs/update %s uniq %.1f vram %s %s" % (
    "PASS" if not bad else "FAIL", sec, tm, uniq, rows[-1].get("vram_gib") if rows else None, bad[:5]), flush=True)
sys.exit(0 if not bad else 5)
