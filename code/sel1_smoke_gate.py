"""XS_SEL1 smoke gate: receipt present, BATCH_LOG has --updates rows, every update n_valid >= 20, no skipped /
non-finite step, finite pg_loss, the --reward score logged (t_dd), HOF.json present. Prints the mean update seconds.
rc 0 pass, 5 fail. New file.
    python -I sel1_smoke_gate.py RUN --updates N
"""
import argparse
import json
import math
import os
import sys

OUT_ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\XS_SEL1"

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--updates", type=int, required=True)
a = ap.parse_args()
d = os.path.join(OUT_ROOT, a.run)
bad = [f for f in ("TRAIN_RECEIPT_V1.json", "HOF.json", "TRAIN_MANIFEST.json") if not os.path.exists(os.path.join(d, f))]
rows = [json.loads(x) for x in open(os.path.join(d, "BATCH_LOG.jsonl"), encoding="utf-8") if x.strip()]
if len(rows) != a.updates:
    bad.append("rows %d != %d" % (len(rows), a.updates))
for r in rows:
    if r["n_valid"] < 20 or r.get("skipped") or r.get("skipped_nonfinite_update"):
        bad.append("u%d valid %d skipped %s" % (r["update"], r["n_valid"], r.get("skipped")))
    if r.get("pg_loss") is None or not math.isfinite(r["pg_loss"]):
        bad.append("u%d pg_loss %s" % (r["update"], r.get("pg_loss")))
    if "score" not in r:
        bad.append("u%d no score" % r["update"])
sec = [r["seconds"] for r in rows[1:]]
print("smoke gate %s: %d rows, mean update %.1fs (after the first) -> %s %s"
      % (a.run, len(rows), sum(sec) / max(len(sec), 1), "PASS" if not bad else "FAIL", bad[:5]), flush=True)
sys.exit(0 if not bad else 5)
