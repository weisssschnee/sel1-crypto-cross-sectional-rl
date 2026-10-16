"""Wait (up to --hours) for the full step-1 gate's DECISION.json (limit 0). Exit 0 = proceed with
the expected plant, 20 = the gate says stop or a different plant, 21 = timed out."""
import argparse
import json
import os
import sys
import time

P = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\B5_GATE1\DECISION.json"
ap = argparse.ArgumentParser()
ap.add_argument("--plant", default="p32")
ap.add_argument("--hours", type=float, default=8.0)
a = ap.parse_args()
t0 = time.time()
while time.time() - t0 < a.hours * 3600:
    if os.path.exists(P):
        d = json.load(open(P, encoding="utf-8"))
        if d.get("limit") == 0:
            print("GATE proceed=%s plant=%s hof_rank=%s" % (d.get("proceed"), d.get("plant"), d.get("hof_rank")), flush=True)
            sys.exit(0 if d.get("proceed") and d.get("plant") == a.plant else 20)
    time.sleep(60)
print("GATE timed out", flush=True)
sys.exit(21)
