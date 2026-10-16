"""Duplicate rate in a B5 run's FORMULAS.jsonl: per update, distinct (body, hold) / rows; last N
updates and overall. One line. Read-only.

    python b5_dup.py RUN_NAME [N]
"""
import json, os, sys
from collections import defaultdict

A = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1"
run = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
rows, keys = defaultdict(int), defaultdict(set)
for l in open(os.path.join(A, run, "FORMULAS.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    u = r["update"]
    rows[u] += 1
    keys[u].add((r["expr"], r["hold_h"]))
us = sorted(rows)
last = us[-N:]
fr = lambda ss: sum(len(keys[u]) for u in ss) / max(sum(rows[u] for u in ss), 1)
print("DUP %s updates %d last_u %d distinct_frac all %.3f last%d %.3f first%d %.3f"
      % (run, len(us), us[-1], fr(us), N, fr(last), N, fr(us[:N])))
