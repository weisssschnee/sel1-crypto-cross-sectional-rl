"""Within-update duplicates in a B5 run's FORMULAS.jsonl: do rows with the same tokens in the same
update carry identical scores? If yes, scoring each (body, hold) once per update is result-neutral.
Read-only. One line.

    python b5_detcheck.py RUN_NAME [FIRST_UPDATE]
"""
import json, os, sys

A = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1"
F = ("invalid_reason", "gross", "net", "ic", "window_gross", "window_ic", "window_nav", "gross_t",
     "corr_s", "live_frac", "turnover_per_day", "beta_mkt", "gross_bp_day")
run = sys.argv[1]
u0 = int(sys.argv[2]) if len(sys.argv) > 2 else 0
first, rows, bad = {}, 0, 0
for l in open(os.path.join(A, run, "FORMULAS.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    if r["update"] < u0:
        continue
    rows += 1
    k = (r["update"], tuple(r["tokens"]))
    v = tuple(json.dumps(r.get(f)) for f in F)
    if k not in first:
        first[k] = v
    elif first[k] != v:
        bad += 1
print("DET %s from u%d rows %d keys %d dup_rows %d mismatched %d"
      % (run, u0, rows, len(first), rows - len(first), bad))
