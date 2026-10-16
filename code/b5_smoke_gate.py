"""BASIN5 s5 smoke gate: reads one finished B5 run dir, writes SMOKE_GATE.json, exit 0 = pass.

Pass needs all of: TRAIN_RECEIPT_V1.json, the last checkpoint, a BATCH_LOG row per update with no
skipped parameter step and a finite gradient norm, a PROGRESS_16.jsonl row, every family with at
least one valid formula in the last 4 updates (T1's terminals are all levels, so at init ~90% of
T1 draws are grammar rejects by construction; the per-update valid / reject sequence is written
out so the trend is visible), and a grammar reject rate below 0.95 overall.

    python b5_smoke_gate.py RUN_NAME --updates 16
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--updates", type=int, default=16)
    a = ap.parse_args()
    d = os.path.join(ROOT, "artifacts", "m2h_v1", a.run)
    fails = []
    rows = []
    p = os.path.join(d, "BATCH_LOG.jsonl")
    if os.path.exists(p):
        rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    if not os.path.exists(os.path.join(d, "TRAIN_RECEIPT_V1.json")):
        fails.append("no receipt")
    if not os.path.exists(os.path.join(d, "checkpoint_%04d.pt" % (a.updates - 1))):
        fails.append("no checkpoint_%04d.pt" % (a.updates - 1))
    if len(rows) != a.updates:
        fails.append("BATCH_LOG rows %d != %d" % (len(rows), a.updates))
    prog = os.path.join(d, "PROGRESS_16.jsonl")
    n_prog = sum(1 for l in open(prog, encoding="utf-8") if l.strip()) if os.path.exists(prog) else 0
    if n_prog < 1:
        fails.append("no PROGRESS_16 row")
    skipped = [r["update"] for r in rows if r.get("skipped_param_step")]
    if skipped:
        fails.append("skipped param step at %s" % skipped)
    badg = [r["update"] for r in rows if r.get("gradient_norm") is None
            or not math.isfinite(float(r["gradient_norm"]))]
    if badg:
        fails.append("gradient norm missing / nonfinite at %s" % badg)
    fam = {}
    for r in rows:
        for f in r.get("per_family", []):
            x = fam.setdefault(f["basin"], dict(n=0, valid=0, rej=0, upd=0, corr=[], valid_seq=[],
                                                rej_seq=[]))
            x["n"] += int(f.get("n", 0))
            x["valid"] += int(f.get("n_valid", 0))
            x["rej"] += int(f.get("grammar_reject", 0) or 0)
            x["valid_seq"].append(int(f.get("n_valid", 0)))
            x["rej_seq"].append(int(f.get("grammar_reject", 0) or 0))
            x["upd"] += 1
            if f.get("corr_s") is not None:
                x["corr"].append(float(f["corr_s"]))
    tot_n = sum(x["n"] for x in fam.values())
    tot_rej = sum(x["rej"] for x in fam.values())
    for b, x in fam.items():
        if x["upd"] >= 4 and sum(x["valid_seq"][-4:]) == 0:
            fails.append("%s no valid formula in the last 4 updates" % b)
    rej_rate = tot_rej / tot_n if tot_n else float("nan")
    if not tot_n or rej_rate >= 0.95:
        fails.append("grammar reject rate %.3f" % rej_rate)
    out = dict(schema="B5_SMOKE_GATE_V1", run=a.run, passed=not fails, fails=fails,
               updates=len(rows), progress_rows=n_prog, grammar_reject_rate=round(rej_rate, 4),
               families={b: dict(valid_per_update=round(x["valid"] / max(x["upd"], 1), 2),
                                 grammar_reject_rate=round(x["rej"] / max(x["n"], 1), 4),
                                 valid_seq=x["valid_seq"], grammar_reject_seq=x["rej_seq"],
                                 corr_s_last=x["corr"][-1] if x["corr"] else None)
                         for b, x in fam.items()},
               sec_per_update=round(sum(r["seconds"] for r in rows) / max(len(rows), 1), 1),
               written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(out, open(os.path.join(d, "SMOKE_GATE.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("SMOKE_GATE %s rej %.3f %.1fs/upd %s" % ("PASS" if not fails else "FAIL", rej_rate,
                                                 out["sec_per_update"], "; ".join(fails)))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
