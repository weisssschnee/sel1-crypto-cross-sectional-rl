"""BASIN5 debug-round smoke gate (after b5_smoke_gate.py on the same dir). Exit 0 = pass.

Checks the one code change: the receipt carries the dbg block; every BATCH_LOG family row has
grammar_reject == 0 (the mask must make the post-hoc check redundant); the valid fraction per family
is reported, not gated (zero-coverage / zero-sd invalids are not a mask property; b5_smoke_gate
already requires valid formulas in the last 4 updates); no skipped parameter step; peak VRAM below
--vram-max GiB. Writes DBG_GATE.json.

    python b5_dbg_gate.py RUN_NAME [--vram-max 14]
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
    ap.add_argument("--vram-max", type=float, default=14.0)
    a = ap.parse_args()
    d = os.path.join(ROOT, "artifacts", "m2h_v1", a.run)
    fails = []
    rec = json.load(open(os.path.join(d, "TRAIN_RECEIPT_V1.json"), encoding="utf-8"))
    if "dbg" not in rec:
        fails.append("receipt has no dbg block")
    rows = [json.loads(l) for l in open(os.path.join(d, "BATCH_LOG.jsonl"), encoding="utf-8") if l.strip()]
    rej, valid, n = {}, {}, {}
    for r in rows:
        if r.get("skipped_param_step"):
            fails.append("u%s skipped %s" % (r["update"], r["skipped_param_step"]))
        g = r.get("gradient_norm")
        if g is None or not math.isfinite(float(g)):
            fails.append("u%s gradient norm %s" % (r["update"], g))
        for f in r.get("per_family", []):
            b = f["basin"]
            rej[b] = rej.get(b, 0) + int(f.get("grammar_reject") or 0)
            valid[b] = valid.get(b, 0) + int(f.get("n_valid") or 0)
            n[b] = n.get(b, 0) + int(f.get("n") or 0)
    for b in n:
        if rej[b]:
            fails.append("%s grammar rejects %d (mask leak)" % (b, rej[b]))
    vram = rec.get("peak_vram_alloc_gib")
    if vram is None or vram > a.vram_max:
        fails.append("peak vram %s GiB" % vram)
    sec = round(sum(r["seconds"] for r in rows) / max(len(rows), 1), 1)
    out = dict(schema="B5_DBG_GATE_V1", run=a.run, passed=not fails, fails=fails[:20], grammar_reject=rej,
               valid_frac={b: round(valid[b] / max(n[b], 1), 3) for b in n}, peak_vram_alloc_gib=vram,
               sec_per_update=sec, written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(out, open(os.path.join(d, "DBG_GATE.json"), "w", encoding="utf-8"), indent=1)
    print("DBG_GATE %s rej %s valid %s vram %s %.1fs/upd %s" % ("PASS" if not fails else "FAIL", rej,
                                                               out["valid_frac"], vram, sec, "; ".join(fails[:3])))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
