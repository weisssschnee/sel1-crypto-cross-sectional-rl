"""BASIN5_PC_ENT smoke gate, run after b5_smoke_gate.py on the same dir. Exit 0 = pass.

Checks the entropy controller only: every BATCH_LOG / PROGRESS_16 family row carries entropy,
ent_alpha and ent_alpha_next; every alpha is inside [alpha_min, alpha_max]; alpha moved in at
least one family (the smoke runs with an H* above the start entropy so the controller has to
act); the last checkpoint holds ent_alpha; the recorded update rule reproduces every logged
alpha_next from alpha and H; peak VRAM below --vram-max GiB. Writes ENT_GATE.json.

    python b5_ent_gate.py RUN_NAME [--vram-max 14]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import torch

ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--vram-max", type=float, default=14.0)
    a = ap.parse_args()
    d = os.path.join(ROOT, "artifacts", "m2h_v1", a.run)
    fails = []
    rec = json.load(open(os.path.join(d, "TRAIN_RECEIPT_V1.json"), encoding="utf-8"))
    ent = rec.get("ent") or {}
    lo, hi, eta, tgt = ent.get("alpha_min"), ent.get("alpha_max"), ent.get("eta"), ent.get("target")
    if None in (lo, hi, eta, tgt):
        fails.append("receipt has no ent block")
        lo, hi, eta, tgt = 0.03, 1.0, 0.05, 2.0
    rows = [json.loads(l) for l in open(os.path.join(d, "BATCH_LOG.jsonl"), encoding="utf-8") if l.strip()]
    prog = [json.loads(l) for l in open(os.path.join(d, "PROGRESS_16.jsonl"), encoding="utf-8") if l.strip()]
    moved, worst = {}, 0.0
    for r in rows:
        for f in r.get("per_family", []):
            b = f.get("basin")
            if any(f.get(k) is None for k in ("entropy", "ent_alpha", "ent_alpha_next")):
                fails.append("u%s %s missing entropy/ent_alpha" % (r["update"], b))
                continue
            al, an, h = float(f["ent_alpha"]), float(f["ent_alpha_next"]), float(f["entropy"])
            if not (lo - 1e-9 <= al <= hi + 1e-9 and lo - 1e-9 <= an <= hi + 1e-9):
                fails.append("u%s %s alpha out of bounds %s %s" % (r["update"], b, al, an))
            if not r.get("skipped_param_step"):
                # entropy is rounded to 4 dp in the log, alpha to 6 dp
                want = min(max(math.exp(math.log(al) + eta * (tgt - h)), lo), hi)
                worst = max(worst, abs(want - an) / an)
            moved[b] = max(moved.get(b, 0.0), an / lo)
    if worst > 1e-3:
        fails.append("alpha_next not reproduced by the rule, worst rel err %.2e" % worst)
    if not moved or max(moved.values()) <= 1.0001:
        fails.append("alpha never moved off %s" % lo)
    for p in prog:
        for b, f in p.get("families", {}).items():
            if any(f.get(k) is None for k in ("entropy", "ent_alpha", "ent_alpha_next")):
                fails.append("PROGRESS u%s %s missing ent fields" % (p.get("update"), b))
    cks = sorted(glob.glob(os.path.join(d, "checkpoint_*.pt")))
    if not cks:
        fails.append("no checkpoint")
    else:
        ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
        if "ent_alpha" not in ck:
            fails.append("checkpoint has no ent_alpha")
    vram = rec.get("peak_vram_alloc_gib")
    if vram is None or vram > a.vram_max:
        fails.append("peak vram %s GiB" % vram)
    out = dict(schema="B5_ENT_GATE_V1", run=a.run, passed=not fails, fails=fails[:20],
               alpha_max_over_init={b: round(v, 4) for b, v in moved.items()},
               rule_worst_rel_err=worst, peak_vram_alloc_gib=vram, target=tgt,
               written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(out, open(os.path.join(d, "ENT_GATE.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("ENT_GATE %s alpha/init %s vram %s %s" % ("PASS" if not fails else "FAIL", out["alpha_max_over_init"],
                                                   vram, "; ".join(fails[:3])))
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
