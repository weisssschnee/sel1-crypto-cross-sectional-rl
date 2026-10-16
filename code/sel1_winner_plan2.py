"""XS_SEL1 SEL1_ARMS winner chain plan v2 = sel1_winner_plan.py (not modified) for the merged ARMS2 result
(supplement 10-16 noon): run -> key by the table RUN_KEY (v1 arms SEL1_ARMS_*, v2 arms SEL1_ARMS2_*); PGRISK and GFN2 run
sel1_trainer_x2.py with their switches, GA2 runs sel1_ga2.py. Header of the v1 file follows.

XS_SEL1 SEL1_ARMS winner chain plan (supplement 10-15 evening): another arm ranked above V2 RL. TRAIN only; no DEV,
no OOS. New file. Reads SEL1_ARMS_RESULT.json and the winner's receipt; writes SEL1_LAUNCH_MANIFEST_W.json and the
chain's inputs (_cmd_W.txt trainer script, _pc_W.txt the PC run, _flags_NULL_W.txt, _flags_REAL_W.txt).

  PC         the winner's SEL1_ARMS run (round pc / round null judge P1, as the V2 chain).
  switches   the winner's own (RL none, DEDUP --dedup, GFN --gfn 1, POOL --pool 20, GA sel1_ga.py); FROZEN runs as
             random + dedup (--frozen --dedup, afternoon order: a random-baseline win means search volume only).
  budget     NULL and REAL equal: total <= 96 h, ROUNDS_H reserved for rounds / combo, the rest split in two; the
             trainer arms in updates (seconds per update of the winner's ARMS run), GA in behaviour-distinct
             formulas (--budget, its ARMS rate); --max-hours HOURS_CAP each as a safety cap (receipt on stop).
    python -I sel1_winner_plan.py [--result F] [--winner RUN] [--dry]
-> rc 0 written (--dry: printed only), 3 refused (W runs exist / winner is RL / no result)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sel1_common as S

TOTAL_H, ROUNDS_H = 96.0, 2.0
EACH_H = (TOTAL_H - ROUNDS_H) / 2.0
HOURS_CAP = EACH_H + 0.5
DD_MAX = 2.283547812033886
BASE = ("--k 5 --reward t_dd --dd-max %r --batch 128 --workers 16 --gpu-workers 0 --memo global --fastts --gpu-select "
        "--combo-sweep --checkpoint-every 16 --receipt-on-stop" % DD_MAX)
EXTRA = dict(RL="", DEDUP="--dedup", GFN="--gfn 1", POOL="--pool 20", FROZEN="--frozen --dedup",
             PGRISK="--risk-seek 0.05 --pq 64 --ent-gamma 1.0", GFN2="--gfn2 1 --explore 0.05 --pq 64")
TRAINER = dict(PGRISK="sel1_trainer_x2.py", GFN2="sel1_trainer_x2.py")
GA_CMD = dict(GA="sel1_ga.py", GA2="sel1_ga2.py")
RUN_KEY = dict(SEL1_ARMS_RL="RL", SEL1_ARMS_DEDUP="DEDUP", SEL1_ARMS_FROZEN="FROZEN", SEL1_ARMS_POOL="POOL",
               SEL1_ARMS_GFN="GFN", SEL1_ARMS_GA="GA", SEL1_ARMS2_PGRISK="PGRISK", SEL1_ARMS2_GFN="GFN2",
               SEL1_ARMS2_GA="GA2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", default=os.path.join(S.OUT_ROOT, "SEL1_ARMS2_RESULT.json"))
    ap.add_argument("--winner", default=None)                   # tests only: override the result's winner run
    ap.add_argument("--dry", action="store_true")               # tests only: print, write nothing
    a = ap.parse_args()
    res_p = a.result
    if not os.path.exists(res_p):
        print("refused: no SEL1_ARMS_RESULT.json")
        sys.exit(3)
    for r in ("SEL1_W_NULL", "SEL1_W_REAL"):
        if os.path.exists(os.path.join(S.OUT_ROOT, r)):
            print("refused: %s exists" % r)
            sys.exit(3)
    res = json.load(open(res_p, encoding="utf-8"))
    run = a.winner or res.get("winner_run")
    if not run or RUN_KEY.get(run) in (None, "RL"):
        print("refused: winner %s" % run)
        sys.exit(3)
    key = RUN_KEY[run]
    rec = json.load(open(os.path.join(S.OUT_ROOT, run, "TRAIN_RECEIPT_V1.json"), encoding="utf-8"))
    sec = float(rec["compute_seconds"])
    if key in GA_CMD:
        rate = float(rec["n_behav"]) / sec
        budget = int(rate * EACH_H * 3600.0)
        cmd = GA_CMD[key]
        tail = "--reward t_dd --dd-max %r --k 5 --workers 16 --budget %d --max-hours %s" % (DD_MAX, budget, HOURS_CAP)
        unit = dict(kind="behaviour-distinct formulas", per_second=rate, each=budget)
    else:
        n_upd = int(rec["last_update"]) + 1
        spu = sec / n_upd
        budget = int(EACH_H * 3600.0 / spu)
        cmd = TRAINER.get(key, "sel1_trainer_x.py")
        tail = "%s %s --updates %d --max-hours %s" % (BASE, EXTRA[key], budget, HOURS_CAP)
        unit = dict(kind="updates", seconds_per_update=spu, each=budget)
    fl = {m: ("--mode %s " % m) + " ".join(tail.split()) for m in ("null", "real")}
    if a.dry:
        print("dry winner %s (%s): %s each %s | %s" % (run, cmd, unit["kind"], budget, fl["null"]))
        sys.exit(0)
    for m, f in fl.items():
        open(os.path.join(S.OUT_ROOT, "_flags_%s_W.txt" % m.upper()), "w").write(f + "\n")
    open(os.path.join(S.OUT_ROOT, "_cmd_W.txt"), "w").write(cmd + "\n")
    open(os.path.join(S.OUT_ROOT, "_pc_W.txt"), "w").write(run + "\n")
    man = dict(schema="SEL1_LAUNCH_MANIFEST_W_V1", order="XS_SEL1 supplement 10-16 noon (ARMS2 merged winner)",
               winner=next(x["arm"] for x in res["arms"] if x["run"] == run), winner_run=run, ranking=res.get("ranking"),
               arms=[{k: a.get(k) for k in ("arm", "rank", "plants", "p1_found", "p2_found", "time_s", "composite_tB",
                                             "behav_tA_above_3", "compute_s")} for a in res["arms"]],
               pc_run=run, trainer=cmd, switches=EXTRA.get(key, key), frozen_rule="random + dedup" if key == "FROZEN" else None,
               budget=dict(total_h=TOTAL_H, rounds_h=ROUNDS_H, each_h=EACH_H, cap_h=HOURS_CAP, **unit),
               runs=dict(null="SEL1_W_NULL", real="SEL1_W_REAL"), flags=fl,
               v2_kept=dict(pc="SEL1_PC_V2", null="SEL1_NULL_V2_PAUSED"),
               dev_read=False, oos_read=False, written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(man, open(os.path.join(S.OUT_ROOT, "SEL1_LAUNCH_MANIFEST_W.json"), "w", encoding="utf-8"), indent=1,
              default=str, ensure_ascii=False)
    print("winner %s (%s): %s each %s, cap %.1f h" % (run, cmd, unit["kind"], budget, HOURS_CAP))
    sys.exit(0)


if __name__ == "__main__":
    main()
