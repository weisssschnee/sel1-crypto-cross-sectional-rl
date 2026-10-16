"""XS_SEL1 LAUNCH_MANIFEST v2 (supplement 10-15 morning: --combo-sweep). New file; v1 and its manifest stay as they are.
  --phase smoke   writes _flags_SMOKE_V2.txt (PC flags, 8 updates); refuses if SEL1_LAUNCH_MANIFEST_V2.json exists.
  --phase final   reads the smoke run's BATCH_LOG (mean update seconds after the first), sets the budget, writes
                  SEL1_LAUNCH_MANIFEST_V2.json and _flags_{PC,NULL,REAL}_V2.txt.
Budget: total ETA <= 96 h -> 125k / 500k / 500k evaluations; above it NULL = REAL scaled to fit 96 h and PC = REAL / 4.
Plant cell, K, alpha, dd_max: unchanged from SEL1_LAUNCH_MANIFEST.json (the plant check stays valid: sel1_sweepcheck).
No training here, no DEV, no OOS.
    python -I sel1_launch_manifest_v2.py --phase smoke|final
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sel1_common as S

OUT = os.path.join(S.OUT_ROOT, "SEL1_LAUNCH_MANIFEST_V2.json")
V1 = os.path.join(S.OUT_ROOT, "SEL1_LAUNCH_MANIFEST.json")
CHECK = os.path.join(S.OUT_ROOT, "SEL1_SWEEPCHECK.json")
BATCH, SMOKE_UPDATES, MAX_HOURS = 128, 8, 96.0
BUDGET = dict(PC=125000, NULL=500000, REAL=500000)
WORKERS = 16
CODE = ("sel1_trainer.py", "sel1_sweep.py", "sel1_common.py", "sel1_fastts.py", "sel1_cse.py", "sel1_round.py",
        "sel1_sweepcheck.py", "sel1_launch_manifest_v2.py")


def sha16(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def flags(mode, updates, v1):
    ch = v1["plant_check"]["chosen"]
    common = ("--k %d --reward t_dd --dd-max %r --batch %d --workers %d --gpu-workers 0 --memo global --fastts "
              "--gpu-select --combo-sweep --checkpoint-every 16" % (ch["K"], v1["reward"]["dd_max_used"], BATCH, WORKERS))
    alpha = " --alpha %r" % ch["alpha"] if mode == "pcnull" else ""
    return "--mode %s%s --updates %d %s" % (mode, alpha, updates, common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=("smoke", "final"))
    ap.add_argument("--smoke", default="SEL1_SMOKE_V2")
    a = ap.parse_args()
    if os.path.exists(OUT):
        raise SystemExit("%s exists" % OUT)
    v1 = json.load(open(V1, encoding="utf-8"))
    chk = json.load(open(CHECK, encoding="utf-8"))
    if not chk["passed"]:
        raise SystemExit("SEL1_SWEEPCHECK did not pass: no launch")
    if a.phase == "smoke":
        open(os.path.join(S.OUT_ROOT, "_flags_SMOKE_V2.txt"), "w").write(flags("pcnull", SMOKE_UPDATES, v1))
        print("smoke v2 flags written", flush=True)
        return
    rows = [json.loads(x) for x in open(os.path.join(S.OUT_ROOT, a.smoke, "BATCH_LOG.jsonl"), encoding="utf-8") if x.strip()]
    sec = sum(r["seconds"] for r in rows[1:]) / max(len(rows) - 1, 1)
    upd = {r: -(-n // BATCH) for r, n in BUDGET.items()}
    eta = sum(upd.values()) * sec / 3600.0
    scaled = eta > MAX_HOURS
    if scaled:
        real = int(MAX_HOURS * 3600.0 / sec / 2.25)               # PC + NULL + REAL = REAL / 4 + 2 REAL
        upd = dict(PC=real // 4, NULL=real, REAL=real)
        eta = sum(upd.values()) * sec / 3600.0
    runs = dict(PC=dict(run="SEL1_PC", flags=flags("pcnull", upd["PC"], v1)),
                NULL=dict(run="SEL1_NULL", flags=flags("null", upd["NULL"], v1)),
                REAL=dict(run="SEL1_REAL", flags=flags("real", upd["REAL"], v1)))
    man = dict(
        schema="SEL1_LAUNCH_MANIFEST_V2", order="XS_SEL1 supplement 10-15 morning (combo sweep)",
        ruling="QuantMind 10-15: the policy samples formula tokens only; each formula takes the A-half best t_dd of all "
               "60 combos (the plant check's rule); hall / B-half / T_null / gates use that A-half combo",
        combo_sweep=dict(flag="--combo-sweep", module="sel1_sweep.py", check_file="SEL1_SWEEPCHECK.json",
                         check=dict(bodies=chk["bodies"], max_abs_diff=chk["max_abs_diff"], checks=chk["checks"],
                                    plant=dict(combo=chk["plant"]["pick_new"], t_A=chk["plant"]["best_rescored"],
                                               nullplant_t_A=chk["plant"]["nullplant_score_A"]),
                                    seconds_per_body=chk["seconds_per_body"]),
                         not_a_leg_decomposition="the stateful controller's L1 band acts on the whole book and the state "
                                                 "gate zeroes desired weights before it, so combos are settled, not mixed "
                                                 "from legs (sel1_sweep docstring); cost0-only pass + one controller call "
                                                 "per (hold, state)"),
        v1_manifest=dict(file="SEL1_LAUNCH_MANIFEST.json", sha16=sha16(V1), old_pc="SEL1_PC_V1_COMBOTOKEN (renamed, kept)"),
        plant_check=v1["plant_check"], reward=v1["reward"], detectable=v1["detectable"], worlds=v1["worlds"],
        smoke=dict(run=a.smoke, updates=len(rows), mean_update_seconds=round(sec, 2)),
        budget=dict(rule="ETA <= %.0f h -> 125k/500k/500k; else NULL = REAL scaled to %.0f h, PC = REAL/4" % (MAX_HOURS, MAX_HOURS),
                    scaled=scaled, updates=upd, evals={k: v * BATCH for k, v in upd.items()},
                    eta_hours=round(eta, 1), workers=WORKERS),
        runs=runs, code_sha16={f: sha16(os.path.join(HERE, f)) for f in CODE},
        rounds=v1["rounds"], one_gpu_run=True, dev_read=False, oos_read=False,
        written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(man, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for r in ("PC", "NULL", "REAL"):
        open(os.path.join(S.OUT_ROOT, "_flags_%s_V2.txt" % r), "w").write(runs[r]["flags"])
    print("launch manifest v2: %.1f s/update, updates %s, scaled %s, eta %.1f h" % (sec, upd, scaled, eta), flush=True)


if __name__ == "__main__":
    main()
