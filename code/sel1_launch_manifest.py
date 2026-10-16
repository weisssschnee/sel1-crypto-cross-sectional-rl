"""XS_SEL1 LAUNCH_MANIFEST (supplement 10-15 late night): written once from SEL1_NULLPLANT.json (chosen cell) and
SEL1_REWARD_RULE.json; the chain bat reads its flags from here. No training, no DEV, no OOS. New file.
    python -I sel1_launch_manifest.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sel1_common as S

OUT = os.path.join(S.OUT_ROOT, "SEL1_LAUNCH_MANIFEST.json")
BATCH = 128
BUDGET = dict(PC=125000, NULL=500000, REAL=500000)          # formula evaluations (order s4 / supplement 10-15)
SMOKE_UPDATES = 8                                            # smoke: PC flags, 8 updates
SEC_PER_UPDATE = 17.0                                        # SEL1_SPEED_V4 (fastts + gpu-select, 8 CPU workers)


def main():
    if os.path.exists(OUT):
        raise SystemExit("%s exists" % OUT)
    g = json.load(open(os.path.join(S.OUT_ROOT, "SEL1_NULLPLANT.json"), encoding="utf-8"))
    rr = json.load(open(os.path.join(S.OUT_ROOT, "SEL1_REWARD_RULE.json"), encoding="utf-8"))
    c = g["chosen"]
    if c is None:
        raise SystemExit("no chosen cell in SEL1_NULLPLANT.json: no launch")
    k, alpha, dd = int(c["K"]), float(c["alpha"]), float(rr["dd_max"][str(int(c["K"]))])
    upd = {r: -(-n // BATCH) for r, n in BUDGET.items()}
    common = ("--k %d --reward t_dd --dd-max %r --batch %d --workers 8 --gpu-workers 0 --memo global --fastts "
              "--gpu-select --checkpoint-every 16" % (k, dd, BATCH))
    runs = dict(PC=dict(run="SEL1_PC", flags="--mode pcnull --alpha %r --updates %d %s" % (alpha, upd["PC"], common)),
                NULL=dict(run="SEL1_NULL", flags="--mode null --updates %d %s" % (upd["NULL"], common)),
                REAL=dict(run="SEL1_REAL", flags="--mode real --updates %d %s" % (upd["REAL"], common)))
    man = dict(
        schema="SEL1_LAUNCH_MANIFEST_V1", order="XS_SEL1 supplement 10-15 late night",
        user_ruling="'回撤你看着来吧，其他决议部署然后赶快训练' (relayed by session QuantMind 训练特征构筑)",
        speed=dict(fastts=True, gpu_select=True,
                   reproducibility="selections and A-half rewards bit-identical to the baseline (SEL1_SPEED_V3/V4, "
                                   "SEL1_SELCHECK/SELCHECK2, SEL1_SPEED_V5); GPU backprop is not bit-reproducible "
                                   "(same code twice: weights differ by 2.1e-6), so pg_loss may differ in the last bit"),
        reward=dict(kind="t_dd", rule_file="SEL1_REWARD_RULE.json", t=rr["t"], dd=rr["dd"], dd_max=rr["dd_max"],
                    dd_max_used=dd),
        plant_check=dict(file="SEL1_NULLPLANT.json", chosen=dict(strength_bpd=c["strength_bpd"], K=k, alpha=alpha,
                                                                  plant_t_B=c["plant"]["t_B"], plant_rank=c["plant_rank"],
                                                                  pool=c["pool"], combo="%s/%s/%s" % (c["plant"]["h"], c["plant"]["state"], c["plant"]["dir"]),
                                                                  fixed_72_all_long_t_B=c["fixed"]["t_B"]),
                         table=[dict(strength_bpd=r["strength_bpd"], K=r["K"], t_B=r["plant"]["t_B"], rank=r["plant_rank"],
                                     passed=r["passed"]) for r in g["table"]]),
        detectable="this training can only find signals of >= %.0f bp/day (cost0, A-half gain of the plant book at "
                   "K %d; the smallest passing strength)" % (c["strength_bpd"], k),
        worlds=dict(PC="NULL world + plant (pcnull, the plant check's world), alpha %r" % alpha,
                    NULL="NULL world (per-coin circular roll, NULL_SEED %d), no plant" % S.NULL_SEED,
                    REAL="V4 TRAIN 2021..2025, real"),
        smoke=dict(run="SEL1_SMOKE", updates=SMOKE_UPDATES, flags="PC flags with --updates %d" % SMOKE_UPDATES),
        budget_evals=BUDGET, updates=upd, runs=runs,
        eta_hours=round(sum(upd.values()) * SEC_PER_UPDATE / 3600.0, 1),
        rounds=dict(pc="sel1_round --mode pc: found_corr = a hall entry with |corr| > 0.7 to the plant, else stop",
                    null="sel1_round --mode null: T_null = max B-half t of the NULL hall; PC found = corr hit with "
                         "t_B > T_null, else stop",
                    real="sel1_round --mode real: gates 1-4 of order s2 + behaviour dedup"),
        one_gpu_run=True, dev_read=False, oos_read=False, written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(man, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for r in ("PC", "NULL", "REAL"):
        open(os.path.join(S.OUT_ROOT, "_flags_%s.txt" % r), "w").write(runs[r]["flags"])
    smoke = runs["PC"]["flags"].replace("--updates %d " % upd["PC"], "--updates %d " % SMOKE_UPDATES)
    assert smoke != runs["PC"]["flags"]
    open(os.path.join(S.OUT_ROOT, "_flags_SMOKE.txt"), "w").write(smoke)
    print("launch manifest: K %d alpha %.3e dd_max %.4f updates %s eta %.1f h" % (k, alpha, dd, upd, man["eta_hours"]),
          flush=True)


if __name__ == "__main__":
    main()
