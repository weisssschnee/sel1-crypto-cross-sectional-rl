"""XS_SEL1 s3.2 plant calibration and the two launch checks (CPU book + GPU VM, V4 TRAIN only, no DEV, no OOS).

  target   the ENR1b retail long/short cell T2_neg_retail_LS_24h|TS_lo|all|H72 (the strength the order names):
           A-half net +31.89 bp per 72 h event; with the entry/exit cost 12 bp x (1 + |beta|), median beta 1.305
           (ENR1 world table), its cost0 excess is ~59.6 bp per 72 h event = 0.83 bp per position-hour, i.e.
           19.9 bp/day for a book at gross exposure 1. The plant book is the plant formula's own sparse book
           (DELTA(TS_MEAN(log_quote_volume,168),168), hold 72, state all, long top 5 = gross 1).
  alpha    fwd[i,t] += alpha zp[i,t-1] (sel1_common pc world); gain(alpha) = A-half cost0 of the plant book in the pc
           world minus in the real world; alpha set by one linear step from alpha0 and checked (|gain-target| <= 10 %).
  check 1  visible: the plant book's B-half t (monthly cost0) >= 2 in the pc world.
  check 2  rankable: real candidate pool = CR1_PC_P8 HOF top 50 + B5_DBG_REAL HOF_RANKED (90), deduplicated; every
           pool body and the plant body get the combo (hold x state x direction, 60) with the best A-half cost0
           in the pc world, as the trainer's reward would pick; the plant's B-half t must rank <= 10 among them.
  Either check failing -> no training (rc 7), back to the order issuer; nothing is re-tuned.
    python -I sel1_calib.py
"""
from __future__ import annotations

import itertools
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import sel1_common as S
import cr1_common as Q
import m2h_dense_common as D

OUT = os.path.join(S.OUT_ROOT, "SEL1_LAUNCH_CHECK.json")
CR1_HOF = os.path.join(S.A_ROOT, "CR1_PC_P8", "HOF.json")
B5_HOF = os.path.join(S.A_ROOT, "B5_DBG_REAL", "HOF_RANKED.json")
LS_NET_A, LS_HOLD, BETA_MED, COST_BP = 31.89, 72, 1.305, 12.0
TARGET_BPD = (LS_NET_A + COST_BP * (1 + BETA_MED)) / LS_HOLD * 24.0
ALPHA0 = 1e-4
PLANT_COMBO = (72, 0, 1)
COMBOS = list(itertools.product(S.HOLDS, range(len(S.STATES)), range(len(S.DIRS))))
RANK_MAX, T_MIN = 10, 2.0


def log(m):
    print(m, flush=True)


def best_combo(task):
    """(key, sels_A, sels_B) -> best A combo and its B-half monthly t (score worker)."""
    key, sa, sb = task
    v = S._WS["view"]
    best = None
    for h, st, dr in COMBOS:
        a = S.bpd(S.score_sel(sa, v, h, st, dr))
        if np.isfinite(a) and (best is None or a > best[0]):
            best = (a, h, st, dr)
    if best is None:
        return key, None
    a, h, st, dr = best
    rb = S.score_sel(sb, v, h, st, dr, halves=("B",))
    return key, dict(A_cost0_bpd=round(a, 3), h=h, state=S.STATES[st], dir=S.DIRS[dr],
                     B=S.summary(rb), tB=S.tstat([1e4 * r["cost0"] / r["days"] for r in rb]))


def main():
    t0 = time.time()
    if os.path.exists(OUT):
        raise SystemExit("%s exists" % OUT)
    device = torch.device("cuda")
    vm = Q.make_vm()
    pb = tuple(D.plant_body(vm))
    real = S.build("real", device, vm, log=log, keep_base=True)   # keep_base: zp as the pc world builds it
    sig_p = S.signal(real, vm, pb)
    h, st, dr = PLANT_COMBO
    g_real = S.bpd(S.score(sig_p, real, h, st, dr))
    zp = real["zp"]
    add = np.zeros_like(real["shard"]["fwd"])
    add[:, 1:] = np.nan_to_num(zp[:, :-1], nan=0.0)

    def world(alpha):
        w = dict(real)
        w["shard"] = S.no_winsor_shard(real["shard"], real["dead"], fwd=real["shard"]["fwd"] + alpha * add)
        return w

    def gain(alpha):
        return S.bpd(S.score(sig_p, world(alpha), h, st, dr)) - g_real
    g0 = gain(ALPHA0)
    alpha = ALPHA0 * TARGET_BPD / g0
    g1 = gain(alpha)
    ok_alpha = bool(abs(g1 - TARGET_BPD) <= 0.1 * TARGET_BPD)
    log("target %.2f bp/day: alpha0 %.2e gain %.2f -> alpha %.3e gain %.2f ok %s" % (TARGET_BPD, ALPHA0, g0, alpha, g1, ok_alpha))
    os.makedirs(S.OUT_ROOT, exist_ok=True)
    json.dump(dict(schema="SEL1_ALPHA_V1", alpha=alpha, target_bpd=TARGET_BPD, gain_bpd=g1, alpha0=ALPHA0, gain0_bpd=g0,
                   plant_body=list(pb), plant_combo=dict(h=h, state=S.STATES[st], dir=S.DIRS[dr]), real_plant_book_bpd=g_real,
                   rule=__doc__, written=time.strftime("%Y-%m-%dT%H:%M:%S%z")),
              open(S.ALPHA_JSON, "w", encoding="utf-8"), indent=1)
    pc = world(alpha)
    pc["mode"] = "pc"
    rb = S.score(sig_p, pc, h, st, dr, halves=("B",))
    tB_plant_fixed = S.tstat([1e4 * r["cost0"] / r["days"] for r in rb])
    visible = bool(tB_plant_fixed >= T_MIN)
    log("visible: plant book B t %.2f (>= %.1f) %s" % (tB_plant_fixed, T_MIN, visible))
    # pool
    bodies = [tuple(e["body"]) for e in json.load(open(CR1_HOF, encoding="utf-8"))["entries"][:50]]
    for fam in json.load(open(B5_HOF, encoding="utf-8"))["families"].values():
        bodies += [tuple(e["body"]) for e in fam]
    bodies = [b for b in dict.fromkeys(bodies) if b != pb]
    tasks, bad = [], {}
    for b in bodies + [pb]:
        try:
            sg = S.signal(pc, vm, b)
        except Exception as e:                                      # invalid on the V4 world: not in the pool
            bad[vm.describe(list(b), list(S.M.TERMINALS))] = str(e)[:80]
            continue
        tasks.append((b, S.select_months(sg, pc, ("A",)), S.select_months(sg, pc, ("B",))))
    cdir = S.write_cache(pc, os.path.join(S.OUT_ROOT, "_calib_cache"))
    with mp.get_context("spawn").Pool(20, initializer=S.init_score_worker, initargs=(cdir,)) as pool:
        res = dict(pool.imap_unordered(best_combo, tasks))
    rows = [dict(body=list(b), expr=vm.describe(list(b), list(S.M.TERMINALS)), plant=(b == pb), **r) for b, r in res.items() if r]
    rows.sort(key=lambda x: -(x["tB"] if np.isfinite(x["tB"]) else -1e9))
    rank = next(i + 1 for i, x in enumerate(rows) if x["plant"])
    rankable = bool(rank <= RANK_MAX)
    log("rankable: plant rank %d / %d by B t (<= %d) %s; plant best combo %s" % (rank, len(rows), RANK_MAX, rankable,
        [(x["h"], x["state"], x["dir"], round(x["tB"], 2)) for x in rows if x["plant"]]))
    rep = dict(schema="SEL1_LAUNCH_CHECK_V1", alpha=alpha, target_bpd=TARGET_BPD, gain_bpd=g1, alpha_ok=ok_alpha,
               visible=dict(rule="plant book (plant formula, H72, all, long) B-half monthly cost0 t >= 2 in the pc world",
                            tB=tB_plant_fixed, B=S.summary(rb), passed=visible),
               rankable=dict(rule="plant (best A combo of 60) B-half t rank <= 10 among the real candidate pool in the pc world",
                             rank=rank, pool=len(rows) - 1, invalid_in_v4=bad, top15=rows[:15],
                             plant=[x for x in rows if x["plant"]][0], passed=rankable),
               passed=bool(ok_alpha and visible and rankable), conventions=__doc__, dev_read=False, oos_read=False,
               seconds=round(time.time() - t0, 1), written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=1, default=float)
    log("launch check -> %s (%.0fs)" % (rep["passed"], time.time() - t0))
    sys.exit(0 if rep["passed"] else 7)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
