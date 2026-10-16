"""XS_SEL1 combo-sweep check (supplement 10-15 morning). GPU VM + CPU book, V4 TRAIN only, no DEV, no OOS. New file.

World: the PC world of the chain (pcnull, the chosen plant cell's alpha, K, DD_MAX[K] from SEL1_LAUNCH_MANIFEST).
Bodies (>= 50): the plant formula, the old SEL1_PC hall top 10, then the sel1_calib pool (CR1_PC_P8 top 50 + B5 ranked)
until 60 bodies. Selections through S.vm_select_gpu (the trainer's --gpu-select path).
  old   per combo S.reward(S.score_sel(...), t_dd, DD_MAX) for all 60 combos, pick = sel1_nullplant's rule
  new   sel1_sweep.sweep
Pass: max |old - new| over all 60 scores of every body <= 1e-9, the same picked combo for every body, the chosen
combo's S.score_sel score == the old best exactly, and the plant picks 4/all/both with t_A == SEL1_NULLPLANT's score_A.
-> SEL1_SWEEPCHECK.json, rc 0 pass / 7 fail.
    python -I sel1_sweepcheck.py --old-pc SEL1_PC_V1_COMBOTOKEN
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import sel1_common as S
import sel1_sweep as W
import sel1_fastts as F
import sel1_calib as C
import cr1_common as Q
import m2h_common as M
import m2h_dense_common as D
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid

N_BODIES, TOL = 60, 1e-9


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-pc", required=True)
    a = ap.parse_args()
    t0 = time.time()
    man = json.load(open(os.path.join(S.OUT_ROOT, "SEL1_LAUNCH_MANIFEST.json"), encoding="utf-8"))
    ch = man["plant_check"]["chosen"]
    k, alpha, dd = int(ch["K"]), float(ch["alpha"]), float(man["reward"]["dd_max_used"])
    np_ = json.load(open(os.path.join(S.OUT_ROOT, "SEL1_NULLPLANT.json"), encoding="utf-8"))
    cell = next(r for r in np_["table"] if r["K"] == k and r["strength_bpd"] == ch["strength_bpd"])
    S.K = k
    vm = Q.make_vm()
    F.install()
    ctx = S.build("pcnull", torch.device("cuda"), vm, alpha=alpha, log=log)
    pb = tuple(D.plant_body(vm))
    bodies = [pb]
    old_hof = json.load(open(os.path.join(S.OUT_ROOT, a.old_pc, "HOF.json"), encoding="utf-8"))["entries"]
    bodies += [tuple(e["body"]) for e in old_hof[:10]]
    pool = [tuple(e["body"]) for e in json.load(open(C.CR1_HOF, encoding="utf-8"))["entries"][:50]]
    for fam in json.load(open(C.B5_HOF, encoding="utf-8"))["families"].values():
        pool += [tuple(e["body"]) for e in fam]
    bodies = list(dict.fromkeys(bodies + pool))
    rows, bad, t_old, t_new = [], {}, 0.0, 0.0
    for b in bodies:
        if len(rows) >= N_BODIES:
            break
        try:
            sels = S.vm_select_gpu(ctx, vm, b)
        except (S.Invalid, CandidateLocalInvalid, M.ZeroCoverage) as e:
            bad[vm.describe(list(b), list(M.TERMINALS))] = str(e)[:60]
            continue
        t_ = time.time()
        old = {c: S.reward(S.score_sel(sels, ctx, *c), "t_dd", dd) for c in W.COMBOS}
        po = W.pick(old)
        t_old += time.time() - t_
        t_ = time.time()
        r = W.sweep(sels, ctx, "t_dd", dd)
        t_new += time.time() - t_
        new = dict(zip(W.COMBOS, r[4])) if r else {}
        diffs = [abs(old[c] - new[c]) for c in W.COMBOS if np.isfinite(old[c]) or np.isfinite(new.get(c, np.nan))]
        same_nan = all(np.isfinite(old[c]) == np.isfinite(new.get(c, np.nan)) for c in W.COMBOS)
        rows.append(dict(expr=vm.describe(list(b), list(M.TERMINALS)), plant=b == pb,
                         max_abs_diff=float(max(diffs)) if diffs else 0.0, nan_pattern_same=bool(same_nan),
                         pick_old=None if po is None else list(po[1]), pick_new=None if r is None else list(r[0]),
                         best_old=None if po is None else po[0], best_rescored=None if r is None else r[2],
                         rescore_exact=bool(po is not None and r is not None and r[2] == po[0])))
    pr = next(x for x in rows if x["plant"])
    mx = max(x["max_abs_diff"] for x in rows)
    ok = dict(n_bodies=len(rows) >= 50, max_abs_diff=mx <= TOL, nan_pattern=all(x["nan_pattern_same"] for x in rows),
              pick_same=all(x["pick_old"] == x["pick_new"] for x in rows),
              rescore_exact=all(x["rescore_exact"] for x in rows if x["pick_old"] is not None),
              plant_combo=pr["pick_new"] == [4, 0, 0], plant_t_A=pr["best_rescored"] == cell["plant"]["score_A"])
    rep = dict(schema="SEL1_SWEEPCHECK_V1", world=dict(mode="pcnull", alpha=alpha, K=k, dd_max=dd), tol=TOL,
               bodies=len(rows), invalid=bad, checks=ok, passed=all(ok.values()), max_abs_diff=mx,
               plant=dict(pr, nullplant_score_A=cell["plant"]["score_A"]),
               seconds_per_body=dict(old_60_score_sel=round(t_old / len(rows), 3), sweep=round(t_new / len(rows), 3)),
               rows=rows, modules=dict(sel1_sweep=S.sha16f(W.__file__), sel1_common=S.sha16f(S.__file__)),
               dev_read=False, oos_read=False, seconds=round(time.time() - t0, 1),
               written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(rep, open(os.path.join(S.OUT_ROOT, "SEL1_SWEEPCHECK.json"), "w", encoding="utf-8"), indent=1,
              default=float, ensure_ascii=False)
    log("sweepcheck %d bodies: max|d| %.3g pick_same %s rescore_exact %s plant %s t_A %r (nullplant %r) | s/body old %.2f "
        "sweep %.2f -> %s" % (len(rows), mx, ok["pick_same"], ok["rescore_exact"], pr["pick_new"], pr["best_rescored"],
                              cell["plant"]["score_A"], rep["seconds_per_body"]["old_60_score_sel"],
                              rep["seconds_per_body"]["sweep"], "PASS" if rep["passed"] else "FAIL %s" % ok))
    sys.exit(0 if rep["passed"] else 7)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
