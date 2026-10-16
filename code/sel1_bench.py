"""XS_SEL1 s3.1 step timing + production-path selftests (CPU + GPU, V4 TRAIN only, no DEV, no OOS). New file.

  timing   per formula: VM on the whole grid (GPU, synchronised), mask/coverage, and one book score on the A half
           for a random (hold, state, direction); 40 bodies from the CR1 PC hall of fame + the plant body.
  test 1   apply_stateful_action_policy_batch over the month windows == m1_ruler.held_path per window (exact).
  test 2   single coin, single hold through the production book layer (held_months -> settle_window on the
           month window) == the ENR1b event settlement (m1_ruler.settle_multi on [coin, BTC], rows t+1..t+h+1).
    python -I sel1_bench.py
"""
from __future__ import annotations

import json
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
import m1_ruler as R
from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod

OUT = os.path.join(S.OUT_ROOT, "SEL1_BENCH.json")
N_BODIES = int(os.environ.get("SEL1_BENCH_N", "8"))
CR1_HOF = os.path.join(S.A_ROOT, "CR1_PC_P8", "HOF.json")


def main():
    t0 = time.time()
    device = torch.device("cuda")
    vm = Q.make_vm()
    ctx = S.build("real", device, vm, log=lambda m: print(m, flush=True))
    t_build = time.time() - t0
    bodies = [tuple(e["body"]) for e in json.load(open(CR1_HOF, encoding="utf-8"))["entries"][:N_BODIES]]
    import m2h_dense_common as D
    bodies.append(tuple(D.plant_body(vm)))
    rng = np.random.default_rng(7)
    tv, tm, tsc, recs, sigs = [], [], [], [], {}
    for b in bodies:
        torch.cuda.synchronize()
        a = time.time()
        raw = S.raw_signal(ctx, vm, b)
        torch.cuda.synchronize()
        tv.append(time.time() - a)
        a = time.time()
        try:
            sig = S.mask_signal(ctx, raw)
        except S.Invalid as e:
            tm.append(time.time() - a)
            recs.append(dict(expr=vm.describe(list(b), list(S.M.TERMINALS)), invalid=str(e)))
            continue
        tm.append(time.time() - a)
        h, st, dr = int(rng.choice(S.HOLDS)), int(rng.integers(5)), int(rng.integers(3))
        a = time.time()
        ms = [m for m in ctx["months"] if m["half"] == "A"]
        des = [S.desired_window(sig, ctx, m["m0"], m["m1"], st, dr) for m in ms]
        t_des = time.time() - a
        hm = S.held_months(des, h)
        t_pol = time.time() - a - t_des
        rows = S.score(sig, ctx, h, st, dr)
        tsc.append(time.time() - a - t_des - t_pol)
        print("body %d vm %.2fs mask %.2fs desired %.2fs policy %.2fs score %.2fs cols %d"
              % (len(recs), tv[-1], tm[-1], t_des, t_pol, tsc[-1], max(c.size for _, c in hm)), flush=True)
        sigs[b] = sig
        recs.append(dict(expr=vm.describe(list(b), list(S.M.TERMINALS))[:90], h=h, st=S.STATES[st], dr=S.DIRS[dr],
                         A=S.summary(rows)))
    vram = torch.cuda.max_memory_allocated() / 2 ** 30
    # test 1: batch policy == held_path, window by window
    d1, n1 = 0.0, 0
    for b in list(sigs)[:6]:
        for h in S.HOLDS:
            ms = [m for m in ctx["months"] if m["half"] == "A"]
            des = [S.desired_window(sigs[b], ctx, m["m0"], m["m1"], 0, 0) for m in ms]
            for (held, cols), dd in zip(S.held_months(des, h), des):
                if cols.size:
                    pad = np.zeros((dd.shape[0], S.width(max(c.size for _, c in S.held_months(des, h)))))
                    pad[:, :cols.size] = dd[:, cols]
                    ref = R.held_path(pad, h - 1, R.BAND, policy_mod)[:, :cols.size]
                    d1 = max(d1, float(np.abs(held - ref).max()))
                n1 += 1
    # test 2: single coin book == ENR1b event
    sh, bi, T = ctx["shard"], ctx["bi"], len(ctx["ts"])
    specs = {c: R.cost_specs(R.NAV_USD)[c] for c in ("flat6", "cost0")}
    ev, tries = [], 0
    while len(ev) < 12 and tries < 5000:
        tries += 1
        m = ctx["months"][int(rng.integers(len(ctx["months"])))]
        h = int(rng.choice(S.HOLDS))
        t = int(rng.integers(m["m0"], m["m1"] - h - 2))
        a = int(rng.integers(len(ctx["syms"])))
        if not (ctx["mask"][a, t] and np.isfinite(sh["fwd_wx"][a, t + 1]) and np.isfinite(sh["fwd_wx"][a, t + h + 1])
                and ctx["Larr"][a] > t + h + 1):
            continue
        d = 1.0 if len(ev) % 2 == 0 else -1.0
        des = np.zeros((m["m1"] - m["m0"], len(ctx["syms"])))
        des[t + 1 - m["m0"]:t + 1 + h - m["m0"], a] = d
        (held, cols), = S.held_months([des], h)
        r = S.settle_window(held, cols, ctx, m["m0"], m["m1"])
        b_ = float(ctx["beta"][a, t])
        rows = np.arange(t + 1, t + h + 2)
        hv = np.zeros((h + 1, 2))
        hv[:h, 0], hv[:h, 1] = d, -d * b_
        f = sh["fwd_wx"][[a, bi]][:, rows].T
        res, _ = R.settle_multi(hv, f, sh["fi"][[a, bi]][:, rows].T, sh["fr"][[a, bi]][:, rows].T, specs)
        ev.append(dict(coin=ctx["syms"][a], t=str(ctx["ts"][t]), h=h, d=d,
                       diff_net=abs(r["net"] - res["flat6"]["log_nav"]), diff_cost0=abs(r["cost0"] - res["cost0"]["log_nav"]),
                       net_bp=round(1e4 * r["net"], 3)))
    d2 = max(max(e["diff_net"], e["diff_cost0"]) for e in ev)
    rep = dict(schema="SEL1_BENCH_V1", build_s=round(t_build, 1), n_bodies=len(bodies), n_valid=len(sigs),
               vm_s=dict(mean=round(float(np.mean(tv)), 4), p90=round(float(np.quantile(tv, 0.9)), 4), max=round(max(tv), 4)),
               mask_s=round(float(np.mean(tm)), 4),
               score_A_s=dict(mean=round(float(np.mean(tsc)), 4), p90=round(float(np.quantile(tsc, 0.9)), 4)),
               vram_gib=round(vram, 2),
               test1=dict(windows=n1, max_abs_diff=d1, ok=bool(d1 == 0.0)),
               test2=dict(events=ev, max_abs_diff=d2, ok=bool(len(ev) == 12 and d2 <= 1e-12)),
               records=recs, dev_read=False, oos_read=False, seconds=round(time.time() - t0, 1))
    os.makedirs(S.OUT_ROOT, exist_ok=True)
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=1, default=float)
    print("vm %.3fs mask %.3fs score %.3fs vram %.2f | test1 %s (%d, %.1e) test2 %s (%.1e) | %.0fs"
          % (rep["vm_s"]["mean"], rep["mask_s"], rep["score_A_s"]["mean"], vram, rep["test1"]["ok"], n1, d1,
             rep["test2"]["ok"], d2, time.time() - t0), flush=True)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
