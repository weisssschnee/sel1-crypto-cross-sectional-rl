"""CR1 round (order V4_COMBO_V1_CR1 s3): freeze on the B half. TRAIN (V4 grid) only; no DEV, no OOS.

  candidates  HOF.json of the run (A half only), top 50 by mean A score, plus the plant formula as a
              reference row (not a candidate).
  ranking     each candidate re-scored on all 30 B tiles in the run's own world: marginal K10 net and full
              net (flat6) of combo + f over combo, corr_s with s_perp on every B tile. Frozen = top 10 by B
              mean marginal net in CR1_ALPHA.json's rank_book (K10 = ADR-0009's conc_k10_net if the ruler
              check passes on K10, else full); bar, found and candidates use the same key.
  modes       pc    -> ROUND_PC.json; found_corr = any frozen with corr_s > 0.7. rc 0 corr found, 20 not.
              null  -> ROUND_NULL.json; bar = max B K10 over the NULL frozen set. Then the PC verdict:
                       found = a PC frozen formula with corr_s > 0.7 and B K10 > bar. rc 0 found, 21 not.
              real  -> ROUND_REAL.json + FROZEN_CR1.json (frozen formulas with B K10 > bar). rc 0 any, 22 none.
    python -I cr1_round.py RUN --mode pc|null|real [--pc RUN] [--null RUN]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cr1_common as Q
import numpy as np
import torch
import m2h_common as M
import m2h_dense_common as D
import v4_world as X
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid

N_CAND, N_FROZEN = 50, 10
RK = "b_k10"                     # set from CR1_ALPHA.json rank_book (declared before the run)


def log(m):
    print(m, flush=True)


def jload(run, name):
    return json.load(open(os.path.join(Q.A_ROOT, run, name), encoding="utf-8"))


def score_b(ctx, vm, body, sl, device):
    try:
        r = Q.marginal(ctx, vm, list(body), ctx["B"], sl, books=(X.K10, ("full",)), corr_tiles=len(ctx["B"]))
    except (M.ZeroCoverage, CandidateLocalInvalid) as e:
        return dict(invalid=str(e))
    k = [Q.bpd(x) for x in r["books"]["conc_k10"]["net"]]
    f = [Q.bpd(x) for x in r["books"]["full"]["net"]]
    return dict(b_k10=round(float(np.mean(k)), 4), b_k10_t=round(Q.tstat(k), 2), b_k10_pos=int(np.sum(np.array(k) > 0)),
                b_full=round(float(np.mean(f)), 4), b_full_t=round(Q.tstat(f), 2), corr_s_b=round(r["corr"], 4),
                turn_k10=round(float(np.mean(r["books"]["conc_k10"]["turn"])), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--mode", required=True, choices=("pc", "null", "real"))
    ap.add_argument("--pc", default=None)
    ap.add_argument("--null", default=None)
    args = ap.parse_args()
    global RK
    cal = json.load(open(Q.ALPHA_JSON, encoding="utf-8"))
    if not cal.get("ruler_ok"):
        raise SystemExit("ruler check failed in CR1_ALPHA.json; no round")
    RK = dict(conc_k10="b_k10", full="b_full")[cal["rank_book"]]
    out_name = dict(pc="ROUND_PC.json", null="ROUND_NULL.json", real="ROUND_REAL.json")[args.mode]
    out_path = os.path.join(Q.A_ROOT, args.run, out_name)
    if os.path.exists(out_path):
        raise SystemExit("%s exists" % out_path)
    t0 = time.time()
    rec = jload(args.run, "TRAIN_RECEIPT_V1.json")
    world = rec["mode"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vm = Q.make_vm()
    ctx = Q.build(world, dev, vm, log=log)
    sl = M.tile_slices(ctx["feats_pad"], ctx["B"], dev)
    hof = jload(args.run, "HOF.json")["entries"][:N_CAND]
    rows = []
    for e in hof:
        r = score_b(ctx, vm, e["body"], sl, dev)
        r.update(expr=e["expr"], basin=e["basin"], body=e["body"], a_mean_full=e["mean_bpd"], a_n=e["n"])
        rows.append(r)
    ok = [r for r in rows if "invalid" not in r]
    ok.sort(key=lambda r: -r[RK])
    frozen = ok[:N_FROZEN]
    ref = score_b(ctx, vm, D.plant_body(vm), sl, dev)
    out = dict(schema="CR1_ROUND_V1", run=args.run, world=world, rule=__doc__, rank_key=RK, n_candidates=len(rows),
               n_invalid=len(rows) - len(ok), frozen=frozen, candidates=ok, plant_reference=ref,
               combo_b_k10=round(float(np.mean([ctx["combo"][s]["conc_k10"]["net"] for s in ctx["B"]])), 4),
               combo_b_full=round(float(np.mean([ctx["combo"][s]["full"]["net"] for s in ctx["B"]])), 4),
               dev_read=False, oos_read=False)
    rc = 0
    if args.mode == "pc":
        hit = [r for r in frozen if r["corr_s_b"] > Q.PLANT_CORR_MIN]
        out.update(found_corr=bool(hit), corr_hits=hit)
        rc = 0 if hit else 20
        log("ROUND PC frozen %d, corr>0.7 hits %d, top %s" % (len(frozen), len(hit), [(r["expr"][:60], r["b_k10"], r["corr_s_b"]) for r in frozen[:3]]))
    elif args.mode == "null":
        bar = max(r[RK] for r in frozen) if frozen else float("inf")
        pcr = jload(args.pc, "ROUND_PC.json")
        hit = [r for r in pcr["frozen"] if r["corr_s_b"] > Q.PLANT_CORR_MIN and r[RK] > bar]
        out.update(null_bar_b_k10=bar, pc_run=args.pc, pc_found=bool(hit), pc_hits=hit)
        rc = 0 if hit else 21
        log("ROUND NULL bar %.3f; PC found %s (%d hits) %s" % (bar, bool(hit), len(hit), [(r["expr"][:60], r["b_k10"]) for r in hit[:3]]))
    else:
        nul = jload(args.null, "ROUND_NULL.json")
        bar = nul["null_bar_b_k10"]
        cand = [r for r in frozen if r[RK] > bar]
        out.update(null_bar_b_k10=bar, null_run=args.null, above_bar=cand)
        if cand:
            blob = json.dumps([r["body"] for r in cand], sort_keys=True).encode()
            fz = dict(schema="FROZEN_CR1", contract="XS_BASIN_TIME_CONTRACT_V4_20261013", run=args.run,
                      null_bar_b_k10=bar, formulas=cand, combo_fit=ctx["fit"], s_perp_sha16=ctx["s_perp_sha16"],
                      sha256=hashlib.sha256(blob).hexdigest(), dev_read=False, oos_read=False,
                      written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            fp = os.path.join(Q.A_ROOT, args.run, "FROZEN_CR1.json")
            if os.path.exists(fp):
                raise SystemExit("%s exists" % fp)
            json.dump(fz, open(fp, "w", encoding="utf-8"), indent=1)
        rc = 0 if cand else 22
        log("ROUND REAL bar %.3f; above %d %s" % (bar, len(cand), [(r["expr"][:60], r["b_k10"], r["b_k10_t"]) for r in cand[:5]]))
    out.update(seconds=round(time.time() - t0, 1), written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1)
    sys.exit(rc)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
