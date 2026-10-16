"""CR1b round (QuantMind 10-14 CR1b approval, point 3): freeze on the B half. V4 TRAIN only; no DEV, no OOS.

  candidates  HOF.json of the run (A half only), top 50 by mean A score; the plant formula is scored as a
              reference row (not a candidate), and its rank among the candidates is reported.
  ranking     each candidate re-scored on all 30 B tiles in the run's own world: marginal K10 / K20 / full
              net (flat6) of combo + r_f over combo, corr_s of r_f with s_perp8 on every B tile. Frozen = top
              10 by B mean marginal net on CR1B_ALPHA.json's rank_book (a concentrated book, declared before
              launch); bar, found and candidates use the same key. Full is reported only.
  modes       pc    -> ROUND_PC.json; found_corr = any frozen with corr_s > 0.7. rc 0 corr found, 20 not.
              null  -> ROUND_NULL.json; bar = max B rank_book net over the NULL frozen set. Then the PC verdict:
                       found = a PC frozen formula with corr_s > 0.7 and B rank_book net > bar. rc 0, 21 not.
              real  -> ROUND_REAL.json + FROZEN_CR1B.json (frozen formulas above the bar). rc 0 any, 22 none.
              Every mode also reports what the run converged to (the policy's mode per family in the last
              16 updates) and where the bar sits.
    python -I cr1b_round.py RUN --mode pc|null|real [--pc RUN] [--null RUN]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cr1b_common as Q
import numpy as np
import torch
import m2h_common as M
import m2h_dense_common as D
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid

N_CAND, N_FROZEN = 50, 10
BOOK_KEYS = dict(conc_k10="k10", conc_k20="k20", full="full")


def log(m):
    print(m, flush=True)


def jload(run, name):
    return json.load(open(os.path.join(Q.A_ROOT, run, name), encoding="utf-8"))


def score_b(ctx, vm, body, sl):
    try:
        r = Q.marginal(ctx, vm, list(body), ctx["B"], sl, books=Q.BOOKS3, corr_tiles=len(ctx["B"]))
    except (M.ZeroCoverage, CandidateLocalInvalid) as e:
        return dict(invalid=str(e))
    out = dict(corr_s_b=round(r["corr"], 4), corr_raw_b=round(r["corr_raw"], 4))
    for b, k in BOOK_KEYS.items():
        v = [Q.bpd(x) for x in r["books"][b]["net"]]
        out["b_" + k] = round(float(np.mean(v)), 4)
        out["b_%s_t" % k] = round(Q.tstat(v), 2)
        out["b_%s_pos" % k] = int(np.sum(np.array(v) > 0))
        out["turn_" + k] = round(float(np.mean(r["books"][b]["turn"])), 4)
    return out


def converged(run):
    """Policy mode per family over the last 16 updates (BATCH_LOG families.mode)."""
    rows = [json.loads(x) for x in open(os.path.join(Q.A_ROOT, run, "BATCH_LOG.jsonl"), encoding="utf-8")
            if x.strip()][-16:]
    out = {}
    for b in rows[-1]["families"]:
        c = collections.Counter(r["families"][b]["mode"]["expr"] for r in rows)
        expr, n = c.most_common(1)[0]
        out[b] = dict(mode_expr=expr, updates_as_mode=n, share_last=rows[-1]["families"][b]["mode"]["share"],
                      entropy_last=rows[-1]["families"][b]["entropy"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--mode", required=True, choices=("pc", "null", "real"))
    ap.add_argument("--pc", default=None)
    ap.add_argument("--null", default=None)
    args = ap.parse_args()
    warnings.simplefilter("ignore", RuntimeWarning)
    cal = json.load(open(Q.ALPHA_JSON, encoding="utf-8"))
    if cal.get("rc") != 0 or cal.get("rank_book") not in ("conc_k10", "conc_k20"):
        raise SystemExit("pre-run check failed in CR1B_ALPHA.json; no round")
    rk = "b_" + BOOK_KEYS[cal["rank_book"]]
    out_name = dict(pc="ROUND_PC.json", null="ROUND_NULL.json", real="ROUND_REAL.json")[args.mode]
    out_path = os.path.join(Q.A_ROOT, args.run, out_name)
    if os.path.exists(out_path):
        raise SystemExit("%s exists" % out_path)
    t0 = time.time()
    rec = jload(args.run, "TRAIN_RECEIPT_V1.json")
    if rec.get("rank_book") != cal["rank_book"]:
        raise SystemExit("run rank_book %s != CR1B_ALPHA %s" % (rec.get("rank_book"), cal["rank_book"]))
    world = rec["mode"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vm = Q.make_vm()
    ctx = Q.build(world, dev, vm, log=log)
    sl = M.tile_slices(ctx["feats_pad"], ctx["B"], dev)
    hof = jload(args.run, "HOF.json")["entries"][:N_CAND]
    rows = []
    for e in hof:
        r = score_b(ctx, vm, e["body"], sl)
        r.update(expr=e["expr"], basin=e["basin"], body=e["body"], a_mean_full=e["mean_bpd"], a_n=e["n"])
        rows.append(r)
    ok = [r for r in rows if "invalid" not in r]
    ok.sort(key=lambda r: -r[rk])
    frozen = ok[:N_FROZEN]
    ref = score_b(ctx, vm, D.plant_body(vm), sl)
    if "invalid" not in ref:
        ref["rank_among_candidates"] = 1 + sum(1 for r in ok if r[rk] > ref[rk])
        ref["of"] = len(ok) + 1
    out = dict(schema="CR1B_ROUND_V1", run=args.run, world=world, rule=__doc__, rank_book=cal["rank_book"],
               rank_key=rk, n_candidates=len(rows), n_invalid=len(rows) - len(ok), frozen=frozen, candidates=ok,
               plant_reference=ref, converged=converged(args.run),
               combo_b={b: round(float(np.mean([ctx["combo"][s][b]["net"] for s in ctx["B"]])), 4) for b in BOOK_KEYS},
               dev_read=False, oos_read=False)
    if args.mode == "pc":
        hit = [r for r in frozen if r["corr_s_b"] > Q.PLANT_CORR_MIN]
        out.update(found_corr=bool(hit), corr_hits=hit)
        rc = 0 if hit else 20
        log("ROUND PC (%s) frozen %d, corr>0.7 hits %d, plant rank %s, top %s"
            % (rk, len(frozen), len(hit), ref.get("rank_among_candidates"),
               [(r["expr"][:60], r[rk], r["corr_s_b"]) for r in frozen[:3]]))
    elif args.mode == "null":
        bar = max(r[rk] for r in frozen) if frozen else float("inf")
        pcr = jload(args.pc, "ROUND_PC.json")
        hit = [r for r in pcr["frozen"] if r["corr_s_b"] > Q.PLANT_CORR_MIN and r[rk] > bar]
        pc_best = max((r[rk] for r in pcr["frozen"]), default=None)
        out.update(null_bar=bar, pc_run=args.pc, pc_found=bool(hit), pc_hits=hit,
                   pc_frozen_above_bar=sum(1 for r in pcr["frozen"] if r[rk] > bar), pc_best=pc_best)
        rc = 0 if hit else 21
        log("ROUND NULL (%s) bar %.3f; PC found %s (%d hits; PC best %s) %s"
            % (rk, bar, bool(hit), len(hit), pc_best, [(r["expr"][:60], r[rk]) for r in hit[:3]]))
    else:
        nul = jload(args.null, "ROUND_NULL.json")
        bar = nul["null_bar"]
        cand = [r for r in frozen if r[rk] > bar]
        out.update(null_bar=bar, null_run=args.null, above_bar=cand,
                   bar_rank_among_real_frozen=1 + sum(1 for r in frozen if r[rk] > bar))
        if cand:
            blob = json.dumps([r["body"] for r in cand], sort_keys=True).encode()
            fz = dict(schema="FROZEN_CR1B", contract="XS_BASIN_TIME_CONTRACT_V4_20261013", run=args.run,
                      rank_book=cal["rank_book"], null_bar=bar, formulas=cand, combo_fit=ctx["fit"],
                      s_perp_sha16=ctx["s_perp_sha16"], sha256=hashlib.sha256(blob).hexdigest(),
                      dev_read=False, oos_read=False, written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            fp = os.path.join(Q.A_ROOT, args.run, "FROZEN_CR1B.json")
            if os.path.exists(fp):
                raise SystemExit("%s exists" % fp)
            json.dump(fz, open(fp, "w", encoding="utf-8"), indent=1)
        rc = 0 if cand else 22
        log("ROUND REAL (%s) bar %.3f; above %d %s"
            % (rk, bar, len(cand), [(r["expr"][:60], r[rk], r[rk + "_t"]) for r in cand[:5]]))
    out.update(seconds=round(time.time() - t0, 1), written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=1)
    sys.exit(rc)


if __name__ == "__main__":
    main()
