"""BASIN5 debug-round report with the hall-of-fame freeze (merged plan 10-13; ADR-0009).

    python b5_round_dbg.py <RUN_NAME> --mode p8|p16|p32|null|real [--workers 20] [--hof-n 150]

 curve    tv1_round.curve unchanged (T1 batch mean corr with s, 4 fixed TRAIN tiles, every 16
          updates); plant / null worlds only.
 HoF      per family: every valid formula of FORMULAS.jsonl, grouped by body; the top --hof-n by
          mean logged reward (gross_t on that update's 16 drawn tiles) plus every body logged >= 5
          times, plus the final-checkpoint pool (256 at seed 20261108, the masked sampler,
          distinct). All re-scored on the 50 fixed TRAIN tiles of the run's world (b5_fixed, the
          b5_round.score_both path), ranked by the step-1 G2 ruler in B5_GATE1\\DECISION.json
          (hof_rank: conc_k10_net = conc K10 TRAIN net mean, or full_gross_t = full-pool gross
          mean / sd over the 50 tiles), dedup on the identical 50-tile vector of that ruler, top 5.
 old      the same ruler on the final-checkpoint pool only (the ADR-0008 freeze), for attribution.
 found    (plant worlds) T1 batch mean corr > 0.3 at a logged update AND a HoF top-5 formula of any
          family with 50-tile corr > 0.7 (ADR-0009). Exit 0 = found, 10 = not found; real -> 0.
Writes ROUND_REPORT.json and HOF_RANKED.json (top 30 per family, for b5_verdict_dbg). TRAIN
only; no DEV, no OOS. Every report states the sensitivity floor of the plant used.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b5_fixed as F

DECISION = os.path.join(F.A_ROOT, "B5_GATE1", "DECISION.json")
FLOOR = {"p8": "p8 = +8 bp/day (about the 1-week momentum baseline)",
         "p16": "p16 = +16 bp/day (about 2x the 1-week momentum baseline)",
         "p32": "p32 = +32 bp/day, about 4x the 1-week momentum baseline gross (~7.9 bp/day)"}


def log(m):
    print(m, flush=True)


def ruler_value(rec, world, hof_rank):
    x = rec["w"][world]
    if hof_rank == "conc_k10_net":
        return float(np.mean(x["cn"])), x["cn"]
    return F.gross_t(x["fg"]), x["fg"]


def rank_pool(bodies, byb, world, hof_rank, n=30):
    rows = []
    for b in bodies:
        r = byb.get(b)
        if r is None or not r["ok"]:
            continue
        v, vec = ruler_value(r, world, hof_rank)
        if np.isfinite(v):
            rows.append((v, b, vec, r))
    rows.sort(key=lambda z: -z[0])
    kept = []
    for z in rows:
        if any(np.array_equal(z[2], k[2]) for k in kept):
            continue
        kept.append(z)
        if len(kept) == n:
            break
    return kept, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--mode", required=True, choices=("p8", "p16", "p32", "null", "real"))
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--hof-n", type=int, default=150)
    ap.add_argument("--freq-min", type=int, default=5)
    ap.add_argument("--out", default="ROUND_REPORT.json")
    ap.add_argument("--final-n", type=int, default=0, help="pipeline test only: cap the final pool per family")
    ap.add_argument("--hof-rank", default=None, choices=("conc_k10_net", "full_gross_t"),
                    help="pipeline test only: skip DECISION.json")
    a = ap.parse_args()
    t_all = time.time()
    run_dir = os.path.join(F.A_ROOT, a.run)
    done = [f for f in ("TRAIN_RECEIPT_V1.json", "STOPPED.json") if os.path.exists(os.path.join(run_dir, f))]
    if not done:
        raise SystemExit(f"{a.run}: no receipt and no STOPPED.json")
    if a.hof_rank:
        dec, hof_rank = dict(test=True), a.hof_rank
    else:
        dec = json.load(open(DECISION, encoding="utf-8"))
        if dec.get("limit") != 0 or not dec.get("proceed"):
            raise SystemExit("step-1 DECISION.json is not a full proceed decision")
        hof_rank = dec["hof_rank"]
    import torch
    torch.set_num_threads(8)
    TR, M, D, W, R = F._imports()
    import m2h_b5_trainer_dbg as TD
    import m2h_b5_grammar as G
    import tv1_round as TV
    import b5_explain as X
    man, arr = F.load_cache()
    vm = F.make_vm()
    world = a.mode
    univ = np.array(arr["univ"])
    s = np.array(arr["s"])
    plant = world in ("p8", "p16", "p32", "null")

    log_rows = [json.loads(l) for l in open(os.path.join(run_dir, "BATCH_LOG.jsonl"), encoding="utf-8") if l.strip()]
    last = int(log_rows[-1]["update"])
    rows = TV.read_formulas(os.path.join(run_dir, "FORMULAS.jsonl"))
    blk = TV.blocks("dense", log_rows, rows)
    pts = sorted(set(range(0, last + 1, 16)) | {last})
    curve_rows = {u: [tuple(TD.strip_hold(vm, list(r["tokens"]))) for r in rows
                      if r["update"] == u and r["basin"] == "T1" and r["valid"]] for u in pts}
    del rows

    # ---- hall-of-fame candidates per family
    agg = {f: collections.defaultdict(lambda: dict(gt=[], n=0)) for f in M.BASIN_NAMES}
    for line in open(os.path.join(run_dir, "FORMULAS.jsonl"), encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("invalid_reason") is not None or r.get("gross_t") is None:
            continue
        body = tuple(TD.strip_hold(vm, list(r["tokens"])))
        x = agg[r["basin"]][body]
        x["gt"].append(r["gross_t"])
        x["n"] += 1
    model, ck_name, ck_upd = TV.load_final(TD, run_dir, torch.device("cpu"))
    fpool = TV.final_pool(TD, model, vm, torch.device("cpu"), TV.PER_FAMILY)
    del model
    cand, src, stats = {}, collections.defaultdict(set), {}
    for f in M.BASIN_NAMES:
        g = agg[f]
        by_r = sorted(g, key=lambda b: -np.mean(g[b]["gt"]))
        sel = list(by_r[:a.hof_n]) + [b for b in g if g[b]["n"] >= a.freq_min]
        for b in sel:
            src[(f, b)].add("logged")
        fin = [tuple(int(t) for t in body) for body, hold in fpool[f]][: (a.final_n or None)]
        for b in fin:
            src[(f, b)].add("final")
        allb = list(dict.fromkeys(sel + fin))
        legal = [b for b in allb if G.check(vm, list(b)) is None]
        cand[f] = legal
        stats[f] = dict(logged_distinct=len(g), hof_logged=len(dict.fromkeys(sel)), final_distinct=len(fin),
                        grammar_illegal=len(allb) - len(legal), candidates=len(legal))
    tasks, tid = [], {}
    for f in M.BASIN_NAMES:
        for b in cand[f]:
            if b not in tid:
                tid[b] = "T%d" % len(tasks)
                tasks.append(dict(id=tid[b], body=list(b), worlds=[world], conc=True))
    cid = {}
    if plant:                                 # tv1_round.curve, same 4 tiles, run in the pool
        for u in pts:
            for b in curve_rows[u]:
                if b not in cid:
                    cid[b] = "C%d" % len(cid)
                    tasks.append(dict(id=cid[b], body=list(b), worlds=[world], corr_only=True,
                                      tiles=list(TV.CURVE_TILES)))
    log("HoF candidates %s, curve bodies %d, tasks %d" % ({f: stats[f]["candidates"] for f in M.BASIN_NAMES},
                                                          len(cid), len(tasks)))
    res = F.run_pool(tasks, workers=a.workers, log=log)
    byb = {b: res[i] for b, i in tid.items()}
    cur = None
    if plant:
        cur = []
        for u in pts:
            cs = np.asarray([float(np.mean(res[cid[b]]["corr"])) for b in curve_rows[u] if res[cid[b]]["ok"]])
            cur.append(dict(update=u, n=int(len(cs)), corr_mean=TV.fmean(cs),
                            corr_max=round(float(cs.max()), 4) if len(cs) else None,
                            frac_gt_07=round(float((cs > 0.7).mean()), 4) if len(cs) else None))
        log("curve %d points, T1 max %s" % (len(cur), max((c["corr_mean"] or -9) for c in cur)))

    def row(f, i, v, b, r):
        x = r["w"][world]
        g = agg[f].get(b)
        return dict(rank=i + 1, body=list(b), expr=vm.describe(list(b), list(M.TERMINALS)), n_tokens=len(b) - 1,
                    level_flag=bool(D.level_flag(vm, list(b))), explain=X.explain(vm, list(b)),
                    src=sorted(src[(f, b)]), n_logged=g["n"] if g else 0,
                    logged_gross_t=round(float(np.mean(g["gt"])), 4) if g else None,
                    ruler_value=round(v, 6), conc_k10_train_net_bpd=round(F.bpd(np.mean(x["cn"])), 3),
                    conc_k10_train_gross_bpd=round(F.bpd(np.mean(x["cg"])), 3),
                    conc_turnover_per_day=round(float(np.mean(x["cturn"])), 3),
                    full_train_gross_bpd=round(F.bpd(np.mean(x["fg"])), 3),
                    full_train_net_bpd=round(F.bpd(np.mean(x["fn"])), 3), full_gross_t50=round(F.gross_t(x["fg"]), 4),
                    corr_s=round(float(np.mean(r["corr"])), 4) if plant else None)
    hof, hof30, old = {}, {}, {}
    for f in M.BASIN_NAMES:
        kept, n_ok = rank_pool(cand[f], byb, world, hof_rank, 30)
        hof30[f] = [row(f, i, v, b, r) for i, (v, b, _, r) in enumerate(kept)]
        hof[f] = hof30[f][:5]
        stats[f]["scored_ok"] = n_ok
        fin = [b for b in cand[f] if "final" in src[(f, b)]]
        ko, _ = rank_pool(fin, byb, world, hof_rank, 5)
        old[f] = [row(f, i, v, b, r) for i, (v, b, _, r) in enumerate(ko)]
        log("  %s HoF top5 %s" % (f, [(h["expr"][:40], h["corr_s"]) for h in hof[f]]))
    found, t1_max, hof_max, old_max = None, None, None, None
    if plant:
        t1_max = max((c["corr_mean"] for c in cur if c["corr_mean"] is not None), default=None)
        hof_max = max((h["corr_s"] for fs in hof.values() for h in fs if h["corr_s"] is not None), default=None)
        old_max = max((h["corr_s"] for fs in old.values() for h in fs if h["corr_s"] is not None), default=None)
        found = bool(t1_max is not None and t1_max > 0.3 and hof_max is not None and hof_max > 0.7)
    prog_p = os.path.join(run_dir, "PROGRESS_16.jsonl")
    prog = [json.loads(l) for l in open(prog_p, encoding="utf-8") if l.strip()] if os.path.exists(prog_p) else []
    modes = {f: (prog[-1]["families"][f].get("mode") if prog else None) for f in M.BASIN_NAMES}
    converged = dict(final_policy_mode={f: dict(expr=(m or {}).get("expr"), share=(m or {}).get("share"))
                                        for f, m in modes.items()},
                     hof_top5={f: [dict(rank=h["rank"], expr=h["expr"], explain=h["explain"], corr_s=h["corr_s"],
                                        conc_k10_train_net_bpd=h["conc_k10_train_net_bpd"]) for h in fs]
                               for f, fs in hof.items()})
    blob = json.dumps(hof, sort_keys=True).encode()
    rep = dict(schema="B5_ROUND_REPORT_DBG_V1", run=a.run, trainer="m2h_b5_trainer_dbg", mode=a.mode,
               ended_by=done, last_update=last, checkpoint=ck_name, checkpoint_update=ck_upd,
               sensitivity_floor=FLOOR.get(world), hof_rank=hof_rank, gate_decision=dec,
               found=found, t1_batch_corr_max=t1_max, hof_corr_max=hof_max, old_freeze_corr_max=old_max,
               rule="found = T1 batch mean corr with s > 0.3 at a logged update AND a hall-of-fame top-5 formula "
                    "(any family, 50 fixed TRAIN tiles, ruler %s) with corr > 0.7 (ADR-0009)" % hof_rank,
               收敛到了什么=converged, hof=hof, hof_sha256=hashlib.sha256(blob).hexdigest(), old_freeze=old,
               hof_stats=stats, hof_rule=dict(logged_top=a.hof_n, freq_min=a.freq_min, final_seed=TV.FINAL_SEED,
                                              final_per_family=TV.PER_FAMILY, tiles=50, world=world,
                                              rank=hof_rank, dedup="identical 50-tile ruler vector"),
               curve=cur, curve_tiles=[man["starts"][i] for i in TV.CURVE_TILES], blocks=blk, progress_16=prog,
               seconds=round(time.time() - t_all, 1), written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               dev_read=False, oos_read=False)
    json.dump(rep, open(os.path.join(run_dir, a.out), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    json.dump(dict(schema="B5_HOF_RANKED_V1", run=a.run, world=world, hof_rank=hof_rank, families=hof30,
                   written=rep["written"]), open(os.path.join(run_dir, "HOF_RANKED.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    log("ROUND %s found=%s t1corr_max=%s hof_corr_max=%s old_freeze_corr_max=%s rank=%s %.0fs"
        % (a.run, found, t1_max, hof_max, old_max, hof_rank, time.time() - t_all))
    return 0 if (found or not plant) else 10


if __name__ == "__main__":
    sys.exit(main())
