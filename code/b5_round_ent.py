"""BASIN5_PC_ENT round report (BASIN5_PC_ENT_ORDER s6). b5_round.py's curve + freeze + found rule
verbatim (b5_round.py NOT modified; its score_both / freeze are called with TR = the ENT trainer,
whose sampler / strip_hold / hold_of / OP_BIAS / END_BIAS are byte-identical copies), plus:

 history   column NOT in the verdict: per family, distinct valid formulas of FORMULAS.jsonl ranked
           by the logged TRAIN reward (gross_t on the drawn tiles, mean over appearances); top 20
           rescored on the 50 TRAIN tiles (b5_round.score_both); reported: top 5 by logged reward
           with 50-tile corr_s / conc net, count corr>0.7 in the top 20, ranks of logged corr>0.7.
           Same column for --ref (default B5_PC_P8, read only) for comparison.
 ent_blocks per 16 updates from BATCH_LOG: per family H_f mean, alpha at block end, batch corr_s,
           gross bp/day, IC, n_valid sum, grammar rejects sum.
 config_diff LAUNCH_MANIFEST args + modules of this run vs --ref.
 prereg    RUN_CONFIG.json's pre-registered checks, evaluated.
    python b5_round_ent.py B5_PC_P8_ENT --mode p8
TRAIN only; no DEV, no OOS. Exit 0 = found, 10 = not found.
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
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b5_round as BR
import m2h_b5_trainer_ent as TE
import b5_explain as X
import m2h_common as M
import m2h_dense_common as D

TV = BR.TV
BR.TR = TE                      # freeze -> TV.final_pool(TR, ...) samples with the ENT trainer
A_ROOT = os.path.join(TE.ROOT, "artifacts", "m2h_v1")
N_RESCORE = 20


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []


def history(run_dir, vm, slices, starts, univ, sh, s, pool, n_rescore=N_RESCORE):
    agg = collections.defaultdict(lambda: dict(gt=[], cs=[], u=[], tokens=None))
    for line in open(os.path.join(run_dir, "FORMULAS.jsonl"), encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("invalid_reason") is not None or r.get("gross_t") is None:
            continue
        a = agg[(r["basin"], r["expr"])]
        a["gt"].append(r["gross_t"])
        a["u"].append(r["update"])
        if r.get("corr_s") is not None:
            a["cs"].append(r["corr_s"])
        a["tokens"] = a["tokens"] or r["tokens"]
    out = {}
    for fam in M.BASIN_NAMES:
        xs = [dict(expr=e, gross_t_logged=float(np.mean(v["gt"])), n_seen=len(v["gt"]),
                   first_update=min(v["u"]), last_update=max(v["u"]),
                   corr_s_logged=float(np.mean(v["cs"])) if v["cs"] else None, tokens=v["tokens"])
              for (f, e), v in agg.items() if f == fam]
        xs.sort(key=lambda x: (-x["gross_t_logged"], -x["n_seen"]))
        hi = [i + 1 for i, x in enumerate(xs) if x["corr_s_logged"] is not None and x["corr_s_logged"] > 0.7]
        top = []
        for i, x in enumerate(xs[:n_rescore]):
            body = list(TE.strip_hold(vm, list(x["tokens"])))
            rec = BR.score_both(vm, body, slices, starts, univ, sh, s, pool)
            top.append(dict(rank=i + 1, expr=x["expr"], n_seen=x["n_seen"], first_update=x["first_update"],
                            last_update=x["last_update"], gross_t_logged=round(x["gross_t_logged"], 4),
                            corr_s_logged=None if x["corr_s_logged"] is None else round(x["corr_s_logged"], 4),
                            corr_s_50=None if rec is None else round(float(rec["corr"].mean()), 4),
                            conc_k10_train_net_bpd=None if rec is None else round(D.bpd(float(rec["cn"].mean())), 3),
                            explain=X.explain(vm, body)))
        c50 = [t["corr_s_50"] for t in top if t["corr_s_50"] is not None]
        out[fam] = dict(distinct_valid=len(xs), logged_corr_gt07_n=len(hi), logged_corr_gt07_ranks=hi[:20],
                        top20_corr50_gt07_n=sum(c > 0.7 for c in c50),
                        top20_corr50_max=max(c50) if c50 else None, top5=top[:5])
        print("  hist %s distinct %d corr>0.7 ranks %s top5 corr50 %s" % (
            fam, len(xs), hi[:8], [t["corr_s_50"] for t in top[:5]]), flush=True)
    return out


def ent_blocks(log):
    out = []
    for b in range(log[-1]["update"] // 16 + 1):
        lg = [x for x in log if x["update"] // 16 == b]
        rec = dict(block=b, updates=[16 * b, 16 * b + 15],
                   skipped=sum(1 for x in lg if x.get("skipped_param_step")))
        for k, fam in enumerate(M.BASIN_NAMES):
            pf = [x["per_family"][k] for x in lg if len(x.get("per_family", [])) > k]
            rec[fam] = dict(H=TV.fmean([p.get("entropy") for p in pf]),
                            alpha_end=pf[-1].get("ent_alpha_next") if pf else None,
                            corr_s=TV.fmean([p.get("corr_s") for p in pf]),
                            gross_bpd=TV.fmean([p.get("gross_bpd") for p in pf]),
                            ic=TV.fmean([p.get("ic") for p in pf]),
                            n_valid=sum(int(p.get("n_valid") or 0) for p in pf),
                            rejected=sum(int(p.get("grammar_reject") or 0) for p in pf))
        out.append(rec)
    return out


def config_diff(run_dir, ref):
    new = json.load(open(os.path.join(run_dir, "LAUNCH_MANIFEST.json"), encoding="utf-8"))
    old = json.load(open(os.path.join(A_ROOT, ref, "LAUNCH_MANIFEST.json"), encoding="utf-8"))
    args = {k: [old["args"].get(k), new["args"].get(k)] for k in sorted(set(old["args"]) | set(new["args"]))
            if old["args"].get(k) != new["args"].get(k)}
    mods = {k: [old["modules"].get(k), new["modules"].get(k)]
            for k in sorted(set(old["modules"]) | set(new["modules"]))
            if old["modules"].get(k) != new["modules"].get(k)}
    import difflib
    a = open(os.path.join(HERE, "m2h_b5_trainer.py"), encoding="utf-8").read().splitlines()
    b = open(TE.__file__, encoding="utf-8").read().splitlines()
    code = [l for l in difflib.unified_diff(a, b, "m2h_b5_trainer.py", "m2h_b5_trainer_ent.py", n=0, lineterm="")]
    return dict(args=args, modules=mods, vocab_size=[old.get("vocab_size"), new.get("vocab_size")],
                model_params=[old.get("model_params"), new.get("model_params")],
                code_diff_changed_lines=sum(1 for l in code if l[:1] in "+-" and l[:3] not in ("+++", "---")),
                code_diff=code)


def prereg(cfg, log, cur, stats):
    p = cfg["prereg"]
    k1 = M.BASIN_NAMES.index("T1")
    h_after = [x["per_family"][k1].get("entropy") for x in log if x["update"] >= p["run_H_from_update"]]
    blk = [np.mean([v for v in h_after[i:i + 16] if v is not None]) for i in range(0, len(h_after), 16)
           if any(v is not None for v in h_after[i:i + 16])]
    pin = {}
    for k, fam in enumerate(M.BASIN_NAMES):
        run_, best, n_at = 0, 0, 0
        for x in log:
            at = (x["per_family"][k].get("ent_alpha_next") or 0.0) >= cfg["ent"]["alpha_max"] - 1e-9
            n_at += at
            run_ = run_ + 1 if at else 0
            best = max(best, run_)
        pin[fam] = dict(updates_at_max=n_at, longest_run=best)
    t1max = max((c["corr_mean"] for c in cur if c["corr_mean"] is not None), default=None)
    out = dict(
        run_T1_H=dict(rule=p["run_H_rule"], block_min=round(float(min(blk)), 4) if blk else None,
                      update_min=round(float(min(v for v in h_after if v is not None)), 4) if blk else None,
                      passed=bool(blk and min(blk) >= p["run_H_floor"])),
        run_alpha_not_pinned=dict(rule=p["pin_rule"], per_family=pin,
                                  passed=all(v["longest_run"] < p["pin_max_run"] for v in pin.values())),
        learn_T1_corr=dict(rule="T1 batch mean corr max > 0.3 (curve)", value=t1max,
                           passed=bool(t1max is not None and t1max > 0.3)),
        learn_T1_distinct=dict(rule=p["distinct_rule"], value=stats["T1"]["distinct"],
                               ref=p["distinct_ref"], passed=stats["T1"]["distinct"] >= p["distinct_min"]),
        verdict_prediction=p["verdict_prediction"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--mode", choices=("p32", "p8", "null"), required=True)
    ap.add_argument("--ref", default="B5_PC_P8", help="read only: the run this one copies")
    ap.add_argument("--per-family", type=int, default=TV.PER_FAMILY, help="smoke test only")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="ROUND_REPORT.json")
    ap.add_argument("--config", default=None, help="RUN_CONFIG.json; default <run dir>; report smoke only")
    ap.add_argument("--no-ref-history", action="store_true", help="report smoke only")
    ap.add_argument("--hist-n", type=int, default=N_RESCORE, help="history formulas rescored per family")
    a = ap.parse_args()
    t_all = time.time()
    run_dir = os.path.join(A_ROOT, a.run)
    done = [f for f in ("TRAIN_RECEIPT_V1.json", "STOPPED.json") if os.path.exists(os.path.join(run_dir, f))]
    if not done:
        raise SystemExit(f"{a.run}: no receipt and no STOPPED.json")
    cfg = json.load(open(a.config or os.path.join(run_dir, "RUN_CONFIG.json"), encoding="utf-8"))
    dev = torch.device(a.device)
    vm = TV.ST.make_vm()
    s = np.load(TV.PL.S_FILE)
    w, sh = TV.trained_world(a.mode)
    univ = w["univ"]
    import pyarrow.parquet as pq
    syms = sorted(pq.read_table(TE.PANEL, columns=["symbol"]).column(0).unique().to_pylist())
    assert len(syms) == univ.shape[0], (len(syms), univ.shape)
    pool = BR.W.pool_cols(syms, os.path.join(HERE, "CONC_POOL.json"))
    starts = M.tile_starts(w["feats"].shape[2])
    assert len(starts) == 50, len(starts)
    slices = M.tile_slices(M.pad_feats(w["feats"]), starts, dev)

    log = jl(os.path.join(run_dir, "BATCH_LOG.jsonl"))
    rows = TV.read_formulas(os.path.join(run_dir, "FORMULAS.jsonl"))
    last = int(log[-1]["update"])
    t0 = time.time()
    cur = TV.curve(TE, vm, rows, [slices[i] for i in TV.CURVE_TILES], [starts[i] for i in TV.CURVE_TILES],
                   univ, s, last)
    print("curve %d points %.0fs" % (len(cur), time.time() - t0), flush=True)
    blk = TV.blocks("dense", log, rows)
    del rows
    prog = jl(os.path.join(run_dir, "PROGRESS_16.jsonl"))
    hist = history(run_dir, vm, slices, starts, univ, sh, s, pool, a.hist_n)
    hist_ref = history(os.path.join(A_ROOT, a.ref), vm, slices, starts, univ, sh, s, pool, a.hist_n) \
        if os.path.exists(os.path.join(A_ROOT, a.ref, "FORMULAS.jsonl")) and not a.no_ref_history else None

    model, ck_name, ck_upd = TV.load_final(TE, run_dir, dev)
    fams, stats = BR.freeze(vm, model, dev, slices, starts, univ, sh, s, pool, a.per_family)
    del model, slices
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    blob = json.dumps(fams, sort_keys=True).encode()
    t1_max = max((p["corr_mean"] for p in cur if p["corr_mean"] is not None), default=None)
    top = max((f["corr_s"] for fam in fams.values() for f in fam), default=None)
    found = bool(t1_max is not None and t1_max > 0.3 and top is not None and top > 0.7)
    converged = {fam: [dict(rank=f["rank"], expr=f["expr"], n_tokens=f["n_tokens"], explain=f["explain"],
                            conc_k10_train_net_bpd=f["conc_k10_train_net_bpd"], corr_s=f["corr_s"])
                       for f in fs] for fam, fs in fams.items()}
    rep = dict(schema="B5_ROUND_REPORT_V1", variant="BASIN5_PC_ENT", run=a.run, trainer="m2h_b5_trainer_ent",
               mode=a.mode, ended_by=done, last_update=last, checkpoint=ck_name, checkpoint_update=ck_upd,
               found=found, t1_batch_corr_max=t1_max, frozen_corr_max=top,
               rule="found = T1 batch mean corr with s > 0.3 at a logged update AND a frozen top-5 corr > 0.7 "
                    "(ADR-0008); batch mean over grammar-legal valid T1 formulas",
               收敛到了什么=converged,
               history_top5_by_train_reward=dict(note="NOT in the verdict; logged gross_t rank, 50-tile rescore",
                                                 this=hist, ref=a.ref, ref_run=hist_ref),
               ent_blocks=ent_blocks(log), progress_16=prog,
               prereg=prereg(cfg, log, cur, stats), step3=cfg.get("step3"), config_diff=config_diff(run_dir, a.ref),
               curve=cur, curve_tiles=[starts[i] for i in TV.CURVE_TILES], blocks=blk,
               frozen=fams, frozen_sha256=hashlib.sha256(blob).hexdigest(), freeze_stats=stats,
               freeze_rule=dict(seed=TV.FINAL_SEED, per_family=a.per_family, mode="train", tiles=50,
                                grammar="m2h_b5_grammar.check, illegal dropped",
                                rank="conc book K=%d flat6 TRAIN net, H%d" % (BR.K, BR.HOLD),
                                dedup="identical 50-tile conc net vector", world=a.mode,
                                conc_pool=int(pool.sum())),
               seconds=round(time.time() - t_all, 1), written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               dev_read=False, oos_read=False)
    json.dump(rep, open(os.path.join(run_dir, a.out), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("ROUND %s found=%s t1corr_max=%s frozen_corr_max=%s T1 distinct %s ck=%s %.0fs"
          % (a.run, found, t1_max, top, stats["T1"]["distinct"], ck_name, time.time() - t_all), flush=True)
    sys.exit(0 if found else 10)


if __name__ == "__main__":
    main()
