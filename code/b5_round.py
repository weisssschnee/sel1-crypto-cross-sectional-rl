"""BASIN5 round report (BASIN5_ORDER s5/s6): tv1_round's curve + freeze, B5 trainer, conc book.

    python b5_round.py <RUN_NAME> --mode p8|p32|null|real

 curve    tv1_round.curve verbatim (T1 batch mean corr with s every 16 updates, 4 fixed TRAIN
          tiles). Grammar-rejected rows are invalid in FORMULAS.jsonl, so the mean is over the
          grammar-legal valid T1 formulas; n per point is in the row.
 freeze   latest checkpoint, 256 per family, seed 20261108, train mode (tv1_round.final_pool);
          grammar-illegal formulas dropped (count reported); each remaining one scored on the 50
          TRAIN tiles of the world it trained on: concentrated book K=10 (m2h_conc_weights, pool
          CONC_POOL.json) net / gross, full-pool book net / gross, corr with s. Ranked by conc
          K=10 TRAIN net (flat6), dedup on the identical 50-tile conc net vector, top 5 per family.
          Columns per formula: n_tokens, level_flag, explanation (b5_explain).
 found    ADR-0008: T1 batch mean corr > 0.3 at a logged update AND a frozen top-5 corr > 0.7.
 收敛到了什么  the frozen top-5 expressions with one sentence each, plus the step-6 PROGRESS_16 rows.
Writes ROUND_REPORT.json in the run dir. TRAIN only; no DEV, no OOS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tv1_round as TV                 # sys.path for m1_* / okx_alpha_discovery / v4 roots
import m2h_b5_trainer as TR
import m2h_b5_grammar as G
import m2h_conc_weights as W
import b5_explain as X
import m2h_common as M
import m2h_dense_common as D
import m1_ruler as R
from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid

K = W.K_DEFAULT
HOLD = 168


def score_both(vm, body, slices, starts, univ, sh, s, pool):
    """Per tile: one VM signal, the full-pool ruler book and the conc K book. None if invalid."""
    E = M.EPISODE
    out = dict(cn=[], cg=[], fn=[], fg=[], corr=[], cturn=[])
    try:
        for s0, sl in zip(starts, slices):
            sigT = M.window_signal(vm, list(body), sl, s0, univ)
            el = sh["eligible"][:, s0:s0 + E].T
            live = np.isfinite(sigT) & el
            probe_t = [t for t in [M.LAG] + list(range(E // 4, E, E // 4)) if t < E]
            if min(float(live[t].sum() / max(el[t].sum(), 1)) for t in probe_t) < M.MIN_COVERAGE:
                return None
            full = R.score_window(sigT, sh, s0, E, [HOLD], policy_mod, costs=M.COSTS,
                                  variants=("win",), band=R.BAND, nav_usd=R.NAV_USD)[HOLD]["win"][0]
            cl, cp = R.score_window(W.conc_signal(sigT, el, pool, K), sh, s0, E, [HOLD], policy_mod,
                                    costs=M.COSTS, variants=("win",), band=R.BAND,
                                    nav_usd=R.NAV_USD)[HOLD]["win"]
            vals = (cl["flat6"], cl["cost0"], full["flat6"], full["cost0"])
            if not all(np.isfinite(v) for v in vals):
                return None
            for k, v in zip(("cn", "cg", "fn", "fg"), vals):
                out[k].append(float(v))
            out["cturn"].append(cp["turnover"] / (E / 24.0))
            out["corr"].append(D.corr_with_s(sigT, s, s0))
    except (M.ZeroCoverage, CandidateLocalInvalid):
        return None
    return {k: np.asarray(v) for k, v in out.items()}


def freeze(vm, model, device, slices, starts, univ, sh, s, pool, n):
    pools = TV.final_pool(TR, model, vm, device, n)
    fams, stats = {}, {}
    for fam in M.BASIN_NAMES:
        t0 = time.time()
        scored, illegal = [], 0
        for body, hold in pools[fam]:
            if G.check(vm, list(body)) is not None:
                illegal += 1
                continue
            rec = score_both(vm, body, slices, starts, univ, sh, s, pool)
            if rec is not None:
                scored.append((float(rec["cn"].mean()), body, hold, rec))
        scored.sort(key=lambda x: -x[0])
        kept = []
        for sc in scored:
            if any(np.array_equal(sc[3]["cn"], k[3]["cn"]) for k in kept):
                continue
            kept.append(sc)
            if len(kept) == 5:
                break
        fams[fam] = [dict(rank=i + 1, body=[int(x) for x in b], hold=int(h),
                          expr=vm.describe(list(b), list(M.TERMINALS)), n_tokens=len(b) - 1,
                          level_flag=bool(D.level_flag(vm, list(b))),
                          explain=X.explain(vm, list(b)),
                          conc_k10_train_net_bpd=round(D.bpd(m), 3),
                          conc_k10_train_gross_bpd=round(D.bpd(float(rec["cg"].mean())), 3),
                          conc_k10_turnover_per_day=round(float(rec["cturn"].mean()), 3),
                          full_train_net_bpd=round(D.bpd(float(rec["fn"].mean())), 3),
                          full_train_gross_bpd=round(D.bpd(float(rec["fg"].mean())), 3),
                          corr_s=round(float(rec["corr"].mean()), 4))
                     for i, (m, b, h, rec) in enumerate(kept)]
        stats[fam] = dict(sampled=n, distinct=len(pools[fam]), grammar_illegal=illegal,
                          valid=len(scored), seconds=round(time.time() - t0, 1))
        print("  %s distinct %d illegal %d valid %d %.0fs" % (fam, len(pools[fam]), illegal,
                                                             len(scored), time.time() - t0), flush=True)
    return fams, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--mode", choices=("p32", "p8", "null", "real"), required=True)
    ap.add_argument("--per-family", type=int, default=TV.PER_FAMILY, help="smoke test only")
    ap.add_argument("--device", default="cuda", help="cpu for a report smoke while the GPU trains")
    ap.add_argument("--out", default="ROUND_REPORT.json")
    a = ap.parse_args()
    t_all = time.time()
    run_dir = os.path.join(TR.ROOT, "artifacts", "m2h_v1", a.run)
    done = [f for f in ("TRAIN_RECEIPT_V1.json", "STOPPED.json") if os.path.exists(os.path.join(run_dir, f))]
    if not done:
        raise SystemExit(f"{a.run}: no receipt and no STOPPED.json")
    dev = torch.device(a.device)
    vm = TV.ST.make_vm()
    s = np.load(TV.PL.S_FILE)
    w, sh = TV.trained_world(a.mode)
    univ = w["univ"]
    import pyarrow.parquet as pq           # long_to_grids orders coins by sorted unique symbol
    syms = sorted(pq.read_table(TR.PANEL, columns=["symbol"]).column(0).unique().to_pylist())
    assert len(syms) == univ.shape[0], (len(syms), univ.shape)
    pool = W.pool_cols(syms, os.path.join(HERE, "CONC_POOL.json"))
    starts = M.tile_starts(w["feats"].shape[2])
    assert len(starts) == 50, len(starts)
    slices = M.tile_slices(M.pad_feats(w["feats"]), starts, dev)

    log = [json.loads(l) for l in open(os.path.join(run_dir, "BATCH_LOG.jsonl"), encoding="utf-8") if l.strip()]
    rows = TV.read_formulas(os.path.join(run_dir, "FORMULAS.jsonl"))
    last = int(log[-1]["update"])
    t0 = time.time()
    cur = TV.curve(TR, vm, rows, [slices[i] for i in TV.CURVE_TILES], [starts[i] for i in TV.CURVE_TILES],
                   univ, s, last)
    print("curve %d points %.0fs" % (len(cur), time.time() - t0), flush=True)
    blk = TV.blocks("dense", log, rows)
    del rows
    prog_p = os.path.join(run_dir, "PROGRESS_16.jsonl")
    prog = [json.loads(l) for l in open(prog_p, encoding="utf-8") if l.strip()] if os.path.exists(prog_p) else []

    model, ck_name, ck_upd = TV.load_final(TR, run_dir, dev)
    fams, stats = freeze(vm, model, dev, slices, starts, univ, sh, s, pool, a.per_family)
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
    rep = dict(schema="B5_ROUND_REPORT_V1", run=a.run, trainer="m2h_b5_trainer", mode=a.mode, ended_by=done,
               last_update=last, checkpoint=ck_name, checkpoint_update=ck_upd,
               found=found, t1_batch_corr_max=t1_max, frozen_corr_max=top,
               rule="found = T1 batch mean corr with s > 0.3 at a logged update AND a frozen top-5 corr > 0.7 "
                    "(ADR-0008); batch mean over grammar-legal valid T1 formulas",
               收敛到了什么=converged, progress_16=prog,
               curve=cur, curve_tiles=[starts[i] for i in TV.CURVE_TILES], blocks=blk,
               frozen=fams, frozen_sha256=hashlib.sha256(blob).hexdigest(), freeze_stats=stats,
               freeze_rule=dict(seed=TV.FINAL_SEED, per_family=a.per_family, mode="train", tiles=50,
                                grammar="m2h_b5_grammar.check, illegal dropped",
                                rank="conc book K=%d flat6 TRAIN net, H%d" % (K, HOLD),
                                dedup="identical 50-tile conc net vector", world=a.mode,
                                conc_pool=int(pool.sum())),
               seconds=round(time.time() - t_all, 1), written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               dev_read=False, oos_read=False)
    json.dump(rep, open(os.path.join(run_dir, a.out), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print("ROUND %s found=%s t1corr_max=%s frozen_corr_max=%s ck=%s %.0fs"
          % (a.run, found, t1_max, top, ck_name, time.time() - t_all), flush=True)
    sys.exit(0 if found else 10)


if __name__ == "__main__":
    main()
