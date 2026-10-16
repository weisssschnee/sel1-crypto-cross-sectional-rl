"""TV1 per-round report (TV1_ORDER s6; TV1_A2_BASIN4_ORDER s0: freeze from the u256 / STOPPED
checkpoint only).

    python tv1_round.py <RUN_NAME> --trainer old|dense --mode p32|p8|null

 curve   updates 0, 16, ..., last: T1 batch mean CS Spearman with s (D.corr_with_s on 4 fixed
         TRAIN tiles, the valid T1 formulas of that update in FORMULAS.jsonl); per 16-update block
         and family: batch gross bp/day, batch IC (dense), turnover; entropy, gnorm, % over clip
 freeze  the latest checkpoint (u255 or the STOPPED one), 256 per family, seed 20261108, train
         mode; distinct (body, hold); each valid one re-scored on the 50 TRAIN tiles of the world
         the run trained on (shard rebuilt as m2h_plant does); ranked by flat6 net at its own
         hold; behaviour dedup on the 50-tile net vector; top 5 per family, sha256. Columns:
         corr with s, TRAIN net / gross bp/day, SIGN/ABS level flag (listed, not dropped)
 found   T1 batch mean corr > 0.3 at some logged update AND a frozen top-5 corr > 0.7
Writes ROUND_REPORT.json in the run dir. TRAIN only; no DEV, no OOS.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m2h_selftest as ST            # sys.path for m1_* / okx_alpha_discovery
import m2h_dense_trainer as DT       # legacy v4 roots
import m2h_common as M
import m2h_dense_common as D
import m2h_plant as PL
from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import OPERATOR_NAMES, CandidateLocalInvalid
from okx_alpha_discovery.adapters.xs_tscs_sampler_v2 import build_legality_table

FINAL_SEED = 20261108
PER_FAMILY = 256
CURVE_TILES = (0, 12, 25, 37)
CLIP = 5.0


def trained_world(mode):
    syms, ts, grids = M.load_train_grids(DT.PANEL)
    w = M.build_world(grids)
    base = w["shard"]
    if mode == "null":
        sh = D.null_shard(base, w["univ"], D.PLANT_SEED)
    elif mode == "real":
        sh = base
    else:
        cal = json.load(open(PL.ALPHA_FILE, encoding="utf-8"))[mode]
        s = np.load(PL.S_FILE)
        if PL.s_sha(s) != cal["s_sha16"]:
            raise RuntimeError("s differs from the calibrated one")
        sh = D.plant_shard(base, s, cal["alpha"])
    return w, sh


def load_final(TR, run_dir, device):
    from train_basin_riskpg_v1 import make_basin_model, apply_output_bias
    import model_core.alphagpt as alphagpt_mod
    from model_core.config import ModelConfig
    from model_core.vocab import FormulaVocab
    ModelConfig.DEVICE = device
    names = (list(OPERATOR_NAMES) + [f"W{x}" for x in M.WINDOWS] + ["END"] + M.HOLD_NAMES
             + list(M.BASIN_TOKEN_NAMES))
    model, _ = make_basin_model(alphagpt_mod=alphagpt_mod, ModelConfig=ModelConfig,
                                FormulaVocab=FormulaVocab, feature_names=list(M.TERMINALS),
                                operator_token_names=names, max_len=M.MAX_PROGRAM_LEN, device=device)
    apply_output_bias(model, len(M.TERMINALS), len(OPERATOR_NAMES), len(M.WINDOWS),
                      TR.OP_BIAS, TR.END_BIAS)
    cks = sorted(glob.glob(os.path.join(run_dir, "checkpoint_*.pt")))
    ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    return model, os.path.basename(cks[-1]), int(ck["update"])


def final_pool(TR, model, vm, device, n):
    out = {}
    for fam in M.BASIN_NAMES:
        tab = build_legality_table(vm, max_len=M.MAX_PROGRAM_LEN, available_terminals=M.basin_terms(fam),
                                   available_windows=M.WINDOWS).to(device)
        torch.manual_seed(FINAL_SEED)
        gen = torch.Generator(device=device)
        gen.manual_seed(FINAL_SEED)
        with torch.no_grad():
            b = TR.sample_batch_with_hold_basins(model, vm, [tab], [n], device=device,
                                                 generator=gen, basins=(fam,))
        seen, pool = set(), []
        for t in b.formulas:
            k = (tuple(int(x) for x in TR.strip_hold(vm, list(t))), int(TR.hold_of(vm, list(t))))
            if k not in seen:
                seen.add(k)
                pool.append(k)
        out[fam] = pool
    return out


def score50(vm, body, hold, slices, starts, univ, sh, s):
    try:
        res = D.score_dense(vm, list(body), slices, starts, univ, sh, policy_mod, holds=(hold,),
                            costs=("flat6", "cost0"), want_ic=False, want_sig=True)
    except (M.ZeroCoverage, CandidateLocalInvalid):
        return None
    net = np.asarray(res["per_hold"][hold]["flat6"], dtype=np.float64)
    if not np.isfinite(net).all():
        return None
    cs = [D.corr_with_s(g, s, s0) for g, s0 in zip(res["sig"], starts)]
    return dict(net=net, gross=np.asarray(res["per_hold"][hold]["cost0"], dtype=np.float64),
                corr=float(np.mean(cs)))


def fmean(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return round(float(np.mean(v)), 4) if v else None


def read_formulas(path):
    """Slim rows: what the curve and the block table need."""
    out = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        out.append(dict(update=int(r["update"]), basin=r["basin"], tokens=r["tokens"],
                        valid=r.get("invalid_reason") is None, gross_bpd=r.get("gross_bp_day"),
                        ic=r.get("ic"), turn=r.get("turnover_per_day")))
    return out


def curve(TR, vm, rows, slices, starts, univ, s, last):
    pts = sorted(set(range(0, last + 1, 16)) | {last})
    want = set(pts)
    by = collections.defaultdict(list)
    for r in rows:
        if r["update"] in want and r["basin"] == "T1" and r["valid"]:
            by[r["update"]].append(r)
    out = []
    for u in pts:
        cs = []
        for r in by[u]:
            body = list(TR.strip_hold(vm, list(r["tokens"])))
            try:
                v = [D.corr_with_s(M.window_signal(vm, body, sl, s0, univ), s, s0)
                     for sl, s0 in zip(slices, starts)]
            except (M.ZeroCoverage, CandidateLocalInvalid):
                continue
            cs.append(float(np.mean(v)))
        cs = np.asarray(cs)
        out.append(dict(update=u, n=int(len(cs)), corr_mean=fmean(cs),
                        corr_max=round(float(cs.max()), 4) if len(cs) else None,
                        frac_gt_07=round(float((cs > 0.7).mean()), 4) if len(cs) else None))
    return out


def blocks(trainer, log, rows):
    nb = log[-1]["update"] // 16 + 1
    by = collections.defaultdict(list)
    for r in rows:
        if r["valid"]:
            by[(r["update"] // 16, r["basin"])].append(r)
    out = []
    for b in range(nb):
        lg = [x for x in log if x["update"] // 16 == b]
        gn = np.asarray([x["gradient_norm"] for x in lg if x.get("gradient_norm") is not None])
        rec = dict(block=b, updates=[16 * b, 16 * b + 15], gnorm=fmean(gn),
                   clip_pct=round(100.0 * float((gn > CLIP).mean()), 1) if len(gn) else None,
                   seconds=fmean([x["seconds"] for x in lg]))
        for k, fam in enumerate(M.BASIN_NAMES):
            v = by[(b, fam)]
            if trainer == "dense":
                ent = fmean([x["per_family"][k].get("entropy") for x in lg])
            else:                       # old row: entropy_loss = -0.03 * mean elite entropy
                ent = fmean([-x["entropy_loss"] / 0.03 for x in lg if x.get("entropy_loss") is not None])
            rec[fam] = dict(gross_bpd=fmean([x["gross_bpd"] for x in v]), ic=fmean([x["ic"] for x in v]),
                            turnover=fmean([x["turn"] for x in v]), entropy=ent, n_valid=len(v))
        out.append(rec)
    return out


def freeze(TR, vm, model, device, slices, starts, univ, sh, s, n):
    pools = final_pool(TR, model, vm, device, n)
    fams, stats = {}, {}
    for fam in M.BASIN_NAMES:
        t0 = time.time()
        scored = []
        for body, hold in pools[fam]:
            rec = score50(vm, body, hold, slices, starts, univ, sh, s)
            if rec is not None:
                scored.append((float(rec["net"].mean()), body, hold, rec))
        scored.sort(key=lambda x: -x[0])
        kept = []
        for sc in scored:
            if any(np.array_equal(sc[3]["net"], k[3]["net"]) for k in kept):
                continue
            kept.append(sc)
            if len(kept) == 5:
                break
        fams[fam] = [dict(rank=i + 1, body=[int(x) for x in b], hold=int(h),
                          expr=vm.describe(list(b), list(M.TERMINALS)),
                          train50_net_bpd=round(D.bpd(m), 3),
                          train50_gross_bpd=round(D.bpd(float(rec["gross"].mean())), 3),
                          corr_s=round(rec["corr"], 4), level_flag=bool(D.level_flag(vm, list(b))))
                     for i, (m, b, h, rec) in enumerate(kept)]
        stats[fam] = dict(sampled=n, distinct=len(pools[fam]), valid=len(scored),
                          pool_mean_net_bpd=round(D.bpd(np.mean([x[0] for x in scored])), 3) if scored else None,
                          seconds=round(time.time() - t0, 1))
        print("  %s distinct %d valid %d %.0fs" % (fam, len(pools[fam]), len(scored), time.time() - t0),
              flush=True)
    return fams, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--trainer", choices=("old", "dense", "dense2"), required=True)
    ap.add_argument("--mode", choices=("p32", "p8", "null", "real"), required=True)
    ap.add_argument("--per-family", type=int, default=PER_FAMILY, help="smoke test only")
    a = ap.parse_args()
    t_all = time.time()
    run_dir = os.path.join(DT.ROOT, "artifacts", "m2h_v1", a.run)
    done = [f for f in ("TRAIN_RECEIPT_V1.json", "STOPPED.json") if os.path.exists(os.path.join(run_dir, f))]
    if not done:
        raise SystemExit(f"{a.run}: no receipt and no STOPPED.json; the freeze waits for u256 / STOPPED")
    if a.trainer == "old":
        import m2h_trainer as TR
    elif a.trainer == "dense2":
        import m2h_dense_trainer_v2 as TR      # R2b: same log schema, --ent-coef/--adv-deadband
    else:
        TR = DT
    dev = torch.device("cuda")
    vm = ST.make_vm()
    s = np.load(PL.S_FILE)
    w, sh = trained_world(a.mode)
    univ = w["univ"]
    starts = M.tile_starts(w["feats"].shape[2])
    assert len(starts) == 50, len(starts)
    slices = M.tile_slices(M.pad_feats(w["feats"]), starts, dev)

    log = [json.loads(l) for l in open(os.path.join(run_dir, "BATCH_LOG.jsonl"), encoding="utf-8") if l.strip()]
    rows = read_formulas(os.path.join(run_dir, "FORMULAS.jsonl"))
    last = int(log[-1]["update"])
    t0 = time.time()
    cur = curve(TR, vm, rows, [slices[i] for i in CURVE_TILES], [starts[i] for i in CURVE_TILES], univ, s, last)
    print("curve %d points %.0fs" % (len(cur), time.time() - t0), flush=True)
    blk = blocks(a.trainer, log, rows)
    del rows

    model, ck_name, ck_upd = load_final(TR, run_dir, dev)
    fams, stats = freeze(TR, vm, model, dev, slices, starts, univ, sh, s, a.per_family)
    del model, slices
    torch.cuda.empty_cache()

    blob = json.dumps(fams, sort_keys=True).encode()
    t1_max = max((p["corr_mean"] for p in cur if p["corr_mean"] is not None), default=None)
    top = max((f["corr_s"] for fam in fams.values() for f in fam), default=None)
    found = bool(t1_max is not None and t1_max > 0.3 and top is not None and top > 0.7)
    rep = dict(schema="TV1_ROUND_REPORT_V1", run=a.run, trainer=a.trainer, mode=a.mode, ended_by=done,
               last_update=last, checkpoint=ck_name, checkpoint_update=ck_upd,
               found=found, t1_batch_corr_max=t1_max, frozen_corr_max=top,
               rule="found = T1 batch mean corr with s > 0.3 at a logged update AND a frozen top-5 corr > 0.7",
               curve=cur, curve_tiles=[starts[i] for i in CURVE_TILES], blocks=blk,
               frozen=fams, frozen_sha256=hashlib.sha256(blob).hexdigest(), freeze_stats=stats,
               freeze_rule=dict(seed=FINAL_SEED, per_family=a.per_family, mode="train", tiles=50,
                                rank="flat6 net at own hold", dedup="identical 50-tile net vector",
                                world=a.mode),
               seconds=round(time.time() - t_all, 1), written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               dev_read=False, oos_read=False)
    json.dump(rep, open(os.path.join(run_dir, "ROUND_REPORT.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print("ROUND %s found=%s t1corr_max=%s frozen_corr_max=%s ck=%s %.0fs"
          % (a.run, found, t1_max, top, ck_name, time.time() - t_all), flush=True)


if __name__ == "__main__":
    main()
