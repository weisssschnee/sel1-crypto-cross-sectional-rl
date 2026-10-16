"""CR1b trainer (QuantMind 10-14 CR1b approval; order V4_COMBO_V1_CR1 s3; contract V4). New file, copied from
cr1_trainer.py (CR1's run record, not modified). Sampler, grammar mask, policy-gradient step and checkpoint
plumbing are m2h_b5_trainer_dbg_memo's functions, imported unchanged; the world and the reward are cr1b_common's.

  reward   per formula: mean over the drawn A tiles of the marginal full-book net log NAV (flat6) of
           combo + r_f over the combo alone, r_f = the formula's z residualised per hour on the 8 cells
           (cr1b_common docstring). Full book is the training signal only; every decision (frozen set,
           NULL bar, found) is on CR1B_ALPHA.json's rank_book, a concentrated book (cr1b_round).
  update   16 of the 30 A tiles per update (B tiles never enter training); batch 128 (T1/T2/T3 quota),
           H168 only; within-family z +-3 over valid, invalid -1, entropy 0.03 (as CR1).
  launch   refuses unless CR1B_ALPHA.json (cr1b_calib) has rc 0: plant visible on a concentrated book and
           ranked <= 10 among the B-half candidates; that rank goes into LAUNCH_MANIFEST.precheck.
  speed    per-update memo (each body once) and a cross-update (body, tile) cache: the VM and the ruler run
           only on tiles a body was never scored on (CR1 PC log: 33 % of the time was repeats). Same per-tile
           numbers, same invalid rules (first failing tile in draw order; RESID_NO_Z over all drawn tiles).
           Selftests at start: pool == in-process and cache path == direct Q.marginal, bit for bit, for the
           plant formula; a failed cache selftest turns the cache off (logged), it does not stop the run.

    python -I cr1b_trainer.py OUT --mode p8 --updates 256 [--resume auto]
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cr1b_common as Q
import numpy as np
import torch
import m2h_b5_trainer_dbg_memo as TB
import m2h_common as M
import m2h_dense_common as D
import m2h_b5_grammar as G
import m1_ruler as R
import v4_world as X
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid, OPERATOR_NAMES
from okx_alpha_discovery.adapters.xs_tscs_sampler_v2 import build_legality_table

TRAIN_BOOK = ("full",)
CORR_TILES = 4
LEARNING_RATE, OP_BIAS, END_BIAS = TB.LEARNING_RATE, TB.OP_BIAS, TB.END_BIAS
CHECKPOINT_EVERY, PROGRESS_EVERY = 16, 16


def modules():
    return dict(cr1b_trainer=TB._sha(os.path.abspath(__file__)), cr1b_common=TB._sha(Q.__file__),
                cr1_common=TB._sha(Q.Q1.__file__),
                v4_world=TB._sha(X.__file__), m1_ruler=TB._sha(R.__file__), m2h_common=TB._sha(M.__file__),
                m2h_dense_common=TB._sha(D.__file__), m2h_b5_grammar=TB._sha(G.__file__),
                dbg_memo=TB._sha(TB.__file__))


def make_pool(ctx, out_dir, workers):
    cdir = Q.write_cache(ctx, os.path.join(out_dir, "_cache"))
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"            # inherited by the workers only
    pool = mp.get_context("spawn").Pool(workers, initializer=Q.init_worker, initargs=(cdir,))
    return pool


def selftest_pool(ctx, vm, pool, device):
    pb = D.plant_body(vm)
    st = ctx["A"][:3]
    sl = M.tile_slices(ctx["feats_pad"], st, device)
    sigs, _ = Q.tile_sigs(ctx, vm, pb, st, sl)
    a = Q.marginal_from_sigs(ctx, sigs, st, (TRAIN_BOOK, X.K10), 3)
    _, b = pool.apply_async(Q.worker_task, (("st", Q.pack(sigs), st, (TRAIN_BOOK, X.K10), 3),)).get()
    d = max(abs(x - y) for k in a["books"] for f in a["books"][k] for x, y in zip(a["books"][k][f], b["books"][k][f]))
    d = max(d, abs(a["corr"] - b["corr"]), abs(a["corr_raw"] - b["corr_raw"]))
    return dict(ok=bool(d <= 1e-12), max_abs_diff=float(d),
                full_net_bpd=[round(Q.bpd(x), 4) for x in a["books"]["full"]["net"]],
                k10_net_bpd=[round(Q.bpd(x), 4) for x in a["books"]["conc_k10"]["net"]], corr=round(a["corr"], 4))


def score_batch(ctx, vm, pool, tcache, uniq, starts, slices, use_cache=True):
    """Score the unique bodies of one update on the drawn tiles. (body, tile) cache: the VM and the ruler
    run only on tiles this body was never scored on. Same semantics as Q.marginal over `starts`: the first
    tile in draw order under the coverage probe makes the body invalid with that tile's reason; RESID_NO_Z
    when the residual is empty on every drawn tile. Returns res, bad, cache hits, vm seconds, wait seconds."""
    if not use_cache:
        tcache.clear()
    pend, bad, lfm, hits = {}, {}, {}, 0
    t0 = time.time()
    for body in uniq:
        sigs, miss, lfs = [], [], []
        for s0, sl in zip(starts, slices):
            c = tcache.get((body, s0))
            if c is not None:
                hits += 1
                if c[0] == "BAD":
                    bad[body] = c[1]
                    break
                continue
            try:
                sg, f = Q.tile_sigs(ctx, vm, list(body), [s0], [sl])
            except (M.ZeroCoverage, CandidateLocalInvalid) as e:
                tcache[(body, s0)] = ("BAD", str(e))
                bad[body] = str(e)
                break
            sigs.append(sg[0])
            miss.append(s0)
            lfs.append(f)
        if body in bad:
            continue
        lfm[body] = lfs
        pend[body] = (miss, pool.apply_async(Q.worker_task, ((body, Q.pack(sigs), miss, (TRAIN_BOOK,), len(miss), False),))
                      if miss else None)
    t_vm = time.time() - t0
    res = {}
    for body, (miss, ar) in pend.items():
        if ar is not None:
            _, r = ar.get()
            if isinstance(r, str):
                bad[body] = r[len("INVALID:"):]
                continue
            b, pt = r["books"]["full"], r["per_tile"]
            for j, s0 in enumerate(miss):
                tcache[(body, s0)] = ("OK", b["net"][j], b["gross"][j], b["turn"][j], pt["nz"][j], pt["corr"][j],
                                      pt["corr_raw"][j], lfm[body][j])
        cs = [tcache[(body, s0)] for s0 in starts]
        if sum(c[4] for c in cs) == 0:
            bad[body] = "RESID_NO_Z"
            continue
        res[body] = dict(net=[c[1] for c in cs], gross=[c[2] for c in cs], turn=[c[3] for c in cs],
                         corr=float(np.mean([c[5] for c in cs[:CORR_TILES]])),
                         corr_raw=float(np.mean([c[6] for c in cs[:CORR_TILES]])),
                         lf=float(np.mean([c[7] for c in cs])))
    if not use_cache:
        tcache.clear()
    return res, bad, hits, t_vm, time.time() - t0 - t_vm


def selftest_cache(ctx, vm, pool, device):
    """Cache path == direct Q.marginal (in-process), bit for bit: cold, fully cached, partly cached."""
    pb = tuple(D.plant_body(vm))
    out, d, tc = [], 0.0, {}
    for st in (ctx["A"][:3], ctx["A"][:3], ctx["A"][1:5]):
        sl = M.tile_slices(ctx["feats_pad"], st, device)
        ref = Q.marginal(ctx, vm, list(pb), st, sl, books=(TRAIN_BOOK,), corr_tiles=CORR_TILES)
        res, bad, hits, _, _ = score_batch(ctx, vm, pool, tc, [pb], st, sl)
        if pb not in res:
            return dict(ok=False, error="plant invalid in cache path: %s" % bad.get(pb), steps=out)
        r = res[pb]
        dd = max([abs(x - y) for x, y in zip(ref["books"]["full"]["net"], r["net"])]
                 + [abs(x - y) for x, y in zip(ref["books"]["full"]["gross"], r["gross"])]
                 + [abs(ref["corr"] - r["corr"]), abs(ref["corr_raw"] - r["corr_raw"])])
        d = max(d, dd)
        out.append(dict(tiles=len(st), hits=hits, diff=dd))
    return dict(ok=bool(d == 0.0), max_abs_diff=float(d), steps=out)


DECISIONS = dict(
    precheck="CR1B_ALPHA.json rc 0 (cr1b_calib): s_perp8 orthogonal to the 8 cells, alpha p8 / p32 within 2 %, "
             "the plant formula as the residualised ninth cell visible on a concentrated book (B half, p8: mean "
             "> 0, t >= 2, corr_s > 0.7) and ranked <= 10 among the B-half candidates on that book. Else no "
             "launch; the chain stops and hands the numbers to the user (no other book, alpha or pool).",
    smoke="16 updates p32: every update n_valid >= 20, finite loss, pool + cache selftests, receipt -> PC; else "
          "stop (code bug, not evidence)",
    pc="p8, budget = --updates. Round on B: frozen = top 10 of the 50-entry hall of fame by B-half marginal net "
       "on rank_book. No frozen formula with corr_s(B) > 0.7 -> STOP, report 'PC not found' (the ruler and the "
       "rank were checked before launch, so this is the trainer). Else NULL with the same budget.",
    null="max B-half rank_book marginal net over the NULL frozen set = the bar. PC found = a PC frozen formula "
         "with corr_s > 0.7 AND B score > bar. Not found -> STOP, report. Found -> REAL, same budget.",
    real="REAL frozen set; formulas with B score > NULL bar = CR1b candidates (FROZEN_CR1B.json), reported "
         "to the user; DEV not opened by this chain. None above the bar -> CR1b negative on V4, reported. "
         "Either way the chain stops after REAL (QuantMind 10-14 point 3).",
    veto="found and not found lead to different next steps (REAL vs stop + trainer report); 'not found' is "
         "attributable to the trainer because the plant was checked visible AND ranked <= 10 before launch "
         "(precheck). Veto rule not hit.")


PREREG = dict(written="2026-10-14 (77o), before any CR1b training number; see CR1B_ALPHA.json prereg for the "
                      "pre-run checks", p_pc_found_given_launch=0.35, p_real_above_null_bar_given_found=0.1,
              why="the residual removes the premium / momentum reweights that filled CR1's top 10, so the plant is "
                  "a larger share of what is left to reward; but the training signal is still the full book on "
                  "16 tiles at +8 bp/day of the plant alone, and the policy collapsed by u130 in CR1.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_name")
    ap.add_argument("--mode", required=True, choices=Q.MODES)
    ap.add_argument("--updates", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--tiles", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--ent-coef", type=float, default=0.03)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--tile-cache", type=int, default=1, choices=(0, 1))
    args = ap.parse_args()
    cal = json.load(open(Q.ALPHA_JSON, encoding="utf-8"))
    if cal.get("rc") != 0 or not cal.get("rank_book"):
        raise SystemExit("pre-run check failed in %s: rc %s rank_book %s; not launched"
                         % (Q.ALPHA_JSON, cal.get("rc"), cal.get("rank_book")))
    out_dir = os.path.join(Q.A_ROOT, args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    t_build = time.time()

    from train_basin_riskpg_v1 import make_basin_model, apply_output_bias
    import model_core.alphagpt as alphagpt_mod
    from model_core.config import ModelConfig
    from model_core.vocab import FormulaVocab
    ModelConfig.DEVICE = device
    vm = Q.make_vm()
    ctx = Q.build(args.mode, device, vm)
    A = list(ctx["A"])
    if len(A) < args.tiles:
        raise SystemExit("A half holds %d tiles, need %d" % (len(A), args.tiles))
    torch.manual_seed(args.seed)
    op_names = (list(OPERATOR_NAMES) + [f"W{w}" for w in M.WINDOWS] + ["END"] + M.HOLD_NAMES
                + list(M.BASIN_TOKEN_NAMES))
    model, fvocab = make_basin_model(alphagpt_mod=alphagpt_mod, ModelConfig=ModelConfig, FormulaVocab=FormulaVocab,
                                     feature_names=list(M.TERMINALS), operator_token_names=op_names,
                                     max_len=M.MAX_PROGRAM_LEN, device=device)
    apply_output_bias(model, len(M.TERMINALS), len(OPERATOR_NAMES), len(M.WINDOWS), OP_BIAS, END_BIAS)
    counts = M.quota(args.batch)
    tables = [build_legality_table(vm, max_len=M.MAX_PROGRAM_LEN, available_terminals=M.basin_terms(b),
                                   available_windows=M.WINDOWS).to(device) for b in M.BASIN_NAMES]
    groups = list(zip(M.BASIN_NAMES, counts))
    gself = G.selftest(vm)
    if not gself["ok"]:
        raise SystemExit("grammar selftest failed")
    pool = make_pool(ctx, out_dir, args.workers)
    pst = selftest_pool(ctx, vm, pool, device)
    print("pool selftest %s" % json.dumps(pst), flush=True)
    if not pst["ok"]:
        raise SystemExit("pool selftest failed: pool result != in-process result")
    cst = selftest_cache(ctx, vm, pool, device)
    use_cache = bool(args.tile_cache and cst["ok"])
    print("cache selftest %s -> cache %s" % (json.dumps(cst), "on" if use_cache else "OFF"), flush=True)
    mods = modules()
    gate = cal["gate"][cal["rank_book"]]
    manifest = dict(schema="CR1B_LAUNCH_MANIFEST_V1", run=args.out_name,
                    order="QuantMind 10-14 CR1b approval (V4_COMBO_V1_CR1 s3)",
                    contract="XS_BASIN_TIME_CONTRACT_V4_20261013", args=vars(args), modules=mods,
                    python=sys.executable, conventions=Q.__doc__, trainer=__doc__, decisions=DECISIONS,
                    prereg=PREREG,
                    world=dict(mode=args.mode, grid=X.GRID, grid_sha16=Q.sha12f(X.GRID),
                               A=[str(ctx["ts"][s]) for s in A], B=[str(ctx["ts"][s]) for s in ctx["B"]],
                               combo_fit=ctx["fit"], s_perp_sha16=ctx["s_perp_sha16"],
                               alpha=None if args.mode in ("real", "null") else Q.load_alpha(args.mode, ctx["s_perp"])),
                    precheck=dict(alpha_json=Q.ALPHA_JSON, rank_book=cal["rank_book"], rc=cal["rc"],
                                  plant_rank=gate["rank"], plant_rank_of=gate["of"], plant_mean_bpd=gate["mean"],
                                  plant_t=gate["t"], plant_corr_s=gate["corr_s"], tenth_bpd=gate["tenth"],
                                  null_pool_max_bpd=gate["null_pool_max"], gate_all=cal["gate"],
                                  orth=cal["orth"], alpha=dict(p8=cal["p8"]["alpha"], p32=cal["p32"]["alpha"])),
                    train_book="full", rank_book=cal["rank_book"], holds=[168], batch=args.batch, quota=counts,
                    tiles_per_update=args.tiles, pool_selftest=pst, cache_selftest=cst, tile_cache=use_cache,
                    grammar_selftest_ok=True,
                    vocab_size=fvocab.size, model_params=sum(p.numel() for p in model.parameters()),
                    build_seconds=round(time.time() - t_build, 1), dev_read=False, oos_read=False,
                    written_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    ck = None
    if args.resume:
        ck_path = TB._latest_checkpoint(out_dir) if args.resume == "auto" else args.resume
        if not ck_path or not os.path.exists(ck_path):
            raise SystemExit("--resume %s: no checkpoint in %s" % (args.resume, out_dir))
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        free = {"updates", "max_hours", "resume", "checkpoint_every", "workers", "tile_cache"}
        diff = {k: (ck["args"].get(k), v) for k, v in vars(args).items() if k not in free and ck["args"].get(k) != v}
        if diff:
            raise SystemExit("--resume refused, args differ: %s" % diff)
        if ck["modules"] != mods:
            raise SystemExit("--resume refused, code differs: %s vs %s" % (ck["modules"], mods))
        manifest["resumed_from"] = dict(checkpoint=ck_path, update=int(ck["update"]))
    json.dump(manifest, open(os.path.join(out_dir, "LAUNCH_MANIFEST.json" if ck is None else
                                          "RESUME_MANIFEST_%s.json" % time.strftime("%Y%m%dT%H%M%S")), "w"),
              indent=1, default=str)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    starts_rng = np.random.default_rng(args.seed)
    hof, n_eval, inval = {}, 0, collections.Counter()
    start_update, elapsed_prev, fmode = 0, 0.0, "w"
    if ck is not None:
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        hof = {tuple(b): e for b, e in ck["hof"]}
        n_eval, inval = int(ck["n_eval"]), collections.Counter(ck["inval"])
        starts_rng.bit_generator.state = ck["starts_rng_state"]
        torch.set_rng_state(ck["torch_rng_state"])
        if device.type == "cuda" and ck.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng_state"], device)
        start_update, elapsed_prev, fmode = int(ck["update"]) + 1, float(ck["elapsed_s"]), "a"
        for fn in ("BATCH_LOG.jsonl", "FORMULAS.jsonl"):
            TB._truncate_jsonl(os.path.join(out_dir, fn), int(ck["update"]))
        print("resumed from update %d" % start_update, flush=True)
    log_fh = open(os.path.join(out_dir, "BATCH_LOG.jsonl"), fmode, encoding="utf-8")
    form_fh = open(os.path.join(out_dir, "FORMULAS.jsonl"), fmode, encoding="utf-8")
    basin_of = sum(([b] * c for b, c in groups), [])
    t0, t_sess, stopped, vram = time.time() - elapsed_prev, time.time(), False, None
    tcache = {}

    for update in range(start_update, args.updates):
        tm, t_ = {}, time.time()
        t_u = t_
        draw = np.sort(starts_rng.choice(len(A), size=args.tiles, replace=False))
        starts = [A[int(i)] for i in draw]
        slices = M.tile_slices(ctx["feats_pad"], starts, device)
        tm["slices"], t_ = time.time() - t_, time.time()
        batch = TB.sample_batch_with_hold_basins(model, vm, tables, counts, device=device)
        tm["sample"], t_ = time.time() - t_, time.time()
        n = args.batch
        bodies, keys, why = [], [None] * n, [None] * n
        uniq = []
        for i, toks in enumerate(batch.formulas):
            if TB.hold_of(vm, list(toks)) != 168:
                raise RuntimeError("hold mask leaked")
            body = tuple(TB.strip_hold(vm, list(toks)))
            bodies.append(body)
            g = G.check(vm, list(body))
            if g is not None:
                why[i] = "GRAMMAR_" + str(g)
                continue
            keys[i] = body
            if body not in uniq:                          # memo: once per update
                uniq.append(body)
        res, bad, hits, tm["vm"], tm["ruler_wait"] = score_batch(ctx, vm, pool, tcache, uniq, starts, slices,
                                                                 use_cache)
        t_ = time.time()
        rew, gro, cor, crw = (np.full(n, np.nan) for _ in range(4))
        valid = np.zeros(n, dtype=bool)
        for i in range(n):
            k = keys[i]
            if k is not None and k in bad:
                why[i] = bad[k]
            if why[i] is not None:
                inval[why[i].split("_")[0]] += 1
                continue
            r = res[k]
            rew[i], gro[i], cor[i], crw[i] = float(np.mean(r["net"])), float(np.mean(r["gross"])), r["corr"], r["corr_raw"]
            valid[i] = True
        nv = int(valid.sum())
        skip = "n_valid=%d<20" % nv if nv < 20 else ("reward_std==0" if rew[valid].std() == 0.0 else None)
        upd, fam_ent = {}, [None] * len(groups)
        if skip is None:
            adv = D.zfam(rew, valid, counts)
            adv[~valid] = -1.0
            upd, fam_ent = TB.dense_update(model, optimizer, batch, np.arange(n), adv, groups, device,
                                           ent_coef=args.ent_coef)
        tm["update"], t_ = time.time() - t_, time.time()
        # hall of fame (A half only): every valid body, mean / best marginal full net over its updates
        counted = set()
        for i in np.flatnonzero(valid):
            e = hof.setdefault(bodies[i], dict(basin=basin_of[i], expr=vm.describe(list(bodies[i]), list(M.TERMINALS)),
                                              n=0, s=0.0, best=-1e9, corr=float(cor[i])))
            if keys[i] not in counted:
                counted.add(keys[i])
                e["n"] += 1
                e["s"] += float(rew[i])
                e["best"] = max(e["best"], float(rew[i]))
        if len(hof) > 5000:
            hof = dict(sorted(hof.items(), key=lambda kv: -kv[1]["s"] / max(kv[1]["n"], 1))[:3000])
        for i in range(n):
            form_fh.write(json.dumps(dict(update=update, basin=basin_of[i], expr=vm.describe(list(bodies[i]), list(M.TERMINALS)),
                                          reward_bpd=round(Q.bpd(rew[i]), 4) if valid[i] else None,
                                          gross_bpd=round(Q.bpd(gro[i]), 4) if valid[i] else None,
                                          corr_s=round(float(cor[i]), 4) if valid[i] and np.isfinite(cor[i]) else None,
                                          corr_raw=round(float(crw[i]), 4) if valid[i] and np.isfinite(crw[i]) else None,
                                          live_frac=round(res[keys[i]]["lf"], 4) if valid[i] else None,
                                          invalid=why[i])) + "\n")
        n_eval += n
        fam = {}
        off = 0
        for k, (b, c) in enumerate(groups):
            m = valid[off:off + c]
            sl = slice(off, off + c)
            top = int(off + np.nanargmax(np.where(m, rew[sl], -np.inf))) if m.any() else None
            mc = collections.Counter(bodies[off:off + c]).most_common(1)[0]
            fam[b] = dict(n_valid=int(m.sum()), reward_bpd=round(Q.bpd(np.mean(rew[sl][m])), 4) if m.any() else None,
                          corr_s=round(float(np.nanmean(cor[sl][m])), 4) if m.any() else None,
                          entropy=round(fam_ent[k], 4) if fam_ent[k] is not None else None,
                          top=None if top is None else dict(expr=vm.describe(list(bodies[top]), list(M.TERMINALS)),
                                                            reward_bpd=round(Q.bpd(rew[top]), 4),
                                                            corr_s=round(float(cor[top]), 4)),
                          mode=dict(expr=vm.describe(list(mc[0]), list(M.TERMINALS)), share=round(mc[1] / c, 4)))
            off += c
        tm["log"] = time.time() - t_
        sec = time.time() - t_u
        if device.type == "cuda":
            vram = torch.cuda.max_memory_allocated() / 2 ** 30
        row = dict(update=update, seconds=round(sec, 2), times={k: round(v, 2) for k, v in tm.items()},
                   n_valid=nv, n_unique=len(uniq), n_scored=len(res), skipped=skip, cache_hits=hits,
                   cache_hit_frac=round(hits / max(len(uniq) * len(starts), 1), 4), cache_size=len(tcache),
                   tiles=starts,
                   reward_mean_bpd=round(Q.bpd(np.mean(rew[valid])), 4) if nv else None,
                   reward_pos_frac=round(float((rew[valid] > 0).mean()), 4) if nv else None,
                   hof_size=len(hof), families=fam, vram_gib=round(vram, 3) if vram else None,
                   **{k: upd.get(k) for k in ("pg_loss", "entropy_loss", "gradient_norm", "mean_entropy",
                                              "skipped_nonfinite_update") if k in upd})
        log_fh.write(json.dumps(row) + "\n")
        log_fh.flush()
        form_fh.flush()
        json.dump(dict(pid=os.getpid(), update=update, updates_total=args.updates, seconds=round(sec, 2),
                       updated=time.strftime("%Y-%m-%dT%H:%M:%S%z")), open(os.path.join(out_dir, "HEARTBEAT.json"), "w"))
        if update < start_update + 4 or (update + 1) % PROGRESS_EVERY == 0 or update == args.updates - 1:
            eta = (time.time() - t_sess) / (update + 1 - start_update) * (args.updates - update - 1)
            print("upd %3d %5.1fs %s valid %d uniq %d | %s | ent %s gn %s vram %s eta %s"
                  % (update, sec, row["times"], nv, row["n_unique"],
                     " ".join("%s r%s c%s e%s" % (b, f["reward_bpd"], f["corr_s"], f["entropy"]) for b, f in fam.items()),
                     None if upd.get("mean_entropy") is None else round(upd["mean_entropy"], 3),
                     None if upd.get("gradient_norm") is None else round(upd["gradient_norm"], 2),
                     round(vram, 2) if vram else "-", TB._hms(eta)), flush=True)
            for b, f in fam.items():
                print("  %s top %s | mode %.2f %s" % (b, (f["top"] or {}).get("expr", "-")[:100],
                                                      f["mode"]["share"], f["mode"]["expr"][:80]), flush=True)
        over = bool(args.max_hours) and (time.time() - t0) >= args.max_hours * 3600.0
        if (update + 1) % args.checkpoint_every == 0 or over or update == args.updates - 1:
            TB._save_checkpoint(os.path.join(out_dir, "checkpoint_%04d.pt" % update), model=model.state_dict(),
                                optimizer=optimizer.state_dict(), update=update, args=vars(args), modules=mods,
                                hof=list(hof.items()), n_eval=n_eval, inval=dict(inval),
                                starts_rng_state=starts_rng.bit_generator.state, torch_rng_state=torch.get_rng_state(),
                                cuda_rng_state=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                                elapsed_s=time.time() - t0)
        if over and update != args.updates - 1:
            stopped = True
            json.dump(dict(reason="max_hours", last_update=update, resume="--resume auto"),
                      open(os.path.join(out_dir, "STOPPED.json"), "w"))
            print("max_hours reached after update %d; checkpointed, stopping" % update, flush=True)
            break
    log_fh.close()
    form_fh.close()
    pool.close()
    pool.join()
    if stopped:
        return
    top = sorted(hof.items(), key=lambda kv: -kv[1]["s"] / kv[1]["n"])[:300]
    json.dump(dict(schema="CR1B_HOF_V1", run=args.out_name, mode=args.mode, rule="mean marginal full-book net over "
                   "the updates a body was sampled in (A tiles only); top 300", entries=[
                       dict(body=list(b), basin=e["basin"], expr=e["expr"], n=e["n"], mean_bpd=round(Q.bpd(e["s"] / e["n"]), 4),
                            best_bpd=round(Q.bpd(e["best"]), 4), corr_s=round(e["corr"], 4)) for b, e in top]),
              open(os.path.join(out_dir, "HOF.json"), "w", encoding="utf-8"), indent=1)
    json.dump(dict(schema="CR1B_TRAIN_RECEIPT_V1", run=args.out_name, mode=args.mode, args=vars(args), modules=mods,
                   rank_book=cal["rank_book"], tile_cache=use_cache,
                   n_eval=n_eval, invalid=dict(inval), hof_size=len(hof), peak_vram_gib=round(vram, 3) if vram else None,
                   compute_seconds=round(time.time() - t0, 1), dev_read=False, oos_read=False,
                   created=time.strftime("%Y-%m-%dT%H:%M:%S%z")),
              open(os.path.join(out_dir, "TRAIN_RECEIPT_V1.json"), "w"), indent=1)
    print("done updates %d seconds %.0f" % (args.updates, time.time() - t0), flush=True)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
