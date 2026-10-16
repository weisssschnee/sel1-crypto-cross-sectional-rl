"""CR1 trainer (order V4_COMBO_V1_CR1 s3; ADR-0010 decision 3; contract V4). New file. The sampler, grammar
mask, policy-gradient step, elite pool and checkpoint plumbing are m2h_b5_trainer_dbg_memo's functions,
imported unchanged (that file is not modified); the world and the reward are cr1_common's.

  reward   per formula: mean over the drawn A tiles of the marginal full-book net log NAV (flat6) of
           combo + f over the combo alone (cr1_common docstring). Cost is in the reward because a ninth
           cell changes the combo's turnover; the combo's own cost is subtracted out.
  update   16 of the 30 A tiles per update (B tiles never enter training); batch 128 (T1/T2/T3 quota),
           H168 only; within-family z +-3 over valid, invalid -1, entropy 0.03 (dbg_memo --select full).
  speed    memo: each (body, hold) scored once per update; the VM runs on the GPU and hands the tile
           signals to --workers CPU processes (1 thread each) as soon as they are ready, so the ruler
           overlaps the VM. Step times are logged per update. Selftest at start: pool result ==
           in-process result, bit for bit, for the plant formula (production path).

    python -I cr1_trainer.py OUT --mode p8 --updates 256 [--resume auto]
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
import cr1_common as Q
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
    return dict(cr1_trainer=TB._sha(os.path.abspath(__file__)), cr1_common=TB._sha(Q.__file__),
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
    d = max(d, abs(a["corr"] - b["corr"]))
    return dict(ok=bool(d <= 1e-12), max_abs_diff=float(d),
                full_net_bpd=[round(Q.bpd(x), 4) for x in a["books"]["full"]["net"]],
                k10_net_bpd=[round(Q.bpd(x), 4) for x in a["books"]["conc_k10"]["net"]], corr=round(a["corr"], 4))
DECISIONS = dict(
    smoke="16 updates p32: every update n_valid >= 20, finite loss, pool selftest bit-equal, receipt -> PC; "
          "else stop (code bug, not evidence)",
    pc="p8, budget = --updates. Round on B: frozen = top 10 of the hall of fame by B-half marginal K10 net. "
       "No frozen formula with corr_s(B) > 0.7 -> STOP, report 'PC not found' (trainer; the ruler is checked "
       "in CR1_ALPHA.json). Else NULL with the same budget.",
    null="max B-half marginal K10 net over the NULL frozen set = the bar. PC found = a PC frozen formula with "
         "corr > 0.7 AND B score > bar. Not found -> STOP, report. Found -> REAL, same budget.",
    real="REAL frozen set; formulas with B score > NULL bar = CR1 candidates (FROZEN_CR1.json), reported to "
         "the user; DEV not opened by this chain. None above the bar -> CR1 negative on V4, reported.",
    veto="found and not found lead to different next steps (REAL vs stop+trainer fix); a 'not found' is "
         "attributable because the ruler check (plant formula as the ninth cell, p8, B half, K10 t >= 2) "
         "passed before PC. Veto rule not hit.")


PREREG = dict(written="2026-10-14 (77o), before any CR1 number", p_pc_found=0.55, p_real_above_null_bar=0.15,
              why="B5 found p32 but not p8 with the single-formula reward; the marginal reward removes what the "
                  "combo already holds, so an orthogonal p8 is a larger share of the reward signal than in B5, but "
                  "16 tiles x full book is noisy at 8 bp/day. Real: V4 COMBO_V1 itself is ~0, and B5's NULL max "
                  "already beat its real top 5.")


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
    args = ap.parse_args()
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
    mods = modules()
    manifest = dict(schema="CR1_LAUNCH_MANIFEST_V1", run=args.out_name, order="V4_COMBO_V1_CR1 s3",
                    contract="XS_BASIN_TIME_CONTRACT_V4_20261013", args=vars(args), modules=mods,
                    python=sys.executable, conventions=Q.__doc__, trainer=__doc__, decisions=DECISIONS,
                    prereg=PREREG,
                    world=dict(mode=args.mode, grid=X.GRID, grid_sha16=Q.sha12f(X.GRID),
                               A=[str(ctx["ts"][s]) for s in A], B=[str(ctx["ts"][s]) for s in ctx["B"]],
                               combo_fit=ctx["fit"], s_perp_sha16=ctx["s_perp_sha16"],
                               alpha=None if args.mode in ("real", "null") else Q.load_alpha(args.mode, ctx["s_perp"])),
                    train_book="full", hof_book="conc_k10", holds=[168], batch=args.batch, quota=counts,
                    tiles_per_update=args.tiles, pool_selftest=pst, grammar_selftest_ok=True,
                    vocab_size=fvocab.size, model_params=sum(p.numel() for p in model.parameters()),
                    build_seconds=round(time.time() - t_build, 1), dev_read=False, oos_read=False,
                    written_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    ck = None
    if args.resume:
        ck_path = TB._latest_checkpoint(out_dir) if args.resume == "auto" else args.resume
        if not ck_path or not os.path.exists(ck_path):
            raise SystemExit("--resume %s: no checkpoint in %s" % (args.resume, out_dir))
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        free = {"updates", "max_hours", "resume", "checkpoint_every", "workers"}
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
        pend, bad, lf = {}, {}, {}
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
            if body in pend or body in bad:              # memo: once per update
                continue
            try:
                sigs, lf[body] = Q.tile_sigs(ctx, vm, list(body), starts, slices)
            except (M.ZeroCoverage, CandidateLocalInvalid) as e:
                bad[body] = str(e)
                continue
            pend[body] = pool.apply_async(Q.worker_task, ((body, Q.pack(sigs), starts, (TRAIN_BOOK,), CORR_TILES),))
        tm["vm"], t_ = time.time() - t_, time.time()
        res = {}
        for body, ar in pend.items():
            _, r = ar.get()
            if isinstance(r, str):
                bad[body] = r[len("INVALID:"):]
            else:
                res[body] = r
        tm["ruler_wait"], t_ = time.time() - t_, time.time()
        rew, gro, cor = (np.full(n, np.nan) for _ in range(3))
        valid = np.zeros(n, dtype=bool)
        for i in range(n):
            k = keys[i]
            if k is not None and k in bad:
                why[i] = bad[k]
            if why[i] is not None:
                inval[why[i].split("_")[0]] += 1
                continue
            r = res[k]["books"]["full"]
            rew[i], gro[i], cor[i] = float(np.mean(r["net"])), float(np.mean(r["gross"])), res[k]["corr"]
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
        for i in np.flatnonzero(valid):
            e = hof.setdefault(bodies[i], dict(basin=basin_of[i], expr=vm.describe(list(bodies[i]), list(M.TERMINALS)),
                                              n=0, s=0.0, best=-1e9, corr=float(cor[i])))
            if keys[i] in res and res[keys[i]].get("_counted") != update:
                res[keys[i]]["_counted"] = update
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
                                          live_frac=round(lf.get(keys[i], np.nan), 4) if valid[i] else None,
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
                   n_valid=nv, n_unique=len({k for k in keys if k is not None}), n_scored=len(res), skipped=skip,
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
    json.dump(dict(schema="CR1_HOF_V1", run=args.out_name, mode=args.mode, rule="mean marginal full-book net over "
                   "the updates a body was sampled in (A tiles only); top 300", entries=[
                       dict(body=list(b), basin=e["basin"], expr=e["expr"], n=e["n"], mean_bpd=round(Q.bpd(e["s"] / e["n"]), 4),
                            best_bpd=round(Q.bpd(e["best"]), 4), corr_s=round(e["corr"], 4)) for b, e in top]),
              open(os.path.join(out_dir, "HOF.json"), "w", encoding="utf-8"), indent=1)
    json.dump(dict(schema="CR1_TRAIN_RECEIPT_V1", run=args.out_name, mode=args.mode, args=vars(args), modules=mods,
                   n_eval=n_eval, invalid=dict(inval), hof_size=len(hof), peak_vram_gib=round(vram, 3) if vram else None,
                   compute_seconds=round(time.time() - t0, 1), dev_read=False, oos_read=False,
                   created=time.strftime("%Y-%m-%dT%H:%M:%S%z")),
              open(os.path.join(out_dir, "TRAIN_RECEIPT_V1.json"), "w"), indent=1)
    print("done updates %d seconds %.0f" % (args.updates, time.time() - t0), flush=True)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
