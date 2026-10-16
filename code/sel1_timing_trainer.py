"""SEL1 timing-layer risk-PG trainer: copy of m1_stage1/m2_trainer.py (the locked original is not touched),
input changed to a trigger-event table (sel1_timing_common, order XS_SEL1 深夜 section).

Changed against m2_trainer:
  input    --events: one row per SEL1 frozen-book holding spell (coin, trigger hour, rows, direction, beta).
           Each update takes one TRAIN month that has events, splits its events at random into SELECT and
           HOLDOUT halves (replacing m2's disjoint windows), and runs the VM once per formula on the whole
           extended month (1440 min of the previous month in front).
  action   the formula's sign per minute = in / out / stay inside each event window, executed LAG = 2 min
           later; |position| <= POS_CAP; no HOLD token (the event fixes the window).
  reward   mean over HOLDOUT events of flat6 log NAV minus DD_LAMBDA x the event's max drawdown, per event-day
           (bp/day), settled by m1_ruler.settle_multi on [coin, BTC hedge = -beta x position], raw-ms funding,
           Rule 0. The always-in book (SEL1's own entry, held to its exit) is logged beside it, never rewarded.
  terms    the 16 columns the TRAIN shards carry (m1_common.SOURCE_COLUMNS); m2's 17-column guard would refuse
           every shard on disk.
Not changed: the pinned novelty_weighted_risk_update, the AlphaGPT policy and output-head bias, the stack VM,
the optimizer, advantage = reward / std with invalid rows at min - 1, n_valid >= 20, elite re-check, behaviour
dedup, checkpoint / resume refusal on any argument or module-sha difference.

Training waits for a frozen SEL1 candidate past its gate and the user's approval; --no-step runs one scoring
pass with no parameter step and no checkpoint (plumbing and per-formula time only).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import m1_common as C
import m1_ruler as R
import sel1_timing_common as TC

ROOT = C.ROOT
OUT_ROOT = os.path.join(ROOT, "artifacts", "m2h_v1", "XS_SEL1", "TIMING")
ELITE_RECHECK = 32
REWARD_FLOOR = C.REWARD_FLOOR
LEARNING_RATE = 5e-4
OP_BIAS = 1.5
END_BIAS = 2.0
CHECKPOINT_EVERY = 16
MIN_EVENTS = 4                    # a month needs >= 2 SELECT + 2 HOLDOUT events
BEHAV_TOL = 1e-5


def behaviour_keep(vecs, groups, tol=BEHAV_TOL):
    vecs = [np.asarray(v, dtype=np.float64) for v in vecs]
    groups = np.asarray(groups)
    kept = []
    for i, v in enumerate(vecs):
        if not any(groups[j] == groups[i] and vecs[j].shape == v.shape and float(np.abs(vecs[j] - v).max()) <= tol
                   for j in kept):
            kept.append(i)
    return kept


def _truncate_jsonl(path, last_update):
    if not os.path.exists(path):
        return 0
    keep, dropped = [], 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            if json.loads(line).get("update", -1) <= last_update:
                keep.append(line if line.endswith("\n") else line + "\n")
            else:
                dropped += 1
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(keep)
    return dropped


def _save_checkpoint(path, **state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _latest_checkpoint(out_dir):
    import glob
    cks = sorted(glob.glob(os.path.join(out_dir, "checkpoint_*.pt")))
    return cks[-1] if cks else None


def _sha(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def _hms(sec):
    sec = int(max(sec, 0))
    return "%d:%02d:%02d" % (sec // 3600, (sec % 3600) // 60, sec % 60)


def eval_one(vm, body, x, sh, sel_w, hold_w, lam, invalid_cls):
    """One formula on one month: (holdout reward, select reward, holdout per-event flat6 vector, diag).
    x = (select slices, holdout slices) from TC.event_slices."""
    seg_s = TC.event_signals(vm, body, x[0], sel_w, sh)
    seg_h = TC.event_signals(vm, body, x[1], hold_w, sh)
    sc_h, sc_s = TC.score_events(seg_h, hold_w, sh), TC.score_events(seg_s, sel_w, sh)
    sm = TC.summary(sc_h, lam)
    diag = dict(in_frac=sm["in_frac"], turn_per_event=sm["turn_per_event"], dd_mean=sm["dd_mean"],
                flat6_bpd=sm["flat6_bpd"], cost0_bpd=sm["cost0_bpd"])
    return TC.reward(sc_h, lam), TC.reward(sc_s, lam), np.array([r["flat6"] for r in sc_h]), diag


def month_windows(ev, months, shard_dir, log):
    """Per month: the event windows on the extended shard and the drop counts (shards loaded one at a time)."""
    out, drops = {}, collections.Counter()
    for m in months:
        sh = TC.ext_shard(m, shard_dir)
        w, d = TC.windows(ev[ev["month"] == m], sh)
        drops.update(d)
        if len(w) >= MIN_EVENTS:
            out[m] = w
        else:
            drops["MONTH_TOO_FEW_%d" % len(w)] += len(w)
        TC.drop_cache(keep=())
    log("events: %d usable in %d months, dropped %s" % (sum(len(v) for v in out.values()), len(out), dict(drops)))
    return out, dict(drops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_name")
    ap.add_argument("--events", required=True, help="event table (.csv / .parquet), sel1_timing_common.EVENT_COLS")
    ap.add_argument("--shard-dir", default=TC.SHARD_DIR)
    ap.add_argument("--updates", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dd-lambda", type=float, default=TC.DD_LAMBDA)
    ap.add_argument("--elite-recheck", type=int, default=ELITE_RECHECK)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-events", type=int, default=32,
                    help="events per half per update (random subset of the month's events; VM slices are built "
                         "once per update and held for the whole batch)")
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--no-step", action="store_true",
                    help="score the batch, no parameter step, no checkpoint (plumbing / timing only)")
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(int(args.threads))

    out_dir = os.path.join(OUT_ROOT, args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)
    log = lambda s: print(s, flush=True)

    sys.path.insert(0, os.path.join(ROOT, "src"))
    sys.path.insert(0, os.path.join(ROOT, "runtime"))
    LEGACY_ROOT = r"D:\v4r_groot\Chengbo\workspaces\alphagpt-rl-dl-base"
    for p in (LEGACY_ROOT, os.path.join(LEGACY_ROOT, "experiments")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from train_basin_riskpg_v1 import make_basin_model, apply_output_bias
    import model_core.alphagpt as alphagpt_mod
    from model_core.config import ModelConfig
    from model_core.vocab import FormulaVocab
    from experiments.rl_v2_autonomous_novelty_search_v4 import novelty_weighted_risk_update
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import XsTsCsStackVM, OPERATOR_NAMES, CandidateLocalInvalid
    import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vm_mod
    vm_mod.BLOCK_ELEMS = 1 << 25
    vm_mod.EMA_KERNEL_CACHE_MAX = 1
    from okx_alpha_discovery.adapters.xs_tscs_sampler_v2 import build_legality_table, sample_batch
    from oos_guard import guard_shard_dir, OOSViolation
    ModelConfig.DEVICE = device
    invalid_types = (CandidateLocalInvalid, TC.Invalid)

    ev = TC.read_events(args.events)
    months = sorted(m for m in set(ev["month"]) if m in TC.TRAIN_MONTHS)
    if not months:
        raise SystemExit("no event falls in a TRAIN 2024-25 shard month")
    try:
        guard_shard_dir(args.shard_dir, "TRAIN", expect_months=list(TC.TRAIN_MONTHS))
    except OOSViolation as exc:
        raise SystemExit(f"OOS GUARD REFUSED: {exc}")

    vm = XsTsCsStackVM(len(TC.TERMS), windows=C.WINDOWS)
    torch.manual_seed(args.seed)
    op_names = list(OPERATOR_NAMES) + [f"W{w}" for w in C.WINDOWS] + ["END"]
    model, fvocab = make_basin_model(alphagpt_mod=alphagpt_mod, ModelConfig=ModelConfig, FormulaVocab=FormulaVocab,
                                     feature_names=list(TC.TERMS), operator_token_names=op_names,
                                     max_len=C.MAX_PROGRAM_LEN, device=device)
    apply_output_bias(model, len(TC.TERMS), len(OPERATOR_NAMES), len(C.WINDOWS), OP_BIAS, END_BIAS)
    if fvocab.size != vm.token_count:
        raise SystemExit(f"vocab {fvocab.size} != vm tokens {vm.token_count}")

    manifest = dict(schema="SEL1_TIMING_LAUNCH_MANIFEST_V1", python=sys.executable, entry=os.path.abspath(__file__),
                    args=vars(args), months=months,
                    modules=dict(trainer=_sha(os.path.abspath(__file__)), timing_common=_sha(TC.__file__),
                                 m1_ruler=_sha(R.__file__), m1_common=_sha(C.__file__)),
                    events=dict(path=args.events, sha=_sha(args.events), rows=int(len(ev)),
                                rows_train_months=int(ev["month"].isin(months).sum())),
                    world=dict(terms=list(TC.TERMS), lag_min=TC.LAG, pos_cap=TC.POS_CAP, dd_lambda=args.dd_lambda,
                               hour_close_min=TC.HOUR_CLOSE_MIN, costs=list(TC.COSTS), exit_search=TC.EXIT_SEARCH,
                               rule0="settle_multi freezes through gaps", hedge="BTC = -beta(trigger hour) x position"),
                    reward="bp per HOLDOUT event-day of flat6 log NAV - dd_lambda x per-event max drawdown",
                    no_step=bool(args.no_step), written_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    ck, ck_path = None, None
    if args.resume:
        ck_path = _latest_checkpoint(out_dir) if args.resume == "auto" else args.resume
        if not ck_path or not os.path.exists(ck_path):
            raise SystemExit(f"--resume {args.resume}: no checkpoint in {out_dir}")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        free = {"updates", "resume", "checkpoint_every", "manifest_only", "threads"}
        diff = {k: (ck["args"].get(k), v) for k, v in vars(args).items() if k not in free and ck["args"].get(k) != v}
        if diff:
            raise SystemExit(f"--resume refused, arguments differ: {diff}")
        if ck["modules"] != manifest["modules"]:
            raise SystemExit(f"--resume refused, code differs: {ck['modules']} vs {manifest['modules']}")
        manifest["resumed_from"] = dict(checkpoint=ck_path, update=int(ck["update"]))
    man_name = "LAUNCH_MANIFEST.json" if ck is None else "RESUME_MANIFEST_%s.json" % time.strftime("%Y%m%dT%H%M%S")
    json.dump(manifest, open(os.path.join(out_dir, man_name), "w"), indent=1)
    if args.manifest_only:
        return

    wins, drops = month_windows(ev, months, args.shard_dir, log)
    months = sorted(wins)
    if not months:
        raise SystemExit(f"no month with >= {MIN_EVENTS} usable events; drops {drops}")
    table = build_legality_table(vm, max_len=C.MAX_PROGRAM_LEN, available_windows=C.WINDOWS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    split_rng = np.random.default_rng(args.seed)
    elite_pool, invalid_reasons, n_eval, start_update, elapsed_prev = {}, collections.Counter(), 0, 0, 0.0
    if ck is not None:
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        elite_pool, n_eval = ck["elite_pool"], int(ck["n_eval"])
        invalid_reasons = collections.Counter(ck["invalid_reasons"])
        split_rng.bit_generator.state = ck["split_rng_state"]
        torch.set_rng_state(ck["torch_rng_state"])
        if device.type == "cuda" and ck.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng_state"], device)
        start_update, elapsed_prev = int(ck["update"]) + 1, float(ck["elapsed_s"])
        for fn in ("BATCH_LOG.jsonl", "FORMULAS.jsonl"):
            _truncate_jsonl(os.path.join(out_dir, fn), int(ck["update"]))
    fmode = "a" if ck is not None else "w"
    log_fh = open(os.path.join(out_dir, "BATCH_LOG.jsonl"), fmode, encoding="utf-8")
    formulas_fh = open(os.path.join(out_dir, "FORMULAS.jsonl"), fmode, encoding="utf-8")
    t0 = time.time() - elapsed_prev
    t_sess0 = time.time()
    lam = float(args.dd_lambda)
    last_updates = start_update + 1 if args.no_step else args.updates

    for update in range(start_update, last_updates):
        tu = time.time()
        month = months[update % len(months)]
        TC.drop_cache(keep=())
        sh = TC.ext_shard(month, args.shard_dir)
        mw = wins[month]
        perm = split_rng.permutation(len(mw))
        n_sel = min(len(mw) // 2, args.max_events)
        n_hold = min(len(mw) - len(mw) // 2, args.max_events)
        sel_ix, hold_ix = perm[:n_sel], perm[len(mw) // 2:len(mw) // 2 + n_hold]
        sel_w, hold_w = [mw[i] for i in sel_ix], [mw[i] for i in hold_ix]
        base = TC.summary(TC.always_in(hold_w, sh), lam)
        x = (TC.event_slices(sel_w, sh, device), TC.event_slices(hold_w, sh, device))

        batch = sample_batch(model, vm, table, args.batch, device=device)
        tokens_list = batch.formulas
        rewards = np.full(args.batch, REWARD_FLOOR)
        sel_rewards = np.full(args.batch, REWARD_FLOOR)
        vecs, diags = [None] * args.batch, [None] * args.batch
        valid = np.zeros(args.batch, dtype=bool)
        for i, toks in enumerate(tokens_list):
            body = list(toks)
            reason = None
            try:
                r_h, r_s, vec, dg = eval_one(vm, body, x, sh, sel_w, hold_w, lam, invalid_types)
                if not np.isfinite(r_h):
                    raise TC.Invalid("NONFINITE")
                rewards[i], sel_rewards[i], vecs[i], diags[i], valid[i] = r_h, r_s, vec, dg, True
            except invalid_types as exc:
                reason = str(exc)
                invalid_reasons[reason.split("_")[0]] += 1
            formulas_fh.write(json.dumps(dict(
                update=update, month=month, body=body, expr=vm.describe(body, list(TC.TERMS)),
                reward=round(float(rewards[i]), 6),
                reward_select=round(float(sel_rewards[i]), 6) if valid[i] else None,
                **(diags[i] or {}), invalid_reason=reason)) + "\n")
        formulas_fh.flush()
        n_valid = int(valid.sum())
        n_eval += len(tokens_list)

        std = float(rewards[valid].std()) if n_valid else 0.0
        skip = None if n_valid >= 20 and std > 0 else ("n_valid=%d<20" % n_valid if n_valid < 20 else "std==0")
        if args.no_step:
            skip = "no_step"
        upd = {}
        if skip is None:
            scaled = np.empty(args.batch)
            scaled[valid] = rewards[valid] / std
            scaled[~valid] = float(scaled[valid].min()) - 1.0
            upd = novelty_weighted_risk_update(model, optimizer, batch, scaled, np.ones(args.batch), device)

        # elite re-check on this update's HOLDOUT events, then behaviour dedup on the same events
        # pool entries: (body, expr, score, HOLDOUT per-event vector, HOLDOUT event key); vectors compare only
        # inside one key (same events)
        hkey = tuple(int(i) for i in hold_ix)
        recheck = []
        if args.elite_recheck > 0 and elite_pool.get(month):
            kept = []
            for body, expr, old, _, _ in sorted(elite_pool[month], key=lambda e: -e[2])[:args.elite_recheck]:
                try:
                    r, _, vec, _ = eval_one(vm, list(body), x, sh, sel_w, hold_w, lam, invalid_types)
                except invalid_types:
                    continue
                recheck.append(dict(expr=expr, old=round(old, 6), new=round(float(r), 6)))
                kept.append((body, expr, float(r), vec, hkey))
            elite_pool[month] = kept
        order = [int(i) for i in np.argsort(-rewards) if valid[i]]
        cands = list(elite_pool.get(month, [])) + [
            (tuple(tokens_list[i]), vm.describe(list(tokens_list[i]), list(TC.TERMS)), float(rewards[i]), vecs[i],
             hkey) for i in order]
        cands.sort(key=lambda e: -e[2])
        gid = {}
        keep_ix = behaviour_keep([e[3] for e in cands], [gid.setdefault(e[4], len(gid)) for e in cands])
        elite_pool[month] = [cands[j] for j in keep_ix][:200]

        row = dict(update=update, month=month, events_select=len(sel_w), events_holdout=len(hold_w),
                   n_valid=n_valid, n_eval_cum=n_eval, skipped_param_step=skip,
                   reward_mean=round(float(rewards[valid].mean()), 4) if n_valid else None,
                   reward_max=round(float(rewards[valid].max()), 4) if n_valid else None,
                   reward_select_mean=round(float(sel_rewards[valid].mean()), 4) if n_valid else None,
                   reward_std=round(std, 4), always_in=base,
                   in_frac_mean=round(float(np.mean([d["in_frac"] for d in diags if d])), 4) if n_valid else None,
                   elite_rechecked=len(recheck), elite_pool=sum(len(v) for v in elite_pool.values()),
                   sec=round(time.time() - tu, 1),
                   **{k: upd.get(k) for k in ("pg_loss", "entropy_loss", "gradient_norm") if k in upd})
        log_fh.write(json.dumps(row) + "\n")
        log_fh.flush()
        eta = (time.time() - t_sess0) / (update + 1 - start_update) * (last_updates - update - 1)
        log("upd %d %s ev %d/%d valid %d reward %s (always-in %s) %.1fs eta %s"
            % (update, month, len(sel_w), len(hold_w), n_valid, row["reward_mean"], base.get("reward"),
               row["sec"], _hms(eta)))
        if not args.no_step and ((update + 1) % args.checkpoint_every == 0 or update == last_updates - 1):
            _save_checkpoint(os.path.join(out_dir, f"checkpoint_{update:04d}.pt"),
                             model=model.state_dict(), optimizer=optimizer.state_dict(), update=update,
                             args=vars(args), modules=manifest["modules"], elite_pool=elite_pool, n_eval=n_eval,
                             invalid_reasons=dict(invalid_reasons), split_rng_state=split_rng.bit_generator.state,
                             torch_rng_state=torch.get_rng_state(),
                             cuda_rng_state=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                             elapsed_s=time.time() - t0)
        del x

    log_fh.close()
    formulas_fh.close()
    receipt = dict(schema="SEL1_TIMING_RECEIPT_V1", created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), config=vars(args),
                   months=months, event_drops=drops, invalid_reasons=dict(invalid_reasons), n_eval=n_eval,
                   no_step=bool(args.no_step), compute_seconds=round(time.time() - t0, 1), oos_read=False)
    json.dump(receipt, open(os.path.join(out_dir, "TRAIN_RECEIPT_V1.json"), "w"), indent=1)
    log("done " + json.dumps(dict(updates=last_updates - start_update, seconds=receipt["compute_seconds"])))


if __name__ == "__main__":
    main()
