"""Cold-start risk-PG training for one basin, on the new searched-window VM.

Reuses the sha256-pinned legacy PG mechanics (novelty_weighted_risk_update,
select_risk_elite, sequence_neglogp_entropy, the AlphaGPT architecture) exactly
as the production Broad893 adapter does -- same legacy_root import pattern, same
update function, unmodified. What's new here: the VM/sampler (xs_tscs_vm_v2,
xs_tscs_sampler_v2), a fresh per-basin vocab sized to AlphaGPT via
make_model_for_ops's own pattern, an output-head bias correction measured
2026-09-28 (op_bias/end_bias), per-formula/per-batch persistence to
FORMULAS.jsonl, and periodic checkpoints. Reward is the mean of 8 fresh,
non-repeating weekly log-NAVs drawn anew every update (not a fixed 8-week
window scored by its 25th percentile -- that construction let updates 96-127
of validate_128x256_v3_fixed memorize those exact 8 weeks: 84% positive on
the original window vs 3% on an unseen non-overlapping window, below plain
random's own 13% on that same unseen window). Evaluation runs on a
multiprocessing pool since it is pure CPU work independent per formula.

Every claim about "learning" is checked against a measured uniform baseline
sampled from the SAME legality table, per the harness rule this line follows:
never report a root/length distribution without its null alongside it.
"""
from __future__ import annotations
import argparse, collections, hashlib, inspect, json, multiprocessing, os, subprocess, sys, time
import numpy as np
import torch

import basin_common_v1 as common
from basin_common_v1 import (
    ROOT, VOCAB, MAX_PROGRAM_LEN, REWARD_FLOOR, ensure_sys_path, sha256_file,
    load_basin_fields, load_panel, episode_starts, evaluate_one, reward_from_logs,
    append_formula_log, UniformStub, root_distribution, summarize_roots,
    pool_init, pool_eval_one,
)

LEARNING_RATE = 5e-4
RISK_EPSILON = 0.05
ENTROPY_GAMMA = 0.7
GRAD_CLIP_NORM = 5.0
# Measured 2026-09-28 (GATE_ROOT_CAUSE_20260928.md) for a per-basin vocab (3-5
# terminals): op_bias=1.5, end_bias=2.0 lands root diversity at <10% for any
# single operator and a length spread from short to near the cap, instead of
# the ~92-99% single-root collapse a zero-bias or inherited-AlphaGPT head gives.
OP_BIAS = 1.5
END_BIAS = 2.0
SEARCH_SEED = 20260928
CHECKPOINT_EVERY = 16


def make_basin_model(*, alphagpt_mod, ModelConfig, FormulaVocab, feature_names,
                      operator_token_names, max_len, device):
    """Fresh AlphaGPT sized to this basin's vocab, mirroring make_broad_model's
    own FORMULA_VOCAB-swap pattern (adapter :81-86) plus a MAX_FORMULA_LEN
    override: AlphaGPT's positional embedding is sized off that global
    (model_core/alphagpt.py :232), fixed at 12+1 by default, which is too short
    for our 48-token length budget. Both globals are restored in `finally`."""
    vocab = FormulaVocab(feature_names=tuple(feature_names),
                          operator_names=tuple(operator_token_names))
    old_vocab = alphagpt_mod.FORMULA_VOCAB
    old_len = ModelConfig.MAX_FORMULA_LEN
    try:
        alphagpt_mod.FORMULA_VOCAB = vocab
        ModelConfig.MAX_FORMULA_LEN = max_len
        model = alphagpt_mod.AlphaGPT().to(device)
    finally:
        alphagpt_mod.FORMULA_VOCAB = old_vocab
        ModelConfig.MAX_FORMULA_LEN = old_len
    return model, vocab


def apply_output_bias(model, feature_count, operator_count, window_count,
                       op_bias, end_bias):
    """Add the measured correction to every task head's bias, before training.

    MTPHead combines three heads with softmax gate weights that sum to 1
    (model_core/alphagpt.py MTPHead.forward), so identical bias on all three
    survives the combination exactly regardless of the (randomly initialized)
    gate: a weighted sum of identical vectors is that same vector.
    """
    end_token = feature_count + operator_count + window_count
    with torch.no_grad():
        for head in model.mtp_head.task_heads:
            head.bias[feature_count:feature_count + operator_count] += op_bias
            head.bias[end_token] += end_bias
@torch.no_grad()
def uniform_baseline(vm, table, batch_size, seed, device, sample_batch_fn):
    """Root/length distribution a zero-logit policy gives, under the SAME table.
    This is the null every 'the policy learned X' claim in this run's receipt is
    checked against (HARNESS.md: 'measure the null before training')."""
    gen = torch.Generator(device=device).manual_seed(seed) if device.type == "cuda" \
        else torch.Generator().manual_seed(seed)
    batch = sample_batch_fn(UniformStub(vm.token_count).to(device), vm, table, batch_size,
                             device=device, generator=gen)
    lengths = batch.lengths.cpu().tolist()
    roots = root_distribution(batch.context["tokens"].cpu().tolist(), lengths, vm)
    return summarize_roots(roots, batch_size), {
        "len_p10": int(np.percentile(lengths, 10)),
        "len_p50": int(np.percentile(lengths, 50)),
        "len_p90": int(np.percentile(lengths, 90)),
    }


def git_info(root):
    """(commit, dirty_digest). dirty_digest is None on a clean tree, else a
    sha256 of `git status --porcelain` (a signature of what's dirty, not the
    full diff). commit is the literal string "NOT_FOUND" on any git failure --
    launch_preflight.py already blocks NOT_-prefixed runtime_binding values,
    so a broken git call fails the preflight check instead of writing a
    manifest that silently claims an unverifiable commit."""
    def run(args):
        r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    commit = run(["rev-parse", "HEAD"])
    porcelain = run(["status", "--porcelain"])
    dirty_digest = hashlib.sha256(porcelain.encode("utf-8")).hexdigest() if porcelain else None
    return commit or "NOT_FOUND", dirty_digest


def module_sha256_of(*objs):
    """sha256 of the source file each obj (module, class, or function) is
    defined in, keyed by dotted module name. Deliberately derives the set of
    hashed files from what this run actually imports and executes -- including
    the sha256-pinned legacy novelty_weighted_risk_update -- rather than a
    hand-maintained path list that can silently drift from what's really
    loaded."""
    out = {}
    for obj in objs:
        try:
            path = inspect.getfile(obj)
        except TypeError:
            continue
        name = obj.__name__ if inspect.ismodule(obj) else obj.__module__
        out[name] = sha256_file(path)
    return out


def write_launch_manifest(out_dir, args, module_sha256):
    """Written by this child process at start, before any heavy compute, so
    launch_preflight.py's runtime_binding.captured_by=='CHILD_PROCESS' check
    reflects what is actually about to run -- not a hand-authored guess."""
    git_commit, git_dirty_digest = git_info(ROOT)
    entry = os.path.abspath(__file__)
    manifest = dict(
        captured_by="CHILD_PROCESS",
        python_executable=sys.executable,
        cwd=os.getcwd(),
        entry=entry,
        entry_sha256=sha256_file(entry),
        module_sha256=module_sha256,
        git_commit=git_commit,
        git_dirty_digest=git_dirty_digest,
        env=dict(
            PYTHONPATH=os.environ.get("PYTHONPATH", ""),
            torch_version=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_device_name=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        ),
        args=vars(args),
        written_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    path = os.path.join(out_dir, "LAUNCH_MANIFEST.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    return path, manifest


def write_heartbeat(out_dir, update, updates_total):
    heartbeat = dict(
        pid=os.getpid(), update=update, updates_total=updates_total,
        updated=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    with open(os.path.join(out_dir, "HEARTBEAT.json"), "w", encoding="utf-8") as fh:
        json.dump(heartbeat, fh, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("basin_id")
    ap.add_argument("out_name")
    ap.add_argument("--updates", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=SEARCH_SEED)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--persist", type=int, default=1, choices=(0, 1),
                     help="1=write FORMULAS.jsonl + checkpoints (default). 0=off, "
                          "for the mp_and_persist_check_v1 determinism A/B test.")
    ap.add_argument("--manifest-only", action="store_true",
                     help="write LAUNCH_MANIFEST.json and exit immediately, before any "
                          "model/pool/GPU work, so launch_preflight.py can capture a real "
                          "CHILD_PROCESS runtime_binding without spending the run's compute "
                          "budget.")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "artifacts/xs_basin_v1", args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "BATCH_LOG.jsonl")
    device = torch.device(args.device)

    ensure_sys_path()
    import model_core.alphagpt as alphagpt_mod
    from model_core.config import ModelConfig
    from model_core.vocab import FormulaVocab
    from experiments.rl_v2_autonomous_novelty_search_v4 import novelty_weighted_risk_update
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import (
        XsTsCsStackVM, OPERATOR_NAMES, WINDOWS)
    from okx_alpha_discovery.adapters.xs_tscs_sampler_v2 import (
        build_legality_table, sample_batch)
    ModelConfig.DEVICE = device

    # Manifest written from what this process actually imported -- including
    # the sha256-pinned novelty_weighted_risk_update -- before any GPU/model
    # work, so a --manifest-only invocation is cheap enough to run just for
    # launch_preflight.py's CHILD_PROCESS capture.
    module_sha256 = module_sha256_of(
        common, alphagpt_mod, ModelConfig, FormulaVocab,
        novelty_weighted_risk_update, XsTsCsStackVM, build_legality_table, sample_batch)
    manifest_path, _manifest = write_launch_manifest(out_dir, args, module_sha256)
    print(f"launch manifest -> {manifest_path}", flush=True)
    if args.manifest_only:
        return

    # Seed before ANY randomness, including model init: AlphaGPT's own
    # nn.Linear/nn.Embedding init draws from the global RNG inside
    # make_basin_model below, so seeding after construction (as an earlier
    # version of this script did) left initial weights un-reproducible across
    # runs regardless of --seed. Caught by mp_and_persist_check_v1's
    # persist-determinism A/B: both runs used --seed 778 but sampled different
    # formulas from update 0, because each process's model started from
    # different random weights.
    torch.manual_seed(args.seed)

    fields = load_basin_fields(args.basin_id)
    print(f"basin {args.basin_id}: {len(fields)} fields = {fields}", flush=True)
    n_times = load_panel(fields)["n_times"]

    F = len(fields)
    vm = XsTsCsStackVM(F)
    op_token_names = list(OPERATOR_NAMES) + [f"W{w}" for w in WINDOWS] + ["END"]
    assert F + len(op_token_names) == vm.token_count
    model, fvocab = make_basin_model(
        alphagpt_mod=alphagpt_mod, ModelConfig=ModelConfig, FormulaVocab=FormulaVocab,
        feature_names=fields, operator_token_names=op_token_names,
        max_len=MAX_PROGRAM_LEN, device=device)
    apply_output_bias(model, F, len(OPERATOR_NAMES), len(WINDOWS), OP_BIAS, END_BIAS)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params, vocab_size {fvocab.size} == vm.token_count {vm.token_count}",
          flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    table = build_legality_table(vm, max_len=MAX_PROGRAM_LEN)

    null_roots, null_len = uniform_baseline(vm, table, args.batch, args.seed, device, sample_batch)
    print(f"UNIFORM NULL: {json.dumps(null_roots)}  len={null_len}", flush=True)

    torch.manual_seed(args.seed)
    starts_rng = np.random.default_rng(args.seed)
    invalid_reasons = collections.Counter()
    log_fh = open(log_path, "w", encoding="utf-8")
    formulas_fh = open(os.path.join(out_dir, "FORMULAS.jsonl"), "w", encoding="utf-8") \
        if args.persist else None
    t_start = time.time()

    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(args.workers, initializer=pool_init, initargs=(args.basin_id, args.seed))
    try:
        for update in range(args.updates):
            starts = episode_starts(n_times, starts_rng)
            batch = sample_batch(model, vm, table, args.batch, device=device)
            tokens_list = batch.formulas
            lengths = batch.lengths.cpu().tolist()

            rewards = np.full(args.batch, REWARD_FLOOR, dtype=np.float64)
            valid_mask = np.zeros(args.batch, dtype=bool)
            n_valid = 0
            results = pool.map(pool_eval_one, [(tk, starts) for tk in tokens_list])
            for i, (tk, (status, payload)) in enumerate(zip(tokens_list, results)):
                logs = None
                reason = None
                if status == "ok":
                    logs = payload
                    rewards[i] = reward_from_logs(np.asarray(logs, dtype=np.float64))
                    valid_mask[i] = True
                    n_valid += 1
                else:
                    reason = payload
                    invalid_reasons[reason] += 1
                if formulas_fh is not None:
                    append_formula_log(formulas_fh, update=update, tokens=tk, vm=vm,
                                        fields=fields, reward=rewards[i],
                                        invalid_reason=reason, logs=logs)
            if formulas_fh is not None:
                formulas_fh.flush()

            # Rescale so the PG/advantage term sits on the same scale as the
            # fixed, sha-pinned entropy coefficient (0.03, tuned for rewards in
            # [0,1]; weekly log-NAV rewards are ~0.01-scale, which is why PG loss
            # was dwarfed by entropy loss in validate_128x256_v3_fixed). Invalid
            # formulas are pinned below the valid minimum so they can never enter
            # the elite set. reward_mean/positive-fraction below still use the
            # original unscaled `rewards`.
            reward_valid_std = float(rewards[valid_mask].std()) if n_valid else 0.0
            skip_reason = None
            if n_valid < 20:
                skip_reason = f"n_valid={n_valid}<20"
            elif reward_valid_std == 0.0:
                skip_reason = "reward_valid_std==0"

            if skip_reason is not None:
                upd = {}
            else:
                scaled_rewards = np.empty(args.batch, dtype=np.float64)
                scaled_rewards[valid_mask] = rewards[valid_mask] / reward_valid_std
                scaled_rewards[~valid_mask] = float(scaled_rewards[valid_mask].min()) - 1.0
                novelty = np.ones(args.batch, dtype=np.float64)
                upd = novelty_weighted_risk_update(model, optimizer, batch, scaled_rewards, novelty, device)

            roots = root_distribution(tokens_list, lengths, vm)
            root_summary = summarize_roots(roots, args.batch)
            row = {
                "update": update, "n_valid": n_valid,
                "reward_mean": round(float(rewards[valid_mask].mean()) if n_valid else REWARD_FLOOR, 6),
                "reward_positive_count": int((rewards > 0).sum()),
                "reward_positive_fraction": round(float((rewards > 0).mean()), 4),
                "reward_valid_std": round(reward_valid_std, 6),
                "starts": starts,
                "skipped_param_step": skip_reason,
                "len_p50": int(np.percentile(lengths, 50)),
                **root_summary,
                **{k: upd.get(k) for k in
                   ("quantile", "elite_count", "elite_fraction_actual",
                    "elite_novelty_mean", "pg_loss", "entropy_loss",
                    "gradient_norm", "skipped_nonfinite_update")
                   if k in upd},
            }
            log_fh.write(json.dumps(row) + "\n"); log_fh.flush()
            write_heartbeat(out_dir, update, args.updates)
            if update < 8 or (update + 1) % 16 == 0:
                print(f"upd {update:4d}  valid {n_valid:3d}/{args.batch}  "
                      f"pos% {row['reward_positive_fraction']*100:5.1f}  "
                      f"top_root {root_summary['top_root']} @ {root_summary['top_root_share_pct']}%  "
                      f"nroots {root_summary['n_distinct_roots']}  "
                      f"pg_loss {upd.get('pg_loss')}"
                      + (f"  SKIPPED({skip_reason})" if skip_reason else ""), flush=True)

            if args.persist and (update + 1) % CHECKPOINT_EVERY == 0:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "update": update, "args": vars(args)},
                           os.path.join(out_dir, f"checkpoint_{update:04d}.pt"))
    finally:
        pool.close(); pool.join()
    log_fh.close()
    if formulas_fh is not None:
        formulas_fh.close()

    final_roots, final_len = uniform_baseline(vm, table, args.batch, args.seed + 1, device, sample_batch)
    # re-measure the trained policy's own distribution on a fresh batch, same
    # size, for a clean final-vs-null comparison
    with torch.no_grad():
        eval_batch = sample_batch(model, vm, table, args.batch, device=device)
    trained_roots = root_distribution(
        eval_batch.formulas, eval_batch.lengths.cpu().tolist(), vm)
    trained_summary = summarize_roots(trained_roots, args.batch)

    receipt = dict(
        schema="XS_BASIN_TRAIN_RECEIPT_V1", created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        basin_id=args.basin_id, fields=fields, n_params=n_params,
        config=dict(updates=args.updates, batch=args.batch, seed=args.seed,
                    lr=LEARNING_RATE, risk_epsilon=RISK_EPSILON, entropy_gamma=ENTROPY_GAMMA,
                    grad_clip=GRAD_CLIP_NORM, max_len=MAX_PROGRAM_LEN,
                    op_bias=OP_BIAS, end_bias=END_BIAS, device=str(device),
                    workers=args.workers, persist=bool(args.persist),
                    reward_definition="mean of N_EPISODES weekly log-NAVs, fresh "
                    "non-repeating window drawn every update via episode_starts(n_times, "
                    "starts_rng) seeded once from --seed",
                    reward_rescaling="valid rewards divided by that update's valid-reward "
                    "std before novelty_weighted_risk_update; invalid formulas pinned to "
                    "scaled_min-1; param step skipped if n_valid<20 or std==0"),
        panel=dict(path=common.PANEL, sha256=sha256_file(common.PANEL)),
        vocab=dict(path=VOCAB, sha256=sha256_file(VOCAB)),
        invalid_reasons=dict(invalid_reasons),
        uniform_null_at_start=null_roots,
        trained_policy_final=trained_summary,
        compute_seconds=time.time() - t_start,
        oos_read=False, q0_read=False,
    )
    with open(os.path.join(out_dir, "TRAIN_RECEIPT_V1.json"), "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=1)
    print(f"\nreceipt -> {os.path.abspath(out_dir)}")
    print(f"uniform null top_root {null_roots['top_root']} @ {null_roots['top_root_share_pct']}%")
    print(f"trained   final top_root {trained_summary['top_root']} @ {trained_summary['top_root_share_pct']}%")


if __name__ == "__main__":
    main()

