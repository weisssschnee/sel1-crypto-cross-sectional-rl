"""M2 risk-PG trainer (stage-1 ruler): 1-min grid, GPU VM, HOLD-as-program-token.

Stage-1 changes against the previous M2 trainer. Each one is a measurement, not
a preference (P1 gate-13 table, okx-m1-p1-gate13-findings):

  rule 0   the stateful policy is told every asset is eligible, so a missing
           1-min bar freezes the position instead of forcing a whole-book
           retarget. The old forced retarget was a hold-independent turnover
           floor (DEV 2026-03: 125 turnover/day at hold 1440), i.e. a fake cost.
  ruler    one function (m1_ruler.eval_formula) scores training, selection,
           elite re-check and DEV: lag-2 rank book, 1/99 cross-sectionally
           winsorised forward returns, flat6 / maker2 / bookDepth cost, raw-ms
           funding, net log NAV.
  reward   mean of the HOLDOUT windows' net log NAV (q25 optional). second_best,
           max and median are banned. The IC term is gone: cs_ic2 is logged only.
  HOLD     extended with 2880 and 10080; every window holds a whole longest hold
           (episode = max(HOLDS)+60), so a month fits only 4 disjoint windows.
  band     hysteresis band of the unchanged stateful policy, one parameter.

Not changed: the pinned novelty_weighted_risk_update, the AlphaGPT policy and
output-head bias, the stack VM, the stateful action policy file, the optimizer.
Search accounting (N = scored formulas) is untouched.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from search_accounting import SearchAccounting, format_report

ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem"
os.environ.setdefault("M2_STAGE", os.path.join(ROOT, "runtime", "m1_staging"))

import m1_common as C
import m1_sample
from m1_sample import sample_batch_with_hold, strip_hold, hold_of

import m1_ruler as R

# ---- stage-1 ruler contract (2026-10-08 work order) ---------------------------
# Scoring lives in m1_ruler.eval_formula and nowhere else; this file only draws
# windows, samples programs and applies the risk update. Nothing here rewards
# turnover or penalises idleness: reward is the mean net-dollar log NAV of the
# lag-2 rank book, and IC is a logged diagnostic that never reaches the gradient.
EPISODE_MINUTES = R.EPISODE_MINUTES   # max(HOLDS)+60: every window holds a whole longest hold
N_WINDOWS = 12                  # requested; capped at the disjoint slots one month holds
ELITE_RECHECK = 32              # historical elites re-scored each update
REWARD_AGG = "mean"             # mean | q25 ; second_best / max / median are banned in m1_ruler
REWARD_FLOOR = C.REWARD_FLOOR
TRAIN_COST = "flat6"            # flat6 | maker2 | depth (depth needs --depth-dir bookDepth arrays)

# optimizer / risk-update constants inherited from M1 (verified in
# M1_TRAIN_PREFLIGHT_20261003.json; not re-derived here)
LEARNING_RATE = 5e-4
OP_BIAS = 1.5
END_BIAS = 2.0
CHECKPOINT_EVERY = 16
# per-formula book diagnostics (post-beta behaviour): never rewarded, always logged
DIAG_KEYS = ("turnover_per_day", "net_exposure", "gross_exposure", "beta_mkt", "gross_bp_day")

_shard_cache = {}


def load_shard(month, shard_dir):
    """Full-shard cache. M1 re-read a 200 MB npz every update and cached one
    month only; at 24 months x 234 MB the whole TRAIN set is 5.6 GB against
    84.9 GiB free, so caching all of it removes both the reload and the
    Windows file-lock crash that killed M1 attempt 1 at update 13."""
    if month in _shard_cache:
        return _shard_cache[month]
    path = os.path.join(shard_dir, f"m1_shard_{month}.npz")
    last = None
    for k in range(6):
        try:
            z = np.load(path, allow_pickle=False)
            break
        except PermissionError as e:
            last = e
            print(f"shard {month} locked, retry {k+1}/6 in 10s: {e}", flush=True)
            time.sleep(10)
    else:
        raise last
    symbols = [str(s) for s in z["symbols"]]
    feats, close = z["feats"], z["close"]
    fwd = C.forward_returns(close, None)
    fund = z["fund"].reshape(-1, 3)
    fi = np.zeros_like(fwd, dtype=bool)
    fr = np.zeros_like(fwd)
    a_i, m_i = fund[:, 0].astype(int), fund[:, 1].astype(int)
    keep = (m_i >= 0) & (m_i < fwd.shape[1]) & (a_i >= 0) & (a_i < fwd.shape[0])
    fi[a_i[keep], m_i[keep]] = True
    fr[a_i[keep], m_i[keep]] = fund[:, 2][keep]
    shard = dict(month=month, symbols=symbols, n_min=int(z["n_min"]), feats=feats,
                 close=close, fwd=fwd, eligible=np.isfinite(fwd), fi=fi, fr=fr)
    _shard_cache[month] = shard
    return shard


def gpu_episode_slices(shard, starts, device):
    out = []
    for s0 in starts:
        lo, hi = s0 - C.LOOKBACK_MINUTES, s0 + EPISODE_MINUTES
        t = torch.from_numpy(np.ascontiguousarray(shard["feats"][:, :, lo:hi]))
        out.append(t.to(device=device, dtype=torch.float32))
    return out


def draw_windows(rng, n_min, k):
    """Disjoint episode starts in a random order (so SELECT/HOLDOUT is not 'early vs late')."""
    starts = R.disjoint_starts(rng, n_min, k, EPISODE_MINUTES)
    return [starts[i] for i in rng.permutation(len(starts))]


def eval_one(vm, body, hold_min, slices, shard, starts, policy_mod, invalid_cls, *,
             cost, band, nav_usd, how):
    """One formula at one hold through the shared ruler.

    Returns (reward, per-window log NAVs, live_frac, cs_ic2, diag). cs_ic2 and diag
    (turnover/day, net and gross exposure, beta to the equal-weight market, gross
    bp/day before cost) are diagnostics only: nothing in them reaches the gradient."""
    res = R.eval_formula(vm, body, [hold_min], slices, shard, starts, policy_mod, invalid_cls,
                         costs=(cost,), variants=("win",), band=band, nav_usd=nav_usd,
                         episode=EPISODE_MINUTES)
    reward, logs = R.reward_of(res, hold_min, cost=cost, variant="win", how=how)
    parts = res["parts"][hold_min]["win"]
    days = EPISODE_MINUTES / 1440.0
    betas = [p["beta_mkt"] for p in parts if np.isfinite(p["beta_mkt"])]
    diag = dict(turnover_per_day=float(np.mean([p["turnover"] for p in parts])) / days,
                net_exposure=float(np.mean([p["net_exposure_mean"] for p in parts])),
                gross_exposure=float(np.mean([p["gross_exposure_mean"] for p in parts])),
                beta_mkt=float(np.mean(betas)) if betas else float("nan"),
                gross_bp_day=1e4 * float(np.mean([p["gross"] for p in parts])) / days)
    return reward, logs, res["live_frac"], res["cs_ic2"], diag


def _reward_variance(reward_pool, reward_sq_pool):
    """Pooled variance of the per-update mean holdout reward.

    V in the deflated-Sharpe hurdle is the variance of the trial returns. The
    per-update mean is the unit being compared across updates, so the variance
    is taken over those means. Returns None until two updates have landed,
    because a single observation has no variance.
    """
    if len(reward_pool) < 2:
        return None
    m = float(np.mean(reward_pool))
    v = float(np.mean(reward_sq_pool) - m * m)
    return v if v > 0 else None


def _sr0_now(n_evaluated, reward_pool, reward_sq_pool):
    """Deflation hurdle at the current N, or None before V is estimable."""
    if n_evaluated < 1:
        return None
    var = _reward_variance(reward_pool, reward_sq_pool)
    try:
        sr0, _ = SearchAccounting(1, 1, n_evaluated).hurdle(variance=var)
    except Exception:
        return None
    return float(sr0) if math.isfinite(sr0) else None


def _stop_budget_now(n_evaluated):
    """Formulas worth spending under the secretary rule, or None for small N."""
    if n_evaluated < 10:
        return None
    return int(SearchAccounting(1, 1, n_evaluated).stop_budget()['stop_after_formulas_rule_of_thumb'])


def _final_accounting(args, n_evaluated, reward_pool, reward_sq_pool, elite_pool):
    """Search-size accounting for the receipt.

    N is the number of formulas that received a reward. The elite pool is
    reported alongside it but is NOT added into N: those formulas were re-scored
    on already-drawn windows rather than being fresh draws from the search
    space, so counting them again would inflate the multiple-testing correction.
    """
    var = _reward_variance(reward_pool, reward_sq_pool)
    acc = SearchAccounting(updates=int(args.updates), batch=int(args.batch),
                           formulas_evaluated=int(n_evaluated),
                           reward_variance_V=var,
                           reward_variance_n=len(reward_pool),
                           notes="N counts scored formulas; elite re-scores excluded")
    d = acc.to_dict()
    d['elite_pool_size'] = sum(len(v) for v in elite_pool.values())
    d['test_statistic'] = None
    d['test_statistic_note'] = ("not measured by the trainer; a DEV-window scorer must "
                                "supply it together with the N=1 baseline")
    return d


def _write_deflation_report(out_dir, args, n_evaluated, reward_pool, reward_sq_pool, elite_pool):
    """Write DEFLATION_REPORT.txt. Reports the hurdle and, when available, the
    comparison. Never prints an absolute number without the hurdle beside it."""
    var = _reward_variance(reward_pool, reward_sq_pool)
    acc = SearchAccounting(updates=int(args.updates), batch=int(args.batch),
                           formulas_evaluated=int(n_evaluated),
                           reward_variance_V=var,
                           reward_variance_n=len(reward_pool),
                           notes="N counts scored formulas; elite re-scores excluded")
    text = format_report(acc, measured_test_stat=None, baseline_n1=None)
    text += ("\n  The trainer does not score a test window. Run the DEV scorer on\n"
             "  D:/ChengboRemote/data/m1_shards_dev and re-issue this report with\n"
             "  test_rank_icir and the N=1 baseline (CS_RANK(NEG(close_to_open_bps)),\n"
             "  hold 60m) filled in. An absolute value reported without SR_0 and\n"
             "  without the N=1 comparison is not a result.\n")
    with open(os.path.join(out_dir, "DEFLATION_REPORT.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


# Behaviour tolerance: two formulas are the same behaviour when their per-window net
# log NAVs on the SAME windows agree within BEHAV_TOL (1e-5 ~ 0.14 bp/day on a 7-day
# window). Calibrated on the 6-month DEV table: R11 and R12 differ by 0.007 bp/day in
# every month; the nearest genuinely different pair differs by 6.8 bp/day.
BEHAV_TOL = 1e-5


def behaviour_keep(vecs, groups, tol=BEHAV_TOL):
    """Indices to keep, scanning in the given (priority) order: an item is dropped when an
    already-kept item with the same group id has every coordinate within tol. Vectors are
    only comparable inside a group (same windows)."""
    vecs = [np.asarray(v, dtype=np.float64) for v in vecs]
    groups = np.asarray(groups)
    kept = []
    for i, v in enumerate(vecs):
        dup = False
        for j in kept:
            if groups[j] == groups[i] and vecs[j].shape == v.shape and float(np.abs(vecs[j] - v).max()) <= tol:
                dup = True
                break
        if not dup:
            kept.append(i)
    return kept


def _truncate_jsonl(path, last_update):
    """Drop rows written after the checkpoint we resume from (they will be re-run)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_name")
    ap.add_argument("--shard-dir", default=os.path.join(ROOT, "runtime", "m1_staging", "m1_shards"))
    ap.add_argument("--updates", type=int, default=128)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--persist", type=int, default=1, choices=(0, 1))
    ap.add_argument("--aggregate", default=REWARD_AGG, choices=R.AGGS,
                    help="second_best / max / median are not offered: m1_ruler.aggregate refuses them")
    ap.add_argument("--windows", type=int, default=N_WINDOWS,
                    help="requested windows per update; capped at the disjoint slots one month holds")
    ap.add_argument("--elite-recheck", type=int, default=ELITE_RECHECK)
    ap.add_argument("--cost", default=TRAIN_COST, choices=R.COSTS,
                    help="cost model the reward is computed under (flat6 = old 6bp, maker2, depth)")
    ap.add_argument("--band", type=float, default=R.BAND,
                    help="hysteresis band (hold_l1_threshold); 0.5 = pre-existing contract default")
    ap.add_argument("--nav-usd", type=float, default=R.NAV_USD, help="assumed NAV for the depth cost")
    ap.add_argument("--depth-dir", default=None,
                    help="dir with depth_{month}.npz; required when --cost depth")
    ap.add_argument("--winsor-pct", type=float, default=R.WINSOR_PCT,
                    help="per-minute cross-sectional winsorisation of forward returns; 0 = off")
    ap.add_argument("--min-live-frac", type=float, default=0.5,
                    help="recorded in the manifest only; coverage is gated inside m1_ruler (probe < 0.1)")
    ap.add_argument("--vm-block-elems", type=int, default=1 << 25,
                    help="VM trailing-window block size (elements). A 10140-min episode makes the "
                         "window-1440 unfold ~4 GB per tensor; the blocked path is bit-identical "
                         "(VM fast-path test). 0 = the original whole-panel path")
    ap.add_argument("--ema-cache-max", type=int, default=1,
                    help="VM EMA kernel cache size. One float32 kernel is 0.5 GB at a 11580-column "
                         "slice; the default 64 held 7 of them (3.5 GB). Cache hits only skip a "
                         "rebuild (the kernel is a pure function of its key)")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY,
                    help="updates between resumable checkpoints (model, optimizer, elite pool, all RNG state)")
    ap.add_argument("--resume", default=None,
                    help="'auto' = newest checkpoint in the output dir, or a checkpoint path. Refuses if any "
                         "argument other than --updates/--max-hours differs from the checkpointed run, or if a "
                         "module sha differs")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="wall-clock ceiling across all sessions of this run (0 = none). On reaching it the "
                         "run checkpoints, writes STOPPED.json and exits; --resume continues it")
    args = ap.parse_args()
    if args.cost == "depth" and not args.depth_dir:
        raise SystemExit("--cost depth needs --depth-dir (no silent flat fallback)")

    out_dir = os.path.join(ROOT, "artifacts", "m2_v1", args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)

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
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import (
        XsTsCsStackVM, OPERATOR_NAMES, CandidateLocalInvalid,
    )
    import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vm_mod
    vm_mod.BLOCK_ELEMS = int(args.vm_block_elems)
    vm_mod.EMA_KERNEL_CACHE_MAX = int(args.ema_cache_max)
    from okx_alpha_discovery.adapters.xs_tscs_sampler_v2 import build_legality_table
    from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod
    ModelConfig.DEVICE = device

    months = sorted(os.path.basename(p)[len("m1_shard_"):-len(".npz")]
                    for p in _glob_shards(args.shard_dir))
    assert months, f"no shards in {args.shard_dir}"
    bad = [m for m in months if not (m.startswith("2024-") or m.startswith("2025-"))]
    assert not bad, f"non-TRAIN shards leaked into {args.shard_dir}: {bad}"
    # Hard seal check: open every shard and verify its last minute precedes
    # 2026-07-01. The month-name assertion above can be fooled by a mislabelled
    # file; this cannot.
    from oos_guard import guard_shard_dir, OOSViolation
    try:
        guard_shard_dir(args.shard_dir, 'TRAIN', expect_months=months)
    except OOSViolation as exc:
        raise SystemExit(f'OOS GUARD REFUSED TO TRAIN: {exc}')
    # The 17-column contract: a 16-column shard would silently misalign every
    # terminal index from 16 upward, and there is no such column today.
    _probe = np.load(os.path.join(args.shard_dir, f"m1_shard_{months[0]}.npz"),
                     allow_pickle=True)
    if _probe["feats"].shape[1] != len(C.TERMINALS):
        raise SystemExit(
            f'shard has {_probe["feats"].shape[1]} terminals but TERMINALS has '
            f'{len(C.TERMINALS)}; rebuild the shards before training')

    vm = XsTsCsStackVM(len(C.TERMINALS), windows=C.WINDOWS)
    torch.manual_seed(args.seed)
    op_names = list(OPERATOR_NAMES) + [f"W{w}" for w in C.WINDOWS] + ["END"] + m1_sample.HOLD_NAMES
    model, fvocab = make_basin_model(
        alphagpt_mod=alphagpt_mod, ModelConfig=ModelConfig, FormulaVocab=FormulaVocab,
        feature_names=list(C.TERMINALS), operator_token_names=op_names,
        max_len=C.MAX_PROGRAM_LEN, device=device)
    apply_output_bias(model, len(C.TERMINALS), len(OPERATOR_NAMES), len(C.WINDOWS), OP_BIAS, END_BIAS)

    cap_windows = len(R.disjoint_starts(np.random.default_rng(0), 30 * 1440, args.windows))
    n_sel_m = max(1, cap_windows // 2)
    n_hold_m = max(1, cap_windows - n_sel_m)
    manifest = dict(schema="M2_TRAIN_LAUNCH_MANIFEST_V2_STAGE1", captured_by="CHILD_PROCESS",
                    python=sys.executable, cwd=os.getcwd(),
                    entry=os.path.abspath(__file__), args=vars(args),
                    modules=dict(m1_ruler=_sha(R.__file__), m1_common=_sha(C.__file__),
                                 m1_sample=_sha(m1_sample.__file__),
                                 stateful_action_policy=_sha(policy_mod.__file__)),
                    months=months, vocab_size=fvocab.size, holds=list(m1_sample.HOLDS),
                    model_params=sum(p.numel() for p in model.parameters()),
                    cost=dict(model=args.cost, specs={k: list(v) for k, v in R.cost_specs(args.nav_usd).items()},
                              nav_usd=args.nav_usd, depth_dir=args.depth_dir,
                              note="depth = fee + book-walk impact only: an optimistic lower bound",
                              funding="raw-instant events, floor_to_minute"),
                    ruler=dict(module="m1_ruler.eval_formula", rule0="policy eligible=all-True; settle freezes through gaps",
                               winsor_pct=args.winsor_pct, band=args.band, lag=C.EXECUTION_LAG_MIN,
                               positions="C.rank_weights, sum|w|=1"),
                    episodes=dict(minutes=EPISODE_MINUTES, windows_requested=args.windows,
                                  windows_per_30d_month=cap_windows,
                                  lookback=C.LOOKBACK_MINUTES, lag=C.EXECUTION_LAG_MIN,
                                  constraint="episode_minutes = max(HOLDS)+60"),
                    reward=dict(aggregate=args.aggregate, floor=REWARD_FLOOR,
                                elite_recheck=args.elite_recheck,
                                definition=f"{args.aggregate} of the HOLDOUT windows' net log NAV "
                                           f"({args.cost}, raw-ms funding, lag 2, per-formula HOLD, "
                                           f"1/99 winsorised forward returns); SELECT windows are "
                                           f"diagnostic only and never enter the gradient",
                                select_windows=n_sel_m, holdout_windows=n_hold_m,
                                banned_aggregates=["second_best", "max", "median"],
                                ic_weight=0.0, ic_role="cs_ic2 logged, never rewarded",
                                no_turnover_term=True),
                    written_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    manifest["modules"]["trainer"] = _sha(os.path.abspath(__file__))
    manifest["behaviour_dedup"] = dict(
        signature="per-window HOLDOUT net log NAV vector on the update's windows (TRAIN only: DEV is eval-only "
                  "and never enters the trainer)", tol=BEHAV_TOL,
        applies_to=["elite pool insertion", "elite pool after re-check", "n_distinct_behaviours (effective N, "
                    "within-update; cross-update repeats are not collapsed, so the cumulative is an upper bound)"])
    manifest["book_diagnostics"] = dict(logged=list(DIAG_KEYS), role="diagnostics only, never in the gradient")
    manifest["checkpoint"] = dict(every=args.checkpoint_every, resumable=True, max_hours=args.max_hours,
                                  state=["model", "optimizer", "elite_pool", "reward pools", "N counters",
                                         "starts_rng", "torch cpu+cuda rng", "elapsed"])
    if args.max_hours and not args.persist:
        raise SystemExit("--max-hours needs --persist 1 (the stop writes a checkpoint)")
    ck, ck_path = None, None
    if args.resume:
        ck_path = _latest_checkpoint(out_dir) if args.resume == "auto" else args.resume
        if not ck_path or not os.path.exists(ck_path):
            raise SystemExit(f"--resume {args.resume}: no checkpoint found in {out_dir}")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        if "elite_pool" not in ck:
            raise SystemExit(f"{ck_path} is not resumable (written before elite pool / RNG state were saved)")
        free = {"updates", "max_hours", "resume", "checkpoint_every", "manifest_only"}
        diff = {k: (ck["args"].get(k), v) for k, v in vars(args).items() if k not in free and ck["args"].get(k) != v}
        if diff:
            raise SystemExit(f"--resume refused, arguments differ from the checkpointed run: {diff}")
        if ck["modules"] != manifest["modules"]:
            raise SystemExit(f"--resume refused, code differs from the checkpointed run: "
                             f"{ck['modules']} vs {manifest['modules']}")
        manifest["resumed_from"] = dict(checkpoint=ck_path, update=int(ck["update"]))
    man_name = ("LAUNCH_MANIFEST.json" if ck is None
                else "RESUME_MANIFEST_%s.json" % time.strftime("%Y%m%dT%H%M%S"))
    json.dump(manifest, open(os.path.join(out_dir, man_name), "w"), indent=1)
    if args.manifest_only:
        return

    table = build_legality_table(vm, max_len=C.MAX_PROGRAM_LEN, available_windows=C.WINDOWS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    starts_rng = np.random.default_rng(args.seed)
    if ck is not None:
        for fn in ("BATCH_LOG.jsonl", "FORMULAS.jsonl", "ELITE_RECHECK.jsonl"):
            n_dropped = _truncate_jsonl(os.path.join(out_dir, fn), int(ck["update"]))
            print(f"resume: {fn} truncated to update <= {ck['update']} ({n_dropped} later rows dropped)", flush=True)
    fmode = "a" if ck is not None else "w"
    log_fh = open(os.path.join(out_dir, "BATCH_LOG.jsonl"), fmode, encoding="utf-8")
    formulas_fh = open(os.path.join(out_dir, "FORMULAS.jsonl"), fmode, encoding="utf-8") if args.persist else None
    elite_fh = open(os.path.join(out_dir, "ELITE_RECHECK.jsonl"), fmode, encoding="utf-8")
    invalid_reasons = collections.Counter()
    # month -> list[(key=(body, hold), hold, expr, holdout_reward, month, holdout window NAV vector)]
    elite_pool = {}
    vram = None
    # Search-size accounting (Bailey & Lopez de Prado 2014). Every formula that
    # receives a reward is a trial; N is the denominator of the deflated Sharpe
    # and the argument of the secretary-problem stop, so it is accumulated here
    # and written into every BATCH_LOG row rather than reconstructed afterwards.
    n_formulas_evaluated = 0
    n_behaviours_cum = 0    # within-update distinct behaviours, summed (upper bound on distinct overall)
    reward_pool = []        # per-update mean holdout reward, for V
    reward_sq_pool = []     # per-update mean squared, for a pooled variance
    start_update, elapsed_prev = 0, 0.0
    if ck is not None:
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        elite_pool = ck["elite_pool"]
        n_formulas_evaluated = int(ck["n_formulas_evaluated"])
        n_behaviours_cum = int(ck["n_behaviours_cum"])
        reward_pool, reward_sq_pool = list(ck["reward_pool"]), list(ck["reward_sq_pool"])
        invalid_reasons = collections.Counter(ck["invalid_reasons"])
        starts_rng.bit_generator.state = ck["starts_rng_state"]
        torch.set_rng_state(ck["torch_rng_state"])
        if device.type == "cuda" and ck.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng_state"], device)
        start_update, elapsed_prev = int(ck["update"]) + 1, float(ck["elapsed_s"])
        print(f"resumed from {ck_path}: next update {start_update}, N={n_formulas_evaluated}, "
              f"elapsed so far {_hms(elapsed_prev)}", flush=True)
    t0 = time.time() - elapsed_prev       # eta and compute_seconds count every session of this run
    t_sess0 = time.time()
    stopped_early = False
    stop_json = os.path.join(out_dir, "STOPPED.json")
    if os.path.exists(stop_json):
        os.replace(stop_json, os.path.join(out_dir, "STOPPED_prev_%s.json" % time.strftime("%Y%m%dT%H%M%S")))

    for update in range(start_update, args.updates):
        month = months[update % len(months)]
        shard = R.prepare_shard(load_shard(month, args.shard_dir), args.winsor_pct,
                                depth_dir=args.depth_dir if args.cost == "depth" else None)
        all_starts = draw_windows(starts_rng, shard["n_min"], args.windows)
        if len(all_starts) < 2:
            raise SystemExit(f"{month}: only {len(all_starts)} disjoint {EPISODE_MINUTES}-min window fits; "
                             "need >=2 for a SELECT/HOLDOUT split")
        slices = gpu_episode_slices(shard, all_starts, device)

        batch = sample_batch_with_hold(model, vm, table, args.batch, device=device)
        tokens_list = batch.formulas

        # ---- split windows: SELECT and HOLDOUT are disjoint draws.
        # The gradient sees only the HOLDOUT windows; SELECT windows are a
        # diagnostic of how much of a formula's score is window luck
        # (select_minus_holdout). A 30-day month holds 4 disjoint 10140-min
        # windows, so this is 2 + 2. Fewer windows than the old 12 is the price
        # of every window containing a whole 1-week hold.
        n_sel = len(all_starts) // 2
        n_hold = len(all_starts) - n_sel
        sel_starts, hold_starts = all_starts[:n_sel], all_starts[n_sel:]
        sel_slices, hold_slices = slices[:n_sel], slices[n_sel:]
        ev = dict(cost=args.cost, band=args.band, nav_usd=args.nav_usd, how=args.aggregate)

        rewards = np.full(args.batch, REWARD_FLOOR, dtype=np.float64)
        sel_rewards = np.full(args.batch, REWARD_FLOOR, dtype=np.float64)
        live_frac = np.full(args.batch, np.nan)
        cs_ic = np.full(args.batch, np.nan)
        diag = {k: np.full(args.batch, np.nan) for k in DIAG_KEYS}
        turn_day = diag["turnover_per_day"]
        hold_logs = [None] * args.batch      # per-window HOLDOUT net log NAV: the behaviour signature
        valid_mask = np.zeros(args.batch, dtype=bool)
        n_valid = 0
        hold_hist = np.zeros(len(m1_sample.HOLDS), dtype=int)
        for i, toks in enumerate(tokens_list):
            hold_min = hold_of(vm, list(toks))
            hold_hist[m1_sample.HOLDS.index(hold_min)] += 1
            body = list(strip_hold(vm, list(toks)))
            invalid_reason = None
            try:
                r_sel, _, _, _, _ = eval_one(
                    vm, body, hold_min, sel_slices, shard, sel_starts,
                    policy_mod, CandidateLocalInvalid, **ev)
                r_hold, logs_hold, lf_h, cs_h, dg_h = eval_one(
                    vm, body, hold_min, hold_slices, shard, hold_starts,
                    policy_mod, CandidateLocalInvalid, **ev)
                rewards[i] = r_hold
                sel_rewards[i] = r_sel
                live_frac[i] = lf_h
                cs_ic[i] = cs_h
                for k in DIAG_KEYS:
                    diag[k][i] = dg_h[k]
                hold_logs[i] = np.asarray(logs_hold, dtype=np.float64)
                valid_mask[i] = True
                n_valid += 1
            except CandidateLocalInvalid as exc:
                invalid_reason = str(exc)
                invalid_reasons[invalid_reason.split("_")[0] + "_" +
                                ("UNTRADABLE" if "UNTRADABLE" in invalid_reason else "VM")] += 1
                logs_hold = None
            if formulas_fh is not None:
                formulas_fh.write(json.dumps(dict(
                    update=update, tokens=list(toks), body=body, hold_min=hold_min,
                    expr=vm.describe(body, list(C.TERMINALS)),
                    reward=round(float(rewards[i]), 6),
                    reward_select=round(float(r_sel), 6) if valid_mask[i] else None,
                    cs_ic2=round(float(cs_ic[i]), 6) if np.isfinite(cs_ic[i]) else None,
                    turnover_per_day=round(float(turn_day[i]), 3) if np.isfinite(turn_day[i]) else None,
                    net_exposure=round(float(diag["net_exposure"][i]), 5) if np.isfinite(diag["net_exposure"][i]) else None,
                    gross_exposure=round(float(diag["gross_exposure"][i]), 5) if np.isfinite(diag["gross_exposure"][i]) else None,
                    beta_mkt=round(float(diag["beta_mkt"][i]), 4) if np.isfinite(diag["beta_mkt"][i]) else None,
                    gross_bp_day=round(float(diag["gross_bp_day"][i]), 3) if np.isfinite(diag["gross_bp_day"][i]) else None,
                    holdout_window_nav=[round(float(x), 7) for x in hold_logs[i]] if valid_mask[i] else None,
                    live_frac=round(float(live_frac[i]), 4) if np.isfinite(live_frac[i]) else None,
                    pos_windows_holdout=int((logs_hold > 0).sum()) if valid_mask[i] else None,
                    windows_holdout=int(len(logs_hold)) if valid_mask[i] else None,
                    invalid_reason=invalid_reason)) + "\n")
        if formulas_fh is not None:
            formulas_fh.flush()

        # ---- advantage: standardised holdout reward, nothing else. No IC term,
        # no turnover term, no idleness term: whatever the formula earns after
        # the ruler's costs is the whole signal.
        reward_valid_std = float(rewards[valid_mask].std()) if n_valid else 0.0
        skip_reason = None
        if n_valid < 20:
            skip_reason = f"n_valid={n_valid}<20"
        elif reward_valid_std == 0.0:
            skip_reason = "reward_valid_std==0"
        if skip_reason is None:
            scaled = np.empty(args.batch)
            scaled[valid_mask] = rewards[valid_mask] / reward_valid_std
            scaled[~valid_mask] = float(scaled[valid_mask].min()) - 1.0
            upd = novelty_weighted_risk_update(model, optimizer, batch, scaled,
                                               np.ones(args.batch), device)
        else:
            upd = {}

        # ---- historical elite re-check, scoped to THIS update's month and run
        # before this update's formulas enter the pool (they were just scored on
        # these same windows). Each entry's stored scalar is replaced by the
        # fresh same-month, same-objective draw.
        recheck = []
        if args.elite_recheck > 0 and elite_pool.get(month):
            pool = sorted(elite_pool[month], key=lambda e: -e[3])[:args.elite_recheck]
            kept = []
            for key, hold, expr, old_score, _, _, _ in pool:
                try:
                    r, logs, lf_v, cs_v, dg_v = eval_one(
                        vm, list(key[0]), hold, hold_slices, shard, hold_starts,
                        policy_mod, CandidateLocalInvalid, **ev)
                except CandidateLocalInvalid:
                    continue
                recheck.append(dict(update=update, month=month, hold_min=hold,
                                    expr=expr, old_score=round(float(old_score), 8),
                                    new_score=round(float(r), 8),
                                    delta=round(float(r - old_score), 8),
                                    cs_ic2=round(float(cs_v), 6) if np.isfinite(cs_v) else None,
                                    live_frac=round(float(lf_v), 4),
                                    turnover_per_day=round(dg_v["turnover_per_day"], 3),
                                    net_exposure=round(dg_v["net_exposure"], 5),
                                    beta_mkt=round(dg_v["beta_mkt"], 4) if np.isfinite(dg_v["beta_mkt"]) else None,
                                    windows=len(logs), pos_windows=int((logs > 0).sum())))
                kept.append((key, hold, expr, float(r), month,
                             np.asarray(logs, dtype=np.float64), tuple(hold_starts)))
            elite_pool[month] = kept
            elite_fh.write(json.dumps(dict(update=update, month=month, n=len(recheck),
                                           items=recheck)) + "\n")
            elite_fh.flush()

        # ---- behaviour dedup. Vectors are per-window HOLDOUT net log NAVs and are only
        # comparable on the same windows: the re-checked pool entries and this update's
        # batch were all scored on hold_starts just now, anything else sits in its own group.
        order = [int(i) for i in np.argsort(-rewards) if valid_mask[i]]
        beh_keep = behaviour_keep([hold_logs[i] for i in order], [0] * len(order))
        n_distinct_beh = len(beh_keep)
        n_behaviours_cum += n_distinct_beh
        month_pool = elite_pool.setdefault(month, [])
        cands = list(month_pool)
        for pos in beh_keep:
            i = order[pos]
            body, hold = list(strip_hold(vm, list(tokens_list[i]))), hold_of(vm, list(tokens_list[i]))
            cands.append(((tuple(body), hold), hold, vm.describe(body, list(C.TERMINALS)),
                          float(rewards[i]), month, hold_logs[i], tuple(hold_starts)))
        cands.sort(key=lambda e: -e[3])          # the incumbent wins a tie with its own clone
        gid = {}
        keep_ix = behaviour_keep([e[5] for e in cands], [gid.setdefault(e[6], len(gid)) for e in cands])
        month_pool = [cands[j] for j in keep_ix][:200]
        elite_pool[month] = month_pool

        bodies = [list(strip_hold(vm, list(t))) for t in tokens_list]
        roots = _root_summary(bodies, args.batch, vm)
        sel_scores = np.array(sel_rewards, dtype=np.float64)
        sel_pos = float((sel_scores[valid_mask] > 0).mean()) if n_valid else 0.0

        # --- search-size accounting -----------------------------------------
        # N counts every formula that was scored this update, not only the ones
        # that passed the tradability gate: a rejected formula was still a draw
        # from the search space and still had a chance to look good on noise.
        n_formulas_evaluated += len(tokens_list)
        if n_valid:
            _rm = float(rewards[valid_mask].mean())
            reward_pool.append(_rm)
            reward_sq_pool.append(float((rewards[valid_mask] ** 2).mean()))
        _sr0 = _sr0_now(n_formulas_evaluated, reward_pool, reward_sq_pool)
        _stop = _stop_budget_now(n_formulas_evaluated)

        row = dict(update=update, month=month, n_valid=n_valid,
                   n_formulas_evaluated_cum=n_formulas_evaluated,
                   sr0_deflation_hurdle=(None if _sr0 is None else round(_sr0, 6)),
                   secretary_stop_after=(_stop if _stop is not None else None),
                   reward_mean=round(float(rewards[valid_mask].mean()) if n_valid else REWARD_FLOOR, 6),
                   reward_positive_fraction=round(float((rewards > 0).mean()), 4),
                   reward_select_mean=round(float(sel_scores[valid_mask].mean()) if n_valid else REWARD_FLOOR, 6),
                   reward_select_positive_fraction=round(sel_pos, 4),
                   select_minus_holdout=round(float((sel_scores[valid_mask].mean() - rewards[valid_mask].mean())), 6) if n_valid else None,
                   reward_valid_std=round(reward_valid_std, 6),
                   cs_ic2_mean=round(float(np.nanmean(cs_ic[valid_mask])), 6) if n_valid else None,
                   turnover_per_day_mean=round(float(np.nanmean(turn_day[valid_mask])), 3) if n_valid else None,
                   turnover_per_day_median=round(float(np.nanmedian(turn_day[valid_mask])), 3) if n_valid else None,
                   live_frac_mean=round(float(np.nanmean(live_frac[valid_mask])), 4) if n_valid else None,
                   abs_net_exposure_mean=round(float(np.nanmean(np.abs(diag["net_exposure"][valid_mask]))), 5) if n_valid else None,
                   abs_net_exposure_max=round(float(np.nanmax(np.abs(diag["net_exposure"][valid_mask]))), 5) if n_valid else None,
                   beta_mkt_abs_mean=round(float(np.nanmean(np.abs(diag["beta_mkt"][valid_mask]))), 4) if n_valid else None,
                   gross_bp_day_mean=round(float(np.nanmean(diag["gross_bp_day"][valid_mask])), 3) if n_valid else None,
                   gross_exposure_mean=round(float(np.nanmean(diag["gross_exposure"][valid_mask])), 4) if n_valid else None,
                   idle_books=int((diag["gross_exposure"][valid_mask] < 0.05).sum()) if n_valid else None,
                   n_distinct_behaviours=n_distinct_beh, n_behaviours_cum=n_behaviours_cum,
                   untradable_rejected=int(args.batch - n_valid),
                   cost=args.cost, band=args.band, aggregate=args.aggregate,
                   windows_total=args.windows, windows_select=n_sel, windows_holdout=n_hold,
                   starts_select=sel_starts, starts_holdout=hold_starts,
                   skipped_param_step=skip_reason,
                   hold_hist=hold_hist.tolist(), hold_names=m1_sample.HOLD_NAMES,
                   elite_pool_size=sum(len(v) for v in elite_pool.values()), elite_rechecked=len(recheck),
                   elite_recheck_old_mean=round(float(np.mean([r["old_score"] for r in recheck])), 8) if recheck else None,
                   elite_recheck_new_mean=round(float(np.mean([r["new_score"] for r in recheck])), 8) if recheck else None,
                   elite_recheck_pos_delta=int(sum(1 for r in recheck if r["delta"] > 0)),
                   top_root_share_pct=roots["top_root_share_pct"],
                   n_distinct_roots=roots["n_distinct_roots"],
                   **{k: upd.get(k) for k in ("pg_loss", "entropy_loss", "gradient_norm") if k in upd})
        if device.type == "cuda":
            vram = torch.cuda.max_memory_allocated() / 2**30
            row["vram_alloc_gib"] = round(vram, 3)
        log_fh.write(json.dumps(row) + "\n")
        log_fh.flush()
        json.dump(dict(pid=os.getpid(), update=update, updates_total=args.updates,
                       updated=time.strftime("%Y-%m-%dT%H:%M:%S%z")),
                  open(os.path.join(out_dir, "HEARTBEAT.json"), "w"), indent=1)
        if update < start_update + 4 or (update + 1) % 16 == 0 or update == args.updates - 1:
            eta = (time.time() - t_sess0) / (update + 1 - start_update) * (args.updates - update - 1)
            print("upd %3d %s valid %3d/%d hold%% %5.1f sel%% %5.1f gap %+.5f "
                  "elite %3d recheck %2d (old %+.5f new %+.5f) beh %d |netexp| %.4f turn %.1f/d "
                  "root %s@%s%% pg %s vram %sGiB eta %s"
                  % (update, month, n_valid, args.batch, row["reward_positive_fraction"] * 100,
                     sel_pos * 100, row["select_minus_holdout"] or 0.0,
                     sum(len(v) for v in elite_pool.values()), len(recheck),
                     row["elite_recheck_old_mean"] or 0.0, row["elite_recheck_new_mean"] or 0.0,
                     n_distinct_beh, row["abs_net_exposure_mean"] or 0.0,
                     row["turnover_per_day_mean"] or 0.0,
                     roots["top_root"], roots["top_root_share_pct"], upd.get("pg_loss"),
                     round(vram, 2) if vram else "-", _hms(eta)), flush=True)
        over_budget = bool(args.max_hours) and (time.time() - t0) >= args.max_hours * 3600.0
        last_update = update == args.updates - 1
        if args.persist and ((update + 1) % args.checkpoint_every == 0 or over_budget):
            _save_checkpoint(
                os.path.join(out_dir, f"checkpoint_{update:04d}.pt"),
                model=model.state_dict(), optimizer=optimizer.state_dict(), update=update,
                args=vars(args), modules=manifest["modules"], elite_pool=elite_pool,
                n_formulas_evaluated=n_formulas_evaluated, n_behaviours_cum=n_behaviours_cum,
                reward_pool=reward_pool, reward_sq_pool=reward_sq_pool,
                invalid_reasons=dict(invalid_reasons),
                starts_rng_state=starts_rng.bit_generator.state,
                torch_rng_state=torch.get_rng_state(),
                cuda_rng_state=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                elapsed_s=time.time() - t0)
        if over_budget and not last_update:
            stopped_early = True
            json.dump(dict(reason="max_hours", max_hours=args.max_hours, last_update=update,
                           next_update=update + 1, updates_total=args.updates,
                           elapsed_s=round(time.time() - t0, 1),
                           resume="--resume auto (same arguments; --updates/--max-hours may change)",
                           written=time.strftime("%Y-%m-%dT%H:%M:%S%z")),
                      open(os.path.join(out_dir, "STOPPED.json"), "w"), indent=1)
            print(f"wall-clock ceiling {args.max_hours} h reached after update {update}; checkpointed, "
                  f"stopping. Resume with --resume auto", flush=True)
            break

    log_fh.close()
    if formulas_fh is not None:
        formulas_fh.close()
    elite_fh.close()
    if stopped_early:
        return
    with torch.no_grad():
        eval_batch = sample_batch_with_hold(model, vm, table, args.batch, device=device)
    final_bodies = [list(strip_hold(vm, list(t))) for t in eval_batch.formulas]
    final_holds = [hold_of(vm, list(t)) for t in eval_batch.formulas]
    final_roots = _root_summary(final_bodies, args.batch, vm)
    receipt = dict(schema="M2_TRAIN_RECEIPT_V1", created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   months=months, config=vars(args),
                   reward_definition=manifest["reward"]["definition"],
                   invalid_reasons=dict(invalid_reasons),
                   elite_pool_final_size=sum(len(v) for v in elite_pool.values()),
                   behaviour_dedup=dict(tol=BEHAV_TOL, n_behaviours_cum_upper_bound=n_behaviours_cum,
                                        note="sum over updates of within-update distinct HOLDOUT-vector behaviours; "
                                             "same behaviour recurring in a later update is counted again"),
                   resumed_from=manifest.get("resumed_from"),
                   trained_policy_final=dict(roots=final_roots,
                                            hold_hist={h: final_holds.count(h) for h in m1_sample.HOLDS}),
                   peak_vram_alloc_gib=round(vram, 3) if vram else None,
                   compute_seconds=round(time.time() - t0, 1), oos_read=False,
                   search_accounting=_final_accounting(args, n_formulas_evaluated,
                                                       reward_pool, reward_sq_pool,
                                                       elite_pool))
    json.dump(receipt, open(os.path.join(out_dir, "TRAIN_RECEIPT_V1.json"), "w"), indent=1)
    _write_deflation_report(out_dir, args, n_formulas_evaluated,
                            reward_pool, reward_sq_pool, elite_pool)
    print("done", json.dumps(dict(updates=args.updates, seconds=receipt["compute_seconds"])))


def _sha(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def _glob_shards(shard_dir):
    import glob
    return glob.glob(os.path.join(shard_dir, "m1_shard_*.npz"))


def _root_summary(bodies, n, vm):
    from basin_common_v1 import root_distribution, summarize_roots
    return summarize_roots(root_distribution(bodies, [len(b) for b in bodies], vm), n)


def _hms(sec):
    sec = int(max(sec, 0))
    return "%d:%02d:%02d" % (sec // 3600, (sec % 3600) // 60, sec % 60)


if __name__ == "__main__":
    main()
