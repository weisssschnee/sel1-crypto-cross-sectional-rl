"""BASIN5_PC_ENT trainer (BASIN5_PC_ENT_ORDER, 2026-10-12). Copy of m2h_b5_trainer.py (sha12
fa34d41b41ea, NOT modified) with one change, the entropy weight:

  ent         the fixed -0.03 * mean(entropy) becomes -mean(alpha_f(i) * entropy_i), one alpha per
              family. alpha_f starts at --ent-coef (0.03); after every parameter step
              log alpha_f += --ent-eta * (--ent-target - H_f), clipped to [--ent-coef, --ent-alpha-max].
              H_f = the family's mean of sequence_neglogp_entropy's entropy (gamma-0.7 decayed sum
              over the token steps, nats) over the rows that entered the gradient = BATCH_LOG /
              PROGRESS_16 families.<f>.entropy. A skipped step leaves alpha unchanged. alpha is in
              the checkpoint. At alpha_f == 0.03 for every family the loss is m2h_b5_trainer's.

m2h_b5_trainer header follows.

BASIN5 trainer (BASIN5_ORDER s1/s2/s6, 2026-10-12). Copy of m2h_dense_trainer_v2.py (sha12 in
LAUNCH_MANIFEST, NOT modified). Three changes, everything else is v2's:

  reward b    --reward gross_t (REWARD_PICK.json pick b): per formula, mean / sd (ddof 1) of the
              cost0 log NAV over the update's drawn tiles, full TOP100 pool (the ruler's book).
              sd == 0 (a book that never trades) is invalid (ZEROSD), like a zero-coverage one.
  grammar     m2h_b5_grammar.check before scoring: a level terminal that reaches a CS operator or
              the root without DELTA / TS_RANK / TS_ZSCORE / own-EMA ratio-diff is invalid
              (reason LEVEL_*, advantage -1 like every invalid formula, not scored).
  logs        BATCH_LOG per family also carries grammar_reject, corr_s (batch mean CS Spearman with
              the TV1 plant s on the first 4 drawn tiles; meaningful in the plant world only) and
              top/mode formulas. Every 16 updates one PROGRESS_16.jsonl row + one printed line: per
              family the best-reward formula and the most-sampled formula, each with n_tokens,
              level_flag and whether the step-2 grammar blocks it; entropy; corr_s.

v2 header follows.

M2H DENSE trainer v2 (TV1 R2b, user-approved 2026-10-10 18:30). Copy of m2h_dense_trainer.py
(sha12 80715f3d2e48, NOT modified) with two extra CLI knobs, defaulting to that file's own
behaviour so a run with no new flags is bit-identical:

  --ent-coef F    entropy weight, default 0.03 (m2h_trainer's)
  --adv-deadband F within-family advantage deadband: |z| < F is mapped to 0, so formulas the
                  reward cannot separate from the family mean stop pushing the policy. 0 =
                  off (m2h_dense_trainer's behaviour). Applied to BOTH signs of --select full.

Why (R2 evidence): entropy fell 6.03 -> 0.23 by update 80 and stayed there, the T1/s
correlation sat at 0.065 for all 256 updates, and the gradient norm never reached the 5.0 clip.
The policy collapsed to a deterministic argmax distribution, not to a bad direction: the
gradient direction is what the A table validated (0.676 corr with held-out gross). With
full-sample selection the mean |advantage| is ~0.8 vs ~0.1 for the old elite path, so the same
fixed -0.03 entropy bonus is relatively 8x weaker. R2b tests the two knobs together.

Original header:
M2H DENSE trainer (TV1 section 3; TV1_A2_BASIN4_ORDER). A copy of m2h_trainer.py (sha12
6969196607bb, not modified) with exactly three changes:

  a  reward   --reward gross (cost0 net log NAV) | ic (D.ic_tile) | C1 (A2: z(0.5 z(IC) +
              0.5 z(gross))) | net (flat6, control only). Cost is charged only in freeze and
              verdict. Mean over --tiles (default 16) drawn tiles, all formulas share the draw.
  b  hold     the HOLD step's mask keeps only H168 (on the sampling logits, so the stored
              logits the update backprops through carry the same mask).
  c  select   --select full: within-family z of the reward, clipped +-3, over VALID formulas
              only; an invalid formula's advantage is -1 and it is not in the mean / var; every
              formula enters the gradient. famq95 / globq95 keep risk-seeking selection
              (per-family = basin_risk_update's rule, whole batch = stage-1's) on reward/std.

Per update the BATCH_LOG row also carries, per family: batch gross bp/day, net bp/day, IC,
turnover/day, entropy (all formulas), plus the gradient norm (clip at 5.0). The optimiser,
policy, VM, ruler, entropy weight and checkpointing are m2h_trainer's.

m2h_trainer's own notes follow.

Copied from m2_trainer.py (stage-1) and changed only where the handoff requires it:

  grid     1 h instead of 1 min: 336 h tiles after a 672 h lookback, 50 disjoint TRAIN tiles,
           4 drawn per update, reward = mean over the 4. Lag is 1 ROW (m1_ruler.LAG is 2 minutes
           and m1_ruler is sha-pinned, so m2h_common re-reads the episode columns at offset 1).
  basins   three legality tables, one per family, built with available_terminals=BASIN_TERMS[f].
           A program therefore cannot reference another family's terminals, and its first token is
           inside its family by construction (at depth 0 only terminals are legal).
  quota    86/85/85 per update, the first token forced into the family by the table.
  elite    select_risk_elite is called ONCE PER FAMILY on that family's own rewards (q95), and the
           three elite sets are concatenated into ONE update. v4 is not modified: the function is
           imported and called, not redefined. Each family keeps its own baseline, so a family is
           never penalised for another family's scale.
  ruler    unchanged: m1_ruler.score_window / held_path / settle_multi, band 0.5, flat6 + cost0.

Everything else (the AlphaGPT policy and output-head bias, the stack VM, the stateful action
policy, Adam lr, the novelty weights, grad clip 5.0, checkpointing, resume, max-hours) is
stage-1's, including the rule that sampling must not run under no_grad.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem"
PANEL = os.path.join(ROOT, r"runtime\train2024_2025_panel_v1\TRAIN2024JAN_2025DEC_HOURLY54_TARGET_LONG_V1.parquet")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# m1_common / m1_ruler live in the sha-pinned stage-1 dir and pull in okx_alpha_discovery
# from ROOT/src; s1_env.bat set PYTHONPATH for m2_trainer, but this run is launched
# directly, so both paths are added here.
for _p in (r"D:\ChengboRemote\runtime\m1_stage1", os.path.join(ROOT, "src"),
           os.path.join(ROOT, "runtime")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LEGACY_ROOT = r"D:\v4r_groot\Chengbo\workspaces\alphagpt-rl-dl-base"
for _p in (LEGACY_ROOT, os.path.join(LEGACY_ROOT, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import m2h_common as M
import m2h_dense_common as D
import m2h_b5_grammar as G
import m1_common as C
import m1_ruler as R
from m1_sample import BodyView
from okx_alpha_discovery.adapters.xs_tscs_sampler_v2 import build_legality_table, sample_batch
from experiments.dso_reference.dso_math import select_risk_elite, sequence_neglogp_entropy

N_TILES_PER_UPDATE = 16
DENSE_HOLDS = (168,)          # change b
REWARDS = ("gross_t", "gross", "ic", "C1", "net")
S_FILE = os.path.join(r"D:\riskpg_stateful_v5s_governor_nextgen_beem", "artifacts", "m2h_v1",
                      "TV1_20261010", "TV1_S.npy")     # = m2h_plant.S_FILE
CORR_TILES = 4                # corr_s on the first 4 drawn tiles (tv1_round uses 4 too)
PROGRESS_EVERY = 16
SELECTS = ("full", "famq95", "globq95")
ELITE_RECHECK = 0            # off: the handoff's elite rule is per-update and per-family
REWARD_AGG = "mean"
REWARD_FLOOR = C.REWARD_FLOOR
TRAIN_COST = "flat6"
EPSILON = 0.05               # q95, the DSO higher-quantile selection

LEARNING_RATE = 5e-4
OP_BIAS = 1.5
END_BIAS = 2.0
CHECKPOINT_EVERY = 16
DIAG_KEYS = ("turnover_per_day", "net_exposure", "gross_exposure", "beta_mkt", "gross_bp_day")
BEHAV_TOL = 1e-5


# --------------------------------------------------------------------------- sampling


class _Batch:
    def __init__(self, formulas, context):
        self.formulas = formulas
        self.context = context

    @property
    def lengths(self):
        return self.context["lengths"]


def basin_token(vm, b):
    """Model-vocab id of basin b's token: after [body | END | HOLDs]. Never sampled; it only
    occupies the sampler's BOS input slot."""
    return vm.token_count + len(M.HOLDS) + M.BASIN_NAMES.index(b)


class BasinView(torch.nn.Module):
    """m1_sample.BodyView plus the forced basin token: the sampler's first input column (its
    all-zero BOS) is replaced by the basin token, so every body step conditions on the family.
    Logits are cut to the body vocabulary, as BodyView does."""

    def __init__(self, model, body_vocab, tok):
        super().__init__()
        self.model, self.body_vocab, self.tok = model, body_vocab, int(tok)

    def forward(self, inp):
        inp = inp.clone()
        inp[:, 0] = self.tok
        logits, a, b = self.model(inp)
        return logits[..., : self.body_vocab], a, b


def _merge_subs(subs, vm, device):
    """Stack per-family rollouts into one batch. The step axis is padded to the widest family
    exactly as the sampler pads its own finished rows: END token, force-END logits (0 on END,
    -inf elsewhere). That row contributes 0 to neglogp and 0 to entropy, like any post-END step."""
    S = max(s.context["tokens"].shape[1] for s in subs)
    V = vm.token_count
    toks, els, lens = [], [], []
    for s in subs:
        tk, el, ln = s.context["tokens"], s.context["effective_logits"], s.context["lengths"]
        n = tk.shape[0]
        if tk.shape[1] < S:
            force = torch.full((V,), float("-inf"), dtype=el.dtype, device=device)
            force[vm.end_token] = 0.0
            tk = torch.cat([tk, torch.full((n, S - tk.shape[1]), vm.end_token,
                                           dtype=torch.long, device=device)], dim=1)
            el = torch.cat([el, force[None, None, :].expand(n, S - el.shape[1], V)], dim=1)
        toks.append(tk)
        els.append(el)
        lens.append(ln)
    return torch.cat(toks, 0), torch.cat(els, 0), torch.cat(lens, 0)


def _append_hold(model, vm, tokens, el, lengths, basin_ids, device, generator):
    """One HOLD token after END, autograd live (m1_sample's hold step, holds in hours). The model
    input is [basin token, body tokens padded with END], as in the body steps."""
    B = tokens.shape[0]
    V = vm.token_count + len(M.HOLDS)
    L = int(lengths.max())
    inp = torch.full((B, L), vm.end_token, dtype=torch.long, device=device)
    pos = torch.arange(L, device=device)[None, :] < lengths[:, None]
    inp = torch.where(pos, tokens[:, :L], inp)
    inp = torch.cat([basin_ids[:, None], inp], dim=1)

    logits, _, _ = model(inp)
    step_logits = (logits[torch.arange(B, device=device), lengths]
                   if logits.dim() == 3 else logits)[:, :V]

    hold_mask = torch.full((V,), float("-inf"), device=device, dtype=step_logits.dtype)
    for h in DENSE_HOLDS:                                   # change b: H168 only
        hold_mask[vm.end_token + 1 + M.HOLDS.index(h)] = 0.0
    effective = step_logits + hold_mask[None, :]
    action = torch.multinomial(torch.softmax(effective, dim=-1), 1, generator=generator).squeeze(1)

    S = tokens.shape[1]
    tokens2 = torch.cat([tokens, torch.full((B, 1), vm.end_token, dtype=torch.long, device=device)], dim=1)
    put = lengths
    tokens2[torch.arange(B, device=device), put] = action
    lengths2 = lengths + 1

    if el.shape[-1] < V:
        el = torch.cat([el, torch.full((B, el.shape[1], V - el.shape[-1]), float("-inf"),
                                       device=device, dtype=el.dtype)], dim=2)
    force = torch.full((V,), float("-inf"), device=device, dtype=el.dtype)
    force[vm.end_token] = 0.0
    el2 = torch.cat([el, force[None, None, :].expand(B, 1, V)], dim=1).clone()
    el2[torch.arange(B, device=device), put] = effective

    formulas = [tuple(int(t) for t in row[:n])
                for row, n in zip(tokens2.tolist(), lengths2.tolist())]
    return _Batch(formulas=formulas, context={
        "tokens": tokens2, "effective_logits": el2, "lengths": lengths2,
        "max_len": S + 1, "hold_logits": effective, "hold_tokens": action})


def sample_batch_with_hold_basins(model, vm, tables, counts, *, device, generator=None,
                                  basins=M.BASIN_NAMES):
    """Stratified draw (86/85/85 at 256): one rollout per family, first input token forced to the
    family's basin token, terminals masked to the family by that family's legality table."""
    subs = [sample_batch(BasinView(model, vm.token_count, basin_token(vm, b)), vm, tab, int(n),
                         device=device, generator=generator)
            for b, tab, n in zip(basins, tables, counts)]
    tokens, el, lengths = _merge_subs(subs, vm, device)
    basin_ids = torch.cat([torch.full((int(n),), basin_token(vm, b), dtype=torch.long, device=device)
                           for b, n in zip(basins, counts)])
    return _append_hold(model, vm, tokens, el, lengths, basin_ids, device, generator)


def hold_token(vm, hold_hours):
    return vm.end_token + 1 + M.HOLDS.index(hold_hours)


def hold_of(vm, tokens):
    end = tokens.index(vm.end_token) if vm.end_token in tokens else len(tokens) - 2
    if end + 1 < len(tokens) and tokens[end + 1] > vm.end_token:
        return M.HOLDS[tokens[end + 1] - vm.end_token - 1]
    return M.HOLDS[0]


def strip_hold(vm, tokens):
    return tuple(tokens[: tokens.index(vm.end_token) + 1])


# --------------------------------------------------------------------------- update


def basin_risk_update(model, optimizer, batch, rewards, novelty, groups, device,
                      epsilon=EPSILON):
    """Per-family select_risk_elite, then ONE merged policy-gradient step.

    v4's novelty_weighted_risk_update selects the elite once over the whole batch, which on a
    three-family batch would let the strongest family take every elite slot. The handoff wants
    q95 inside each family, so select_risk_elite (imported from the pinned v4 / dso_math) is
    called once per family and the results are concatenated. Everything after selection is
    v4's update verbatim: novelty weights in [1,2] normalised to mean 1, advantage against the
    family's own baseline (the R_e baseline select_risk_elite returns), gamma-0.7 entropy
    bonus at -0.03, grad clip 5.0."""
    rewards = np.asarray(rewards, dtype=np.float64)
    novelty = np.asarray(novelty, dtype=np.float64)
    idxs, advs, per = [], [], []
    off = 0
    for name, n in groups:
        r = rewards[off:off + n]
        sel = select_risk_elite(r, epsilon)
        idxs.append(sel.elite_indices + off)
        advantage = np.asarray(sel.clipped_rewards, dtype=np.float64) - float(sel.baseline)
        advs.append(advantage)
        per.append(dict(basin=name, n=int(n), q95=float(sel.quantile),
                        reward_valid_std=float(np.std(r)) if len(r) else 0.0,
                        elite_count=int(len(sel.elite_indices)),
                        elite_advantage_mean=float(np.mean(advantage)) if len(advantage) else None))
        off += n
    if not any(len(a) for a in advs):
        return {}, per
    elite_np = np.concatenate(idxs)
    idx = torch.as_tensor(elite_np, dtype=torch.long, device=device)
    tokens = batch.context["tokens"].index_select(0, idx)
    logits = batch.context["effective_logits"].index_select(0, idx)
    advantage = torch.as_tensor(np.concatenate(advs), dtype=torch.float32, device=device)
    elite_novelty = torch.as_tensor(novelty[elite_np], dtype=torch.float32, device=device)

    weights = 1.0 + elite_novelty.clamp(0.0, 1.0)
    weights = weights / weights.mean().clamp_min(1e-6)
    lengths = torch.full((len(elite_np),), tokens.shape[1], dtype=torch.long, device=device)
    neglogp, entropy = sequence_neglogp_entropy(logits, tokens, lengths, 0.7)
    pg_loss = torch.mean(weights * advantage * neglogp)
    entropy_loss = -0.03 * torch.mean(entropy)
    loss = pg_loss + entropy_loss

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0,
                                               error_if_nonfinite=False)
    grad_norm_value = float(grad_norm.detach().item())
    skipped = not math.isfinite(grad_norm_value)
    if skipped:
        optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.step()

    # per-family elite neglogp mean, for the BATCH_LOG
    off = 0
    with torch.no_grad():
        nl = neglogp.detach().cpu().numpy()
    k = 0
    for p, a in zip(per, advs):
        p["elite_neglogp_mean"] = float(np.mean(nl[k:k + len(a)])) if len(a) else None
        k += len(a)

    upd = dict(quantile=None, elite_count=int(len(elite_np)),
               elite_fraction_actual=float(len(elite_np) / len(rewards)),
               pg_loss=float(pg_loss.detach().item()),
               entropy_loss=float(entropy_loss.detach().item()),
               mean_entropy=float(torch.mean(entropy).detach().item()),
               gradient_norm=grad_norm_value, gradient_clip_max_norm=5.0,
               skipped_nonfinite_update=skipped,
               elite_indices=[int(x) for x in elite_np])
    return upd, per


def dense_update(model, optimizer, batch, idx_np, adv_np, groups, device, ent_alpha):
    """Change c. The policy-gradient step of basin_risk_update after selection, verbatim
    (novelty weights 1, gamma-0.7 entropy, clip 5.0), on an explicit index set and
    advantage: full sample = every formula; globq95 = stage-1's whole-batch elite.
    ent: each row's entropy is weighted by its family's alpha (ent_alpha[k]).
    Also returns per-family mean entropy over the rows that entered."""
    fam_all = np.concatenate([[k] * int(n) for k, (_, n) in enumerate(groups)])
    fam_of = fam_all[np.asarray(idx_np)]
    idx = torch.as_tensor(np.asarray(idx_np), dtype=torch.long, device=device)
    tokens = batch.context["tokens"].index_select(0, idx)
    logits = batch.context["effective_logits"].index_select(0, idx)
    advantage = torch.as_tensor(np.asarray(adv_np), dtype=torch.float32, device=device)
    weights = torch.ones(len(idx_np), dtype=torch.float32, device=device)
    lengths = torch.full((len(idx_np),), tokens.shape[1], dtype=torch.long, device=device)
    neglogp, entropy = sequence_neglogp_entropy(logits, tokens, lengths, 0.7)
    pg_loss = torch.mean(weights * advantage * neglogp)
    alpha_rows = torch.as_tensor([float(ent_alpha[k]) for k in fam_of], dtype=torch.float32,
                                 device=device)
    entropy_loss = -torch.mean(alpha_rows * entropy)
    loss = pg_loss + entropy_loss
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0,
                                               error_if_nonfinite=False)
    grad_norm_value = float(grad_norm.detach().item())
    skipped = not math.isfinite(grad_norm_value)
    if skipped:
        optimizer.zero_grad(set_to_none=True)
    else:
        optimizer.step()
    ent = entropy.detach().cpu().numpy()
    fam_ent = [float(ent[fam_of == k].mean()) if (fam_of == k).any() else None
               for k in range(len(groups))]
    upd = dict(elite_count=int(len(idx_np)), elite_fraction_actual=float(len(idx_np) / len(fam_all)),
               pg_loss=float(pg_loss.detach().item()), entropy_loss=float(entropy_loss.detach().item()),
               mean_entropy=float(ent.mean()), gradient_norm=grad_norm_value,
               gradient_clip_max_norm=5.0, skipped_nonfinite_update=skipped)
    return upd, fam_ent


def behaviour_keep(vecs, groups, tol=BEHAV_TOL):
    vecs = [np.asarray(v, dtype=np.float64) for v in vecs]
    groups = np.asarray(groups)
    kept = []
    for i, v in enumerate(vecs):
        dup = False
        for j in kept:
            if groups[j] == groups[i] and vecs[j].shape == v.shape and \
                    float(np.abs(vecs[j] - v).max()) <= tol:
                dup = True
                break
        if not dup:
            kept.append(i)
    return kept


# --------------------------------------------------------------------------- plumbing


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]


def _hms(sec):
    sec = int(max(sec, 0))
    return "%d:%02d:%02d" % (sec // 3600, (sec % 3600) // 60, sec % 60)


def _save_checkpoint(path, **state):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def _latest_checkpoint(out_dir):
    cks = sorted(glob.glob(os.path.join(out_dir, "checkpoint_*.pt")))
    return cks[-1] if cks else None


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


def _root_summary(bodies, n, vm):
    from basin_common_v1 import root_distribution, summarize_roots
    return summarize_roots(root_distribution(bodies, [len(b) for b in bodies], vm), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_name")
    ap.add_argument("--panel", default=PANEL)
    ap.add_argument("--updates", type=int, default=128)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--reward", required=True, choices=REWARDS)
    ap.add_argument("--select", required=True, choices=SELECTS)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--tiles", type=int, default=N_TILES_PER_UPDATE)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--persist", type=int, default=1, choices=(0, 1))
    ap.add_argument("--cost", default=TRAIN_COST)
    ap.add_argument("--band", type=float, default=R.BAND)
    ap.add_argument("--nav-usd", type=float, default=R.NAV_USD)
    ap.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    ap.add_argument("--vm-block-elems", type=int, default=1 << 25,
                    help="VM trailing-window block size (elements); the blocked path is bit-identical")
    ap.add_argument("--ema-cache-max", type=int, default=1,
                    help="VM EMA kernel cache size; one float32 kernel is 0.5 GB at this slice width")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--manifest-only", action="store_true")
    # v2 only; both default to m2h_dense_trainer's own values
    ap.add_argument("--ent-coef", type=float, default=0.03,
                    help="entropy bonus weight; 0.03 in m2h_dense_trainer")
    ap.add_argument("--ent-target", type=float, default=2.0,
                    help="ent: per-family entropy floor H* (nats, gamma-0.7 sequence entropy)")
    ap.add_argument("--ent-eta", type=float, default=0.05,
                    help="ent: log alpha_f += eta * (H* - H_f) after every parameter step")
    ap.add_argument("--ent-alpha-max", type=float, default=1.0,
                    help="ent: alpha_f upper clip; the lower clip is --ent-coef")
    ap.add_argument("--adv-deadband", type=float, default=0.0,
                    help="within-family |z| below this is mapped to 0 advantage (full select)")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "artifacts", "m2h_v1", args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)

    from train_basin_riskpg_v1 import make_basin_model, apply_output_bias
    import model_core.alphagpt as alphagpt_mod
    from model_core.config import ModelConfig
    from model_core.vocab import FormulaVocab
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import (
        XsTsCsStackVM, OPERATOR_NAMES, CandidateLocalInvalid)
    import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vm_mod
    # The module defaults are BLOCK_ELEMS=0 (whole-panel unfold: a 672-window over a
    # 1009-column slice is ~1.3 GB per tensor) and EMA_KERNEL_CACHE_MAX=64 (0.5 GB each).
    # stage-1's trainer sets both; without them the peak on a 16 GiB card reached 15.2 GiB.
    vm_mod.BLOCK_ELEMS = int(args.vm_block_elems)
    vm_mod.EMA_KERNEL_CACHE_MAX = int(args.ema_cache_max)
    from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod
    ModelConfig.DEVICE = device

    # ---- data -------------------------------------------------------------
    syms, ts, grids = M.load_train_grids(args.panel)
    world = M.build_world(grids)
    feats_pad = M.pad_feats(world["feats"])
    univ, shard = world["univ"], world["shard"]
    all_starts = M.tile_starts(world["feats"].shape[2])
    n_tiles = len(all_starts)
    if n_tiles < args.tiles:
        raise SystemExit(f"panel holds only {n_tiles} tiles, need {args.tiles}")

    vm = XsTsCsStackVM(len(M.TERMINALS), windows=M.WINDOWS)
    torch.manual_seed(args.seed)
    op_names = (list(OPERATOR_NAMES) + [f"W{w}" for w in M.WINDOWS] + ["END"] + M.HOLD_NAMES
                + list(M.BASIN_TOKEN_NAMES))
    model, fvocab = make_basin_model(
        alphagpt_mod=alphagpt_mod, ModelConfig=ModelConfig, FormulaVocab=FormulaVocab,
        feature_names=list(M.TERMINALS), operator_token_names=op_names,
        max_len=M.MAX_PROGRAM_LEN, device=device)
    apply_output_bias(model, len(M.TERMINALS), len(OPERATOR_NAMES), len(M.WINDOWS),
                      OP_BIAS, END_BIAS)

    counts = M.quota(args.batch)
    tables = [build_legality_table(vm, max_len=M.MAX_PROGRAM_LEN,
                                   available_terminals=M.basin_terms(b),
                                   available_windows=M.WINDOWS).to(device)
              for b in M.BASIN_NAMES]
    groups = list(zip(M.BASIN_NAMES, counts))

    manifest = dict(schema="M2H_DENSE_LAUNCH_MANIFEST_V1", run=args.out_name,
                    v2=dict(ent_coef=args.ent_coef, adv_deadband=args.adv_deadband,
                            note="identical to m2h_dense_trainer when ent_coef=0.03 and "
                                 "adv_deadband=0; deadband applied to the within-family z of "
                                 "--select full only"),
                    dense=dict(reward=args.reward, select=args.select, holds=list(DENSE_HOLDS),
                               note="cost charged only in freeze / verdict; full = within-family "
                                    "z+-3 over valid, invalid -1, all formulas in the gradient"),
                    python=sys.executable, entry=os.path.abspath(__file__), args=vars(args),
                    modules=dict(m1_ruler=_sha(R.__file__), m1_common=_sha(C.__file__),
                                 m2h_common=_sha(M.__file__),
                                 stateful_action_policy=_sha(policy_mod.__file__)),
                    panel=args.panel, symbols=len(syms), hours=int(world["feats"].shape[2]),
                    n_tiles=n_tiles, tiles_per_update=args.tiles,
                    vocab_size=fvocab.size, holds=list(M.HOLDS), windows=list(M.WINDOWS),
                    basins={b: dict(terminals=[M.TERMINALS[i] for i in M.basin_terms(b)],
                                    quota=counts[k]) for k, b in enumerate(M.BASIN_NAMES)},
                    grid=dict(step_hours=1, lag_rows=M.LAG, episode_h=M.EPISODE,
                              lookback_h=M.LOOKBACK, notes="m1_ruler.LAG is 2 minutes and is "
                              "sha-pinned; m2h_common reads the episode columns at offset 1"),
                    universe=dict(top_n=M.UNIV_N, window_h=M.UNIV_WIN, min_h=M.UNIV_MIN,
                                  rank_method="first", applied_to="formula output at the signal row"),
                    ruler=dict(module="m1_ruler", functions=["score_window", "held_path",
                               "settle_multi", "rank_weights", "prepare_shard"],
                               winsor_pct=R.WINSOR_PCT, band=args.band,
                               positions="C.rank_weights, sum w = 0, sum|w| = 1"),
                    reward=dict(aggregate=REWARD_AGG, floor=REWARD_FLOOR, cost=args.cost,
                                definition=f"{REWARD_AGG} over {args.tiles} drawn disjoint "
                                           f"{M.EPISODE} h tiles of net log NAV ({args.cost}, "
                                           "funding charged, lag 1, per-formula HOLD)",
                                declared_choice="the handoff says 'draw 4 tiles, take the mean'; "
                                                "no SELECT/HOLDOUT split, so all drawn tiles "
                                                "enter the gradient",
                                banned_aggregates=["second_best", "max", "median"]),
                    elite=dict(epsilon=EPSILON, per_family=True,
                               rule="select_risk_elite called once per family (v4 unmodified); "
                                    "elite sets concatenated into one update; each family keeps "
                                    "its own baseline"),
                    model_params=sum(p.numel() for p in model.parameters()),
                    checkpoint=dict(every=args.checkpoint_every, max_hours=args.max_hours),
                    written_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    manifest["modules"]["trainer"] = _sha(os.path.abspath(__file__))
    manifest["modules"]["m2h_b5_grammar"] = _sha(G.__file__)
    manifest["modules"]["m2h_dense_common"] = _sha(D.__file__)
    gself = G.selftest(vm)
    if not gself["ok"]:
        raise SystemExit("grammar selftest failed: %s" % [c for c in gself["cases"] if not c["pass_"]])
    s_plant = np.load(S_FILE) if os.path.exists(S_FILE) else None
    if s_plant is not None and s_plant.shape != tuple(world["fwd"].shape):
        raise SystemExit(f"plant s shape {s_plant.shape} != fwd {world['fwd'].shape}")
    manifest["b5"] = dict(
        order="BASIN5_ORDER s1/s2/s6", copied_from="m2h_dense_trainer_v2.py",
        reward_def=("gross_t = mean/sd(ddof 1) of per-tile cost0 log NAV over the drawn tiles, "
                    "full TOP100 pool; sd==0 -> invalid ZEROSD" if args.reward == "gross_t"
                    else args.reward),
        grammar=dict(file="m2h_b5_grammar.py", selftest_ok=True,
                     selftest=[(c["expr"], c["want_legal"], c["reason"]) for c in gself["cases"]]),
        corr_s=dict(s_file=S_FILE if s_plant is not None else None, tiles=CORR_TILES,
                    note="CS Spearman with the TV1 plant s; meaningful only in the plant world"),
        progress_every=PROGRESS_EVERY)
    manifest["ent"] = dict(
        order="BASIN5_PC_ENT_ORDER", copied_from="m2h_b5_trainer.py",
        alpha_init=args.ent_coef, alpha_min=args.ent_coef, alpha_max=args.ent_alpha_max,
        eta=args.ent_eta, target=args.ent_target,
        rule="after every parameter step: log a_f += eta*(H* - H_f), clip [alpha_min, alpha_max]; "
             "skipped step -> a_f unchanged; loss entropy term = -mean_i(a_f(i) * H_i)",
        entropy_def="sequence_neglogp_entropy(gamma 0.7): per formula, sum over token steps "
                    "t < MAX_PROGRAM_LEN of gamma^t * H(softmax(effective_logits_t)), nats; "
                    "post-END steps are one-hot (H 0). H_f = family mean over rows in the gradient",
        vocab_size=fvocab.size, log_vocab=round(math.log(fvocab.size), 4),
        per_step_max_sum=round(math.log(fvocab.size) / (1 - 0.7), 4))
    if args.max_hours and not args.persist:
        raise SystemExit("--max-hours needs --persist 1")
    if args.ent_coef < 0.0 or args.adv_deadband < 0.0:
        raise SystemExit("--ent-coef and --adv-deadband must be >= 0")
    if not 0.0 < args.ent_coef <= args.ent_alpha_max or args.ent_eta < 0.0:
        raise SystemExit("ent: need 0 < --ent-coef <= --ent-alpha-max and --ent-eta >= 0")
    if args.select == "famq95":
        raise SystemExit("ent: famq95 goes through basin_risk_update's fixed 0.03; not wired")

    ck = None
    if args.resume:
        ck_path = _latest_checkpoint(out_dir) if args.resume == "auto" else args.resume
        if not ck_path or not os.path.exists(ck_path):
            raise SystemExit(f"--resume {args.resume}: no checkpoint in {out_dir}")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        free = {"updates", "max_hours", "resume", "checkpoint_every", "manifest_only"}
        diff = {k: (ck["args"].get(k), v) for k, v in vars(args).items()
                if k not in free and ck["args"].get(k) != v}
        if diff:
            raise SystemExit(f"--resume refused, args differ: {diff}")
        if ck["modules"] != manifest["modules"]:
            raise SystemExit(f"--resume refused, code differs: {ck['modules']} vs {manifest['modules']}")
        manifest["resumed_from"] = dict(checkpoint=ck_path, update=int(ck["update"]))
    json.dump(manifest, open(os.path.join(out_dir, "LAUNCH_MANIFEST.json" if ck is None
              else "RESUME_MANIFEST_%s.json" % time.strftime("%Y%m%dT%H%M%S")), "w"), indent=1)
    if args.manifest_only:
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    starts_rng = np.random.default_rng(args.seed)
    n_evaluated = 0
    reward_pool, reward_sq_pool = [], []
    invalid_reasons = collections.Counter()
    elite_pool = []
    start_update, elapsed_prev = 0, 0.0
    fmode = "w"
    ent_alpha = [float(args.ent_coef)] * len(groups)          # ent: one alpha per family
    if ck is not None:
        ent_alpha = [float(x) for x in ck["ent_alpha"]]
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        n_evaluated = int(ck["n_evaluated"])
        reward_pool, reward_sq_pool = list(ck["reward_pool"]), list(ck["reward_sq_pool"])
        invalid_reasons = collections.Counter(ck["invalid_reasons"])
        elite_pool = list(ck["elite_pool"])
        starts_rng.bit_generator.state = ck["starts_rng_state"]
        torch.set_rng_state(ck["torch_rng_state"])
        if device.type == "cuda" and ck.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng_state"], device)
        start_update, elapsed_prev = int(ck["update"]) + 1, float(ck["elapsed_s"])
        fmode = "a"
        for fn in ("BATCH_LOG.jsonl", "FORMULAS.jsonl", "PROGRESS_16.jsonl"):
            print(f"resume: {fn} dropped {_truncate_jsonl(os.path.join(out_dir, fn), int(ck['update']))} rows",
                  flush=True)
        print(f"resumed from update {start_update}, N={n_evaluated}", flush=True)
    log_fh = open(os.path.join(out_dir, "BATCH_LOG.jsonl"), fmode, encoding="utf-8")
    formulas_fh = open(os.path.join(out_dir, "FORMULAS.jsonl"), fmode, encoding="utf-8") if args.persist else None
    t0 = time.time() - elapsed_prev
    t_sess0 = time.time()
    stopped_early = False
    stop_json = os.path.join(out_dir, "STOPPED.json")
    if os.path.exists(stop_json):
        os.replace(stop_json, os.path.join(out_dir, "STOPPED_prev_%s.json" % time.strftime("%Y%m%dT%H%M%S")))
    vram = None
    t_upd_last = None

    for update in range(start_update, args.updates):
        t_upd0 = time.time()
        draw = np.sort(starts_rng.choice(n_tiles, size=args.tiles, replace=False))
        starts = [all_starts[int(i)] for i in draw]
        slices = M.tile_slices(feats_pad, starts, device)

        batch = sample_batch_with_hold_basins(model, vm, tables, counts, device=device)
        tokens_list = batch.formulas

        rewards = np.full(args.batch, REWARD_FLOOR, dtype=np.float64)
        live_frac = np.full(args.batch, np.nan)
        diag = {k: np.full(args.batch, np.nan) for k in DIAG_KEYS}
        hold_logs = [None] * args.batch
        hold_hist = np.zeros(len(M.HOLDS), dtype=int)
        basin_of = []
        off = 0
        for b, n in groups:
            basin_of += [b] * n
            off += n
        valid_mask = np.zeros(args.batch, dtype=bool)
        n_valid = 0
        # change a: per formula, mean over the drawn tiles of gross (cost0), net (args.cost)
        # and IC; the training reward is built from these below
        gross = np.full(args.batch, np.nan)
        net = np.full(args.batch, np.nan)
        icm = np.full(args.batch, np.nan)
        tiles_g, tiles_ic = [None] * args.batch, [None] * args.batch
        gt = np.full(args.batch, np.nan)                    # b5: gross tile t
        corr = np.full(args.batch, np.nan)                  # b5: corr with plant s
        gram_bad = np.zeros(args.batch, dtype=bool)
        bodies = []
        for i, toks in enumerate(tokens_list):
            hold_h = hold_of(vm, list(toks))
            hold_hist[M.HOLDS.index(hold_h)] += 1
            body = list(strip_hold(vm, list(toks)))
            bodies.append(tuple(body))
            invalid_reason = None
            logs = None
            try:
                if hold_h not in DENSE_HOLDS:
                    raise RuntimeError(f"hold mask leaked: H{hold_h}")
                gram = G.check(vm, body)                    # b5: step-2 grammar, before scoring
                if gram is not None:
                    gram_bad[i] = True
                    raise CandidateLocalInvalid(gram)
                res = D.score_dense(vm, body, slices, starts, univ, shard, policy_mod,
                                    holds=(hold_h,), costs=(args.cost, "cost0"),
                                    band=args.band, nav_usd=args.nav_usd, want_ic=True,
                                    want_sig=s_plant is not None)
                logs = [float(x) for x in res["per_hold"][hold_h][args.cost]]
                tiles_g[i] = [float(x) for x in res["per_hold"][hold_h]["cost0"]]
                tiles_ic[i] = [float(x) for x in res["ic"]]
                sd = float(np.std(tiles_g[i], ddof=1)) if len(tiles_g[i]) > 1 else 0.0
                if args.reward == "gross_t" and not sd > 0.0:
                    raise CandidateLocalInvalid("ZEROSD_GROSS_TILES")
                gt[i] = float(np.mean(tiles_g[i])) / sd if sd > 0.0 else np.nan
                if s_plant is not None:
                    corr[i] = float(np.mean([D.corr_with_s(g_, s_plant, s0_) for g_, s0_ in
                                             zip(res["sig"][:CORR_TILES], starts[:CORR_TILES])]))
                    res["sig"] = None
                gross[i] = float(np.mean(tiles_g[i]))
                net[i] = float(np.mean(logs))
                icm[i] = float(np.mean(tiles_ic[i]))
                live_frac[i] = res["live_frac"]
                for k, v in M.diagnostics(res["parts"][hold_h]).items():
                    diag[k][i] = v
                valid_mask[i] = True
                n_valid += 1
            except (M.ZeroCoverage, CandidateLocalInvalid) as exc:
                invalid_reason = str(exc)
                invalid_reasons[invalid_reason.split("_")[0]] += 1
            if formulas_fh is not None:
                formulas_fh.write(json.dumps(dict(
                    update=update, basin=basin_of[i], tokens=list(toks),
                    expr=vm.describe(body, list(M.TERMINALS)), hold_h=hold_h,
                    gross=round(float(gross[i]), 7) if valid_mask[i] else None,
                    net=round(float(net[i]), 7) if valid_mask[i] else None,
                    ic=round(float(icm[i]), 6) if valid_mask[i] else None,
                    window_gross=[round(x, 7) for x in tiles_g[i]] if valid_mask[i] else None,
                    window_ic=[round(x, 6) for x in tiles_ic[i]] if valid_mask[i] else None,
                    live_frac=round(float(live_frac[i]), 4) if np.isfinite(live_frac[i]) else None,
                    turnover_per_day=round(float(diag["turnover_per_day"][i]), 3)
                    if np.isfinite(diag["turnover_per_day"][i]) else None,
                    beta_mkt=round(float(diag["beta_mkt"][i]), 4)
                    if np.isfinite(diag["beta_mkt"][i]) else None,
                    gross_bp_day=round(float(diag["gross_bp_day"][i]), 3)
                    if np.isfinite(diag["gross_bp_day"][i]) else None,
                    window_nav=[round(float(x), 7) for x in logs] if valid_mask[i] else None,
                    gross_t=round(float(gt[i]), 5) if valid_mask[i] and np.isfinite(gt[i]) else None,
                    corr_s=round(float(corr[i]), 4) if valid_mask[i] and np.isfinite(corr[i]) else None,
                    n_tokens=len(body) - 1,
                    invalid_reason=invalid_reason)) + "\n")
        if formulas_fh is not None:
            formulas_fh.flush()

        # change a: the training reward (raw, valid formulas); cost stays out of it
        if args.reward == "gross_t":
            raw = gt
        elif args.reward == "gross":
            raw = gross
        elif args.reward == "ic":
            raw = icm
        elif args.reward == "net":
            raw = net
        else:                                               # C1 (A2)
            raw = D.combo(icm, gross, valid_mask, counts)
        rewards[valid_mask] = raw[valid_mask]
        hold_logs = [tiles_g[i] for i in range(args.batch)]
        reward_valid_std = float(rewards[valid_mask].std()) if n_valid else 0.0
        skip_reason = None
        if n_valid < 20:
            skip_reason = f"n_valid={n_valid}<20"
        elif reward_valid_std == 0.0:
            skip_reason = "reward_valid_std==0"
        fam_ent = [None] * len(groups)
        if skip_reason is not None:
            upd, per_family = {}, [dict(basin=b, n=n) for b, n in groups]
        elif args.select == "full":
            # change c: within-family z clipped +-3 over VALID formulas; invalid -> -1, not in
            # the mean / var; every formula of the batch enters the gradient
            adv = D.zfam(rewards, valid_mask, counts)
            adv[~valid_mask] = -1.0
            if args.adv_deadband > 0.0:                    # v2: |z| below the band stops pushing
                adv = np.where(np.abs(adv) < args.adv_deadband, 0.0, adv)
            upd, fam_ent = dense_update(model, optimizer, batch, np.arange(args.batch), adv,
                                        groups, device, ent_alpha)
            upd["z_count"] = int(np.isfinite(D.zfam(rewards, valid_mask, counts)).sum())
            per_family = [dict(basin=b, n=n) for b, n in groups]
        else:
            scaled = np.empty(args.batch)
            scaled[valid_mask] = rewards[valid_mask] / reward_valid_std
            scaled[~valid_mask] = float(scaled[valid_mask].min()) - 1.0
            if args.select == "famq95":
                upd, per_family = basin_risk_update(model, optimizer, batch, scaled,
                                                    np.ones(args.batch), groups, device)
            else:                                           # globq95: stage-1's whole batch
                sel = select_risk_elite(scaled, EPSILON)
                upd, fam_ent = dense_update(model, optimizer, batch, sel.elite_indices,
                                            np.asarray(sel.clipped_rewards) - sel.baseline,
                                            groups, device, ent_alpha)
                per_family = [dict(basin=b, n=n) for b, n in groups]
        # ent: the alpha used by this step is logged; the step then moves alpha towards H*
        alpha_used = list(ent_alpha)
        if skip_reason is None and not upd.get("skipped_nonfinite_update"):
            for k in range(len(groups)):
                if fam_ent[k] is not None:
                    a_ = math.exp(math.log(ent_alpha[k]) + args.ent_eta * (args.ent_target - fam_ent[k]))
                    ent_alpha[k] = min(max(a_, args.ent_coef), args.ent_alpha_max)
        # per-family batch stats (TV1 section 6, BASIN4 log): gross / net bp/day, IC,
        # turnover, entropy of the rows that entered the gradient
        off = 0
        for k, (p, (b, n)) in enumerate(zip(per_family, groups)):
            m = valid_mask[off:off + n]
            sl = slice(off, off + n)
            p["n_valid"] = int(m.sum())
            p["gross_bpd"] = round(D.bpd(np.mean(gross[sl][m])), 3) if m.any() else None
            p["net_bpd"] = round(D.bpd(np.mean(net[sl][m])), 3) if m.any() else None
            p["ic"] = round(float(np.mean(icm[sl][m])), 5) if m.any() else None
            p["turnover_per_day"] = round(float(np.nanmean(diag["turnover_per_day"][sl][m])), 3) \
                if m.any() else None
            p["entropy"] = round(fam_ent[k], 4) if fam_ent[k] is not None else p.get("entropy")
            p["ent_alpha"] = round(alpha_used[k], 6)              # ent: alpha this step used
            p["ent_alpha_next"] = round(ent_alpha[k], 6)
            # b5 step 6: grammar rejects, corr with plant s, best and most-sampled formula
            p["grammar_reject"] = int(gram_bad[sl].sum())
            c_ = corr[sl][m]
            p["corr_s"] = round(float(np.nanmean(c_)), 4) if m.any() and np.isfinite(c_).any() else None
            p["gross_t"] = round(float(np.nanmean(gt[sl][m])), 4) if m.any() else None
            fam_idx = np.arange(off, off + n)
            if m.any():
                bi = int(fam_idx[m][np.argmax(rewards[sl][m])])
                p["top"] = dict(expr=vm.describe(list(bodies[bi]), list(M.TERMINALS)),
                                n_tokens=len(bodies[bi]) - 1, reward=round(float(rewards[bi]), 5),
                                corr_s=round(float(corr[bi]), 4) if np.isfinite(corr[bi]) else None,
                                level_flag=bool(D.level_flag(vm, list(bodies[bi]))),
                                grammar_blocks=False)
            mc = collections.Counter(bodies[j] for j in fam_idx)
            mb, mcount = mc.most_common(1)[0]
            g_ = G.check(vm, list(mb))
            p["mode"] = dict(expr=vm.describe(list(mb), list(M.TERMINALS)), n_tokens=len(mb) - 1,
                             share=round(mcount / max(n, 1), 4), level_flag=bool(D.level_flag(vm, list(mb))),
                             grammar_blocks=g_ is not None, grammar_reason=g_)
            off += n
        # raw (unscaled, valid-only) family stats next to the scaled ones the update used
        off = 0
        for p, (b, n) in zip(per_family, groups):
            m = valid_mask[off:off + n]
            r = rewards[off:off + n][m]
            p["n_valid"] = int(m.sum())
            p["reward_mean_raw"] = round(float(r.mean()), 6) if m.any() else None
            p["reward_valid_std_raw"] = round(float(r.std()), 6) if m.any() else None
            p["q95_raw"] = round(float(np.quantile(r, 0.95, method="higher")), 6) if m.any() else None
            off += n

        n_evaluated += len(tokens_list)
        if n_valid:
            reward_pool.append(float(rewards[valid_mask].mean()))
            reward_sq_pool.append(float((rewards[valid_mask] ** 2).mean()))
        order = [int(i) for i in np.argsort(-rewards) if valid_mask[i]]
        keep = behaviour_keep([hold_logs[i] for i in order], [0] * len(order))
        cands = list(elite_pool) + [
            (tuple(strip_hold(vm, list(tokens_list[order[p]]))), basin_of[order[p]],
             vm.describe(list(strip_hold(vm, list(tokens_list[order[p]]))), list(M.TERMINALS)),
             float(rewards[order[p]])) for p in keep]
        cands.sort(key=lambda e: -e[3])
        elite_pool = cands[:200]

        t_upd_last = time.time() - t_upd0
        row = dict(update=update, seconds=round(t_upd_last, 2),
                   n_valid=n_valid, n_evaluated_cum=n_evaluated,
                   reward_mean=round(float(rewards[valid_mask].mean()), 6) if n_valid else REWARD_FLOOR,
                   reward_positive_fraction=round(float((rewards[valid_mask] > 0).mean()), 4) if n_valid else None,
                   skipped_param_step=skip_reason, reward_valid_std=round(reward_valid_std, 6),
                   tiles=starts, hold_hist=hold_hist.tolist(), hold_names=M.HOLD_NAMES,
                   live_frac_mean=round(float(np.nanmean(live_frac[valid_mask])), 4) if n_valid else None,
                   turnover_per_day_mean=round(float(np.nanmean(diag["turnover_per_day"][valid_mask])), 3) if n_valid else None,
                   abs_net_exposure_mean=round(float(np.nanmean(np.abs(diag["net_exposure"][valid_mask]))), 5) if n_valid else None,
                   beta_mkt_abs_mean=round(float(np.nanmean(np.abs(diag["beta_mkt"][valid_mask]))), 4) if n_valid else None,
                   gross_bp_day_mean=round(float(np.nanmean(diag["gross_bp_day"][valid_mask])), 3) if n_valid else None,
                   elite_pool_size=len(elite_pool),
                   per_family=per_family,
                   **{k: upd.get(k) for k in ("pg_loss", "entropy_loss", "gradient_norm",
                                              "elite_count", "z_count", "mean_entropy",
                                              "skipped_nonfinite_update") if k in upd})
        if device.type == "cuda":
            vram = torch.cuda.max_memory_allocated() / 2**30
            row["vram_alloc_gib"] = round(vram, 3)
        log_fh.write(json.dumps(row) + "\n")
        log_fh.flush()
        json.dump(dict(pid=os.getpid(), update=update, updates_total=args.updates,
                       seconds=round(t_upd_last, 2),
                       updated=time.strftime("%Y-%m-%dT%H:%M:%S%z")),
                  open(os.path.join(out_dir, "HEARTBEAT.json"), "w"), indent=1)
        if (update + 1) % PROGRESS_EVERY == 0 or update == args.updates - 1:   # b5 step 6
            prog = dict(update=update, entropy_all=upd.get("mean_entropy"),
                        gradient_norm=upd.get("gradient_norm"), n_valid=n_valid,
                        families={p["basin"]: {k: p.get(k) for k in (
                            "n_valid", "grammar_reject", "entropy", "ent_alpha", "ent_alpha_next",
                            "corr_s", "gross_t",
                            "gross_bpd", "net_bpd", "ic", "top", "mode")} for p in per_family},
                        written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            with open(os.path.join(out_dir, "PROGRESS_16.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(prog, ensure_ascii=False) + "\n")
            for p in per_family:
                t_, md = p.get("top") or {}, p.get("mode") or {}
                print("P16 u%d %s ent %s a %s corr %s rej %s | top[%s tok lf%d] %s | mode[%.2f %s tok blk%d] %s"
                      % (update, p["basin"], p.get("entropy"), p.get("ent_alpha"), p.get("corr_s"),
                         p.get("grammar_reject"),
                         t_.get("n_tokens"), int(bool(t_.get("level_flag"))), str(t_.get("expr"))[:90],
                         md.get("share", 0.0), md.get("n_tokens"), int(bool(md.get("grammar_blocks"))),
                         str(md.get("expr"))[:90]), flush=True)
        if update < start_update + 4 or (update + 1) % 16 == 0 or update == args.updates - 1:
            eta = (time.time() - t_sess0) / (update + 1 - start_update) * (args.updates - update - 1)
            fam = " ".join("%s g%s t%s c%s r%s" % (p.get("basin"), p.get("gross_bpd"), p.get("gross_t"),
                                                    p.get("corr_s"), p.get("grammar_reject"))
                           for p in per_family)
            print("upd %3d %5.1fs valid %3d/%d | %s | gn %s vram %s eta %s"
                  % (update, t_upd_last, n_valid, args.batch, fam,
                     None if upd.get("gradient_norm") is None else round(upd["gradient_norm"], 2),
                     round(vram, 2) if vram else "-", _hms(eta)), flush=True)

        over_budget = bool(args.max_hours) and (time.time() - t0) >= args.max_hours * 3600.0
        last_update = update == args.updates - 1
        if args.persist and ((update + 1) % args.checkpoint_every == 0 or over_budget):
            _save_checkpoint(os.path.join(out_dir, f"checkpoint_{update:04d}.pt"),
                             model=model.state_dict(), optimizer=optimizer.state_dict(),
                             update=update, args=vars(args), modules=manifest["modules"],
                             elite_pool=elite_pool, n_evaluated=n_evaluated,
                             reward_pool=reward_pool, reward_sq_pool=reward_sq_pool,
                             invalid_reasons=dict(invalid_reasons),
                             starts_rng_state=starts_rng.bit_generator.state,
                             torch_rng_state=torch.get_rng_state(),
                             cuda_rng_state=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                             elapsed_s=time.time() - t0, ent_alpha=list(ent_alpha))
        if over_budget and not last_update:
            stopped_early = True
            json.dump(dict(reason="max_hours", max_hours=args.max_hours, last_update=update,
                           next_update=update + 1, updates_total=args.updates,
                           elapsed_s=round(time.time() - t0, 1),
                           resume="--resume auto (same arguments; --updates/--max-hours may change)",
                           written=time.strftime("%Y-%m-%dT%H:%M:%S%z")),
                      open(os.path.join(out_dir, "STOPPED.json"), "w"), indent=1)
            print(f"max_hours {args.max_hours} reached after update {update}; checkpointed, stopping",
                  flush=True)
            break

    log_fh.close()
    if formulas_fh is not None:
        formulas_fh.close()
    if stopped_early:
        return
    with torch.no_grad():
        final = sample_batch_with_hold_basins(model, vm, tables, counts, device=device)
    receipt = dict(schema="M2H_TRAIN_RECEIPT_V1", run=args.out_name, b5=manifest["b5"],
                   ent=dict(manifest["ent"], alpha_final=list(ent_alpha)),
                   created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), config=vars(args),
                   n_tiles=n_tiles, invalid_reasons=dict(invalid_reasons),
                   elite_pool_final_size=len(elite_pool),
                   trained_policy_final=dict(
                       hold_hist={h: sum(1 for t in final.formulas
                                         if hold_of(vm, list(t)) == h) for h in M.HOLDS},
                       exprs=[vm.describe(list(strip_hold(vm, list(t))), list(M.TERMINALS))
                              for t in final.formulas[:8]]),
                   peak_vram_alloc_gib=round(vram, 3) if vram else None,
                   compute_seconds=round(time.time() - t0, 1), oos_read=False, dev_read=False)
    json.dump(receipt, open(os.path.join(out_dir, "TRAIN_RECEIPT_V1.json"), "w"), indent=1)
    print("done", json.dumps(dict(updates=args.updates, seconds=receipt["compute_seconds"])))


if __name__ == "__main__":
    main()
