"""XS_SEL1 trainer x = sel1_trainer.py (sha16 89b4e4ac2abddc00, NOT modified; the V2 chain runs that file) plus
switches that default off, so a run without them is sel1_trainer's (supplement 10-15 afternoon / evening):
  --dedup    behaviour identity = sel1_dedup.behaviour_hash of the A-half selections (the long / short coin sets of
             every trading A hour). A body whose identity was seen before (an earlier update, or an earlier row of
             this update) is invalid DUP_* (advantage -1), not scored; the same body again is DUP_BODY without a VM
             call. The family z uses sel1_dedup.zfam (std < 1e-12 -> the family's advantages are 0). --combo-sweep only.
  --pool N   AlphaGen pool reward (evening item 5 + correction), sel1_pool composites. Candidate f, rho = max corr of
             its A-half monthly cost0 (own t_dd book) with the members', j = that member:
               rho <= 0.7  gain = t_dd(pool + f) - t_dd(pool);  rho > 0.7  gain = t_dd(pool - j + f) - t_dd(pool);
               f selects exactly what a member selects (behaviour hash) -> invalid POOL_SAME.
             t_dd(empty pool) = 0. Advantage = gain / std(valid gains) clipped +-3 (std < 1e-12 -> 0), invalid -1,
             no mean subtracted. After the update the best gain > 0 is executed (add / replace; a full pool drops the
             one of 21 whose removal leaves the highest t_dd, newcomer included). Hall / single gate unchanged.
             POOL_LOG.jsonl per update, POOL.json at the end. Needs --combo-sweep --gpu-select --gpu-workers 0.
  --gfn B    trajectory balance (sel1_gfn): loss (log Z + sum log pi - B t_dd)^2, invalid t_dd = -3; replaces the
             policy-gradient step. --frozen: no update at all (random baseline, policy stays at u0).
  --plant2 F second plant (sel1_world2, SEL1_ARMS world): F = json with body / alpha; --mode pcnull only.
  --fork     resume a checkpoint with --pool / --dedup / --gfn / --frozen / --plant2 changed (self-checks, ARMS).
  --receipt-on-stop  a --max-hours stop also writes HOF.json and the receipt (stopped=true).

sel1_trainer header follows.

XS_SEL1 trainer: sparse cross-sectional selection book (order XS_SEL1 s1-s4; ADR-0012; contract V4). New file.
The grammar-masked sampler, the basin view, the policy-gradient step and the checkpoint plumbing are
m2h_b5_trainer_dbg_memo's functions, imported unchanged; the world and the book are sel1_common's.

  action   body (B5 grammar, basin token T1/T2/T3 with its terminals, quota 43/43/42 of 128) + three appended
           tokens with autograd live: hold {4,24,72,168}, state switch (5), direction {both,long,short}.
  reward   A half (V4 odd months, 30 windows): --reward mean (default, the original) = cost0 beta-hedged log NAV
           of the book, bp/day; t = monthly t of it; t_dd = t, 0 above the A-half max drawdown --dd-max
           (S.reward). Logs keep reward_bpd = cost0 bp/day and add score when --reward is not mean.
           Within-family z +-3 over valid candidates, invalid -1; dense update (dbg_memo, every candidate),
           entropy coefficient ent_coef.
  floor    entropy floor: floor = ent_floor_frac x the batch mean entropy of update 0; while an update's mean
           entropy is under the floor the coefficient is multiplied by 1.5 (cap 0.3), otherwise it decays by
           1/1.1 back towards ent_coef. State kept in the checkpoint.
  hall     every valid candidate (body, hold, state, direction) with its A reward (deterministic: all 30 A
           windows every time), top 5000 kept; HOF.json = top 500 at the end. B is never computed here.
  speed    --workers 0 --gpu-workers 0 --memo update: everything in this process, each body / candidate scored
           once per update (the baseline). --memo global: a candidate's reward and a body's selections are
           deterministic, so they are cached across updates. --workers N: book scores in N CPU processes;
           --gpu-workers G: the VM in G GPU processes. None of these changes a result: the zero-diff check
           resumes the baseline's checkpoint with the fast settings and compares the logs line by line.
  log      BATCH_LOG.jsonl per update, FORMULAS.jsonl per candidate; every 64 updates: distinct-formula share
           and the most frequent formula per family.

    python -I sel1_trainer.py OUT --mode real|null|pc|pcnull --updates N [--k K] [--alpha A]
                              [--reward mean|t|t_dd --dd-max D] [--resume auto]
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
import sel1_common as S
import numpy as np
import torch
import cr1_common as Q
import m2h_b5_trainer_dbg_memo as TB
import m2h_common as M
import m2h_dense_common as D
import m2h_b5_grammar as G
import m1_ruler as R
import v4_world as X
import sel1_cse
import sel1_fastts
import sel1_sweep as W
import sel1_dedup as DD
import sel1_pool as P
import sel1_gfn as GF
import sel1_world2 as W2
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid, OPERATOR_NAMES
from okx_alpha_discovery.adapters.xs_tscs_sampler_v2 import build_legality_table

LEARNING_RATE, OP_BIAS, END_BIAS = TB.LEARNING_RATE, TB.OP_BIAS, TB.END_BIAS
STATE_NAMES = tuple("S_" + s for s in S.STATES)
DIR_NAMES = tuple("D_" + d for d in S.DIRS)
FREE_ARGS = {"updates", "max_hours", "resume", "checkpoint_every", "workers", "gpu_workers", "memo", "cse_gb",
             "speed_resume", "fastts", "gpu_select"}
SPEED_MODULES = {"sel1_trainer", "sel1_common", "sel1_cse", "sel1_fastts", "sel1_dedup", "sel1_pool", "sel1_gfn",
                 "sel1_world2"}   # --speed-resume: may differ, logs then compared line by line
LEGACY_ARGS = dict(reward="mean", dd_max=0.0, k=5, alpha=None, combo_sweep=False, dedup=False, pool=0, gfn=None,
                   frozen=False, plant2=None, receipt_on_stop=False, fork=False)
FORK_ARGS = {"pool", "dedup", "gfn", "frozen", "plant2", "receipt_on_stop", "fork"}
POOL_RHO = 0.7
REPORT_EVERY = 64
ENT_CAP, ENT_UP, ENT_DOWN = 0.3, 1.5, 1.1


def modules():
    return dict(sel1_trainer=TB._sha(os.path.abspath(__file__)), sel1_common=TB._sha(S.__file__),
                sel1_cse=TB._sha(sel1_cse.__file__), sel1_fastts=TB._sha(sel1_fastts.__file__), sel1_sweep=TB._sha(W.__file__),
                v4_world=TB._sha(X.__file__), m1_ruler=TB._sha(R.__file__), m2h_common=TB._sha(M.__file__),
                m2h_dense_common=TB._sha(D.__file__), m2h_b5_grammar=TB._sha(G.__file__), dbg_memo=TB._sha(TB.__file__),
                cr1_common=TB._sha(Q.__file__), sel1_dedup=TB._sha(DD.__file__), sel1_pool=TB._sha(P.__file__),
                sel1_gfn=TB._sha(GF.__file__), sel1_world2=TB._sha(W2.__file__))


# --------------------------------------------------------------------------- vocabulary / sampling
def extra_ids(vm):
    """Model-vocab ids after [body | END | HOLDs | BASINs]: holds, states, directions."""
    base = vm.token_count
    hold = [base + i for i in range(len(M.HOLDS))]
    st0 = base + len(M.HOLDS) + len(M.BASIN_NAMES)
    state = [st0 + i for i in range(len(STATE_NAMES))]
    dr = [st0 + len(STATE_NAMES) + i for i in range(len(DIR_NAMES))]
    return hold, state, dr


def append_choice(model, vm, b, basin_ids, allowed, device, generator=None):
    """One token after the current sequence, restricted to `allowed`, autograd live (TB._append_hold's step on
    the full model vocabulary)."""
    tokens, el, lengths = b.context["tokens"], b.context["effective_logits"], b.context["lengths"]
    B = tokens.shape[0]
    L = int(lengths.max())
    inp = torch.full((B, L), vm.end_token, dtype=torch.long, device=device)
    pos = torch.arange(L, device=device)[None, :] < lengths[:, None]
    inp = torch.where(pos, tokens[:, :L], inp)
    inp = torch.cat([basin_ids[:, None], inp], dim=1)
    logits, _, _ = model(inp)
    step = logits[torch.arange(B, device=device), lengths] if logits.dim() == 3 else logits
    V = step.shape[-1]
    mask = torch.full((V,), float("-inf"), device=device, dtype=step.dtype)
    mask[torch.as_tensor(allowed, device=device)] = 0.0
    eff = step + mask[None, :]
    action = torch.multinomial(torch.softmax(eff, dim=-1), 1, generator=generator).squeeze(1)
    tokens2 = torch.cat([tokens, torch.full((B, 1), vm.end_token, dtype=torch.long, device=device)], dim=1)
    tokens2[torch.arange(B, device=device), lengths] = action
    if el.shape[-1] < V:
        el = torch.cat([el, torch.full((B, el.shape[1], V - el.shape[-1]), float("-inf"), device=device,
                                       dtype=el.dtype)], dim=2)
    force = torch.full((V,), float("-inf"), device=device, dtype=el.dtype)
    force[vm.end_token] = 0.0
    el2 = torch.cat([el, force[None, None, :].expand(B, 1, V)], dim=1).clone()
    el2[torch.arange(B, device=device), lengths] = eff
    lengths2 = lengths + 1
    formulas = [tuple(int(t) for t in row[:n]) for row, n in zip(tokens2.tolist(), lengths2.tolist())]
    return TB._Batch(formulas=formulas, context=dict(tokens=tokens2, effective_logits=el2, lengths=lengths2,
                                                     max_len=tokens2.shape[1]))


def sample(model, vm, tables, counts, device, sweep=False):
    subs = [TB.sample_batch_g(TB.BasinView(model, vm.token_count, TB.basin_token(vm, b)), vm, tab, int(n), device=device)
            for b, tab, n in zip(M.BASIN_NAMES, tables, counts)]
    tokens, el, lengths = TB._merge_subs(subs, vm, device)
    if sweep:                                                   # --combo-sweep: the formula only, no combo tokens
        formulas = [tuple(int(t) for t in row[:n]) for row, n in zip(tokens.tolist(), lengths.tolist())]
        return TB._Batch(formulas=formulas, context=dict(tokens=tokens, effective_logits=el, lengths=lengths,
                                                         max_len=tokens.shape[1]))
    basin_ids = torch.cat([torch.full((int(n),), TB.basin_token(vm, b), dtype=torch.long, device=device)
                           for b, n in zip(M.BASIN_NAMES, counts)])
    b = TB._Batch(formulas=None, context=dict(tokens=tokens, effective_logits=el, lengths=lengths))
    for allowed in extra_ids(vm):
        b = append_choice(model, vm, b, basin_ids, allowed, device)
    return b


def decode(vm, toks):
    hold, state, dr = extra_ids(vm)
    toks = list(toks)
    e = toks.index(vm.end_token)
    body = tuple(toks[:e + 1])
    h, s, d = toks[e + 1], toks[e + 2], toks[e + 3]
    return body, M.HOLDS[hold.index(h)], state.index(s), dr.index(d)


# --------------------------------------------------------------------------- scoring
class Scorer:
    """VM + book for one update's candidates; in-process or over pools; per-update or global memo."""

    def __init__(self, ctx, vm, out_dir, workers, gpu_workers, memo, cse_gb, fastts=False, gpu_select=False):
        self.ctx, self.vm, self.memo = ctx, vm, memo
        self.sel_fn = S.vm_select_gpu if gpu_select else S.vm_select
        ctx["cse"] = sel1_cse.CachedExec(vm, cse_gb) if cse_gb and not gpu_workers else None
        self.sel_cache, self.rew_cache = {}, {}
        self.layout, self.zh = None, {}                         # --pool: sel1_pool.Layout, body -> (rank z, hash)
        self.vpool = self.spool = None
        if workers or gpu_workers:
            cdir = S.write_cache(ctx, os.path.join(out_dir, "_cache"))
            spawn = mp.get_context("spawn")
            if gpu_workers:
                self.vpool = spawn.Pool(gpu_workers, initializer=S.init_vm_worker, initargs=(cdir, cse_gb, fastts))
            if workers:
                env = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}
                os.environ["OMP_NUM_THREADS"] = "1"
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"       # inherited by the CPU workers only
                self.spool = spawn.Pool(workers, initializer=S.init_score_worker, initargs=(cdir,))
                for k, v in env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    def close(self):
        for p in (self.vpool, self.spool):
            if p is not None:
                p.close()
                p.join()

    def _select_local(self, body):
        try:
            if self.layout is not None:                         # --pool: vm_select_gpu's lines, signal kept for rank z
                s = P.masked_signal(self.ctx, self.vm, body)
                sels = P.select_signal(self.ctx, s)
                self.zh[body] = (self.layout.rankz(s).float(), DD.behaviour_hash(sels))
                del s
                return sels
            return self.sel_fn(self.ctx, self.vm, body)
        except (S.Invalid, CandidateLocalInvalid, M.ZeroCoverage) as e:
            return "INVALID:" + str(e)

    def zhash(self, body):
        """--pool: (rank z on the A layout, behaviour hash) of a body; recomputed when the update did not run its VM."""
        if body not in self.zh:
            r = self._select_local(body)
            if isinstance(r, str):
                raise RuntimeError("pool body turned invalid: %s" % r)
        return self.zh[body]

    def sweep_sels(self, sel_list, kind, dd_max):
        """sels per item -> the rescored score of each item's t_dd combo (nan when no finite combo); CPU pool if any."""
        tasks = [(k, s, kind, dd_max) for k, s in enumerate(sel_list)]
        if self.spool is not None:
            res = dict(self.spool.imap_unordered(W.sweep_task, tasks))
        else:
            res = {k: (lambda r: None if r is None else r[:4])(W.sweep(s, self.ctx, kind, dd_max)) for k, s, _, _ in tasks}
        return [float("nan") if res[k] is None else float(res[k][2]) for k in range(len(sel_list))]

    def run(self, cands, tm):
        """cands: list of (body, h, st, dr) with grammar-valid bodies -> {cand: rows | 'INVALID:...'}."""
        if self.memo == "update":
            self.sel_cache, self.rew_cache = {}, {}
        t_ = time.time()
        need_b = [b for b in dict.fromkeys(c[0] for c in cands) if b not in self.sel_cache]
        if self.vpool is not None:
            for b, r in self.vpool.imap(S.vm_task, need_b):
                self.sel_cache[b] = r
        else:
            for b in need_b:
                self.sel_cache[b] = self._select_local(b)
        tm["vm"] = time.time() - t_
        t_ = time.time()
        need_c = [c for c in dict.fromkeys(cands) if c not in self.rew_cache]
        todo = []
        for c in need_c:
            sel = self.sel_cache[c[0]]
            if isinstance(sel, str):
                self.rew_cache[c] = sel
            else:
                todo.append(c)
        if self.spool is not None:
            for c, rows in self.spool.imap(S.score_task, [(c, self.sel_cache[c[0]]) + c[1:] for c in todo], chunksize=2):
                self.rew_cache[c] = rows
        else:
            for c in todo:
                self.rew_cache[c] = S.rows_lite(S.score_sel(self.sel_cache[c[0]], self.ctx, *c[1:]))
        tm["book"] = time.time() - t_
        out = {c: self.rew_cache[c] for c in cands}
        if self.memo == "global":                                   # bounded: ~0.45 MB per body, ~10 KB per candidate
            for cache, cap, drop in ((self.sel_cache, 4000, 1000), (self.rew_cache, 200000, 50000)):
                if len(cache) > cap:
                    for k in list(cache)[:drop]:
                        del cache[k]
        return out, len(need_b), len(todo)

    def run_sweep(self, bodies, tm, kind, dd_max, seen_beh=None):
        """--combo-sweep: bodies -> {body: (combo, rows, score, sweep score) | 'INVALID:...'} (sel1_sweep: all 60
        combos, the A-half best by `kind`, rows / score of that combo through S.score_sel). seen_beh (--dedup): dict
        behaviour hash -> first body; a body whose selections hash to a seen behaviour is INVALID:DUP_BEHAV, not swept."""
        if self.memo == "update":
            self.sel_cache, self.rew_cache = {}, {}
        if self.layout is not None:
            self.zh = {}                                        # this update's rank z only (~9 MB each on the GPU)
        t_ = time.time()
        need_b = [b for b in dict.fromkeys(bodies) if b not in self.sel_cache and b not in self.rew_cache]
        if self.vpool is not None:
            for b, r in self.vpool.imap(S.vm_task, need_b):
                self.sel_cache[b] = r
        else:
            for b in need_b:
                self.sel_cache[b] = self._select_local(b)
        tm["vm"] = time.time() - t_
        t_ = time.time()
        todo = []
        for b in need_b:
            sel = self.sel_cache[b]
            if isinstance(sel, str):
                self.rew_cache[b] = sel
                continue
            if seen_beh is not None:
                h = DD.behaviour_hash(sel)
                if h in seen_beh:
                    self.rew_cache[b] = "INVALID:DUP_BEHAV"
                    self.sel_cache.pop(b, None)
                    continue
                seen_beh[h] = b
            todo.append(b)
        if self.spool is not None:
            it = self.spool.imap_unordered(W.sweep_task, [(b, self.sel_cache[b], kind, dd_max) for b in todo])
        else:
            it = ((b, (lambda r: None if r is None else r[:4])(W.sweep(self.sel_cache[b], self.ctx, kind, dd_max)))
                  for b in todo)
        for b, r in it:
            self.rew_cache[b] = "INVALID:NONFINITE" if r is None else r
        for b in todo:                                          # selections are only needed until the sweep is done
            self.sel_cache.pop(b, None)
        tm["book"] = time.time() - t_
        out = {b: self.rew_cache[b] for b in bodies}
        if self.memo == "global" and len(self.rew_cache) > 200000:
            for k in list(self.rew_cache)[:50000]:
                del self.rew_cache[k]
        return out, len(need_b), len(todo)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_name")
    ap.add_argument("--mode", required=True, choices=S.MODES)
    ap.add_argument("--updates", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--gpu-workers", type=int, default=0)
    ap.add_argument("--memo", default="update", choices=("update", "global"))
    ap.add_argument("--cse-gb", type=float, default=0.0)
    ap.add_argument("--speed-resume", action="store_true")
    ap.add_argument("--fastts", action="store_true")         # sel1_fastts fused TS operators (bit-identical)
    ap.add_argument("--gpu-select", action="store_true")    # mask / NULL roll / selection on the GPU (bit-identical)
    ap.add_argument("--reward", default="mean", choices=S.REWARDS)  # S.reward: mean = the original A-half bp/day
    ap.add_argument("--dd-max", type=float, default=0.0)             # t_dd: A-half max drawdown line (log NAV)
    ap.add_argument("--k", type=int, default=5)                       # coins per side (S.K)
    ap.add_argument("--alpha", type=float, default=None)              # plant strength (pc: default SEL1_ALPHA.json; pcnull: required)
    ap.add_argument("--combo-sweep", action="store_true")             # formula only; every body takes its best of 60 combos (sel1_sweep)
    ap.add_argument("--dedup", action="store_true")                   # behaviour dedup + zfam std fix (sel1_dedup)
    ap.add_argument("--pool", type=int, default=0)                    # AlphaGen pool reward, pool size (0 = off)
    ap.add_argument("--gfn", type=float, default=None)                # trajectory balance, beta (off = None)
    ap.add_argument("--frozen", action="store_true")                  # no update: the policy stays at u0
    ap.add_argument("--plant2", default=None)                         # second plant spec json (SEL1_ARMS world)
    ap.add_argument("--fork", action="store_true")                    # resume with the switches above changed
    ap.add_argument("--receipt-on-stop", action="store_true")         # --max-hours stop writes HOF + receipt too
    ap.add_argument("--ent-coef", type=float, default=0.03)
    ap.add_argument("--ent-floor-frac", type=float, default=0.35)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint-every", type=int, default=16)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-hours", type=float, default=0.0)
    args = ap.parse_args()
    out_dir = os.path.join(S.OUT_ROOT, args.out_name)
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
    if args.fastts:
        sel1_fastts.install()
    if args.alpha is not None and args.mode not in ("pc", "pcnull"):
        raise SystemExit("--alpha is for --mode pc / pcnull")
    if args.reward == "t_dd" and not args.dd_max > 0:
        raise SystemExit("--reward t_dd needs --dd-max > 0")
    if args.combo_sweep and args.reward == "mean":
        raise SystemExit("--combo-sweep picks the combo by t / t_dd")
    if args.dedup and not args.combo_sweep:
        raise SystemExit("--dedup needs --combo-sweep")
    if args.pool and not (args.combo_sweep and args.gpu_select and args.gpu_workers == 0):
        raise SystemExit("--pool needs --combo-sweep --gpu-select --gpu-workers 0")
    if bool(args.pool) + (args.gfn is not None) + bool(args.frozen) > 1:
        raise SystemExit("--pool / --gfn / --frozen: one at a time")
    S.K = args.k                                                    # before write_cache: workers read it from there
    lg = lambda m: print(m, flush=True)
    ctx = (W2.build(args.mode, device, vm, args.alpha, args.plant2, log=lg) if args.plant2 else
           S.build(args.mode, device, vm, alpha=args.alpha, log=lg))
    torch.manual_seed(args.seed)
    op_names = (list(OPERATOR_NAMES) + [f"W{w}" for w in M.WINDOWS] + ["END"] + M.HOLD_NAMES
                + list(M.BASIN_TOKEN_NAMES) + list(STATE_NAMES) + list(DIR_NAMES))
    model, fvocab = make_basin_model(alphagpt_mod=alphagpt_mod, ModelConfig=ModelConfig, FormulaVocab=FormulaVocab,
                                     feature_names=list(M.TERMINALS), operator_token_names=op_names,
                                     max_len=M.MAX_PROGRAM_LEN + 3, device=device)
    apply_output_bias(model, len(M.TERMINALS), len(OPERATOR_NAMES), len(M.WINDOWS), OP_BIAS, END_BIAS)
    hold_ids, state_ids, dir_ids = extra_ids(vm)
    if fvocab.size != dir_ids[-1] + 1:
        raise SystemExit("vocab size %d != %d" % (fvocab.size, dir_ids[-1] + 1))
    counts = M.quota(args.batch)
    tables = [build_legality_table(vm, max_len=M.MAX_PROGRAM_LEN, available_terminals=M.basin_terms(b),
                                   available_windows=M.WINDOWS).to(device) for b in M.BASIN_NAMES]
    groups = list(zip(M.BASIN_NAMES, counts))
    if not G.selftest(vm)["ok"]:
        raise SystemExit("grammar selftest failed")
    scorer = Scorer(ctx, vm, out_dir, args.workers, args.gpu_workers, args.memo, args.cse_gb, args.fastts, args.gpu_select)
    pool = None
    if args.pool:
        scorer.layout = P.Layout(ctx)

        def pool_tdd(z_sets):
            sels = [scorer.layout.select(scorer.layout.composite(zs)) for zs in z_sets]
            return scorer.sweep_sels(sels, args.reward, args.dd_max)
        pool = P.Pool(args.pool, scorer.layout, pool_tdd, POOL_RHO)
    logz = z_opt = None
    if args.gfn is not None:
        logz = GF.LogZ().to(device)
        z_opt = torch.optim.Adam(logz.parameters(), lr=GF.LR_LOGZ)
    mods = modules()
    manifest = dict(schema="SEL1_TRAIN_MANIFEST_V1", run=args.out_name, order="XS_SEL1", contract="XS_BASIN_TIME_CONTRACT_V4_20261013",
                    args=vars(args), modules=mods, python=sys.executable, conventions=S.__doc__, trainer=__doc__,
                    world=dict(mode=args.mode, grid=X.GRID, alpha=ctx["alpha"], months=len(ctx["months"]),
                               null_seed=S.NULL_SEED if args.mode in ("null", "pcnull") else None, k=S.K,
                               **({"plant2": ctx["plant2"]} if args.plant2 else {})),
                    batch=args.batch, quota=counts, vocab_size=fvocab.size,
                    model_params=sum(p.numel() for p in model.parameters()),
                    build_seconds=round(time.time() - t_build, 1), dev_read=False, oos_read=False,
                    written_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    ck = None
    if args.resume:
        ck_path = TB._latest_checkpoint(out_dir) if args.resume == "auto" else args.resume
        if not ck_path or not os.path.exists(ck_path):
            raise SystemExit("--resume %s: no checkpoint in %s" % (args.resume, out_dir))
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        free = FREE_ARGS | ({"out_name"} if args.speed_resume else set()) | (FORK_ARGS if args.fork else set())
        old = dict(LEGACY_ARGS, **ck["args"])                    # checkpoints from before --reward ran mean
        diff = {k: (old.get(k), v) for k, v in vars(args).items() if k not in free and old.get(k) != v}
        if diff:
            raise SystemExit("--resume refused, args differ: %s" % diff)
        dm = {k for k in set(ck["modules"]) | set(mods) if ck["modules"].get(k) != mods.get(k)}
        if dm and not (args.speed_resume and dm <= SPEED_MODULES):
            raise SystemExit("--resume refused, code differs: %s" % sorted(dm))
        manifest["resumed_from"] = dict(checkpoint=ck_path, update=int(ck["update"]), speed_code_differs=sorted(dm),
                                        checkpoint_modules=ck["modules"])
    json.dump(manifest, open(os.path.join(out_dir, "TRAIN_MANIFEST.json" if ck is None else
                                          "RESUME_MANIFEST_%s.json" % time.strftime("%Y%m%dT%H%M%S")), "w"),
              indent=1, default=str)

    def sc(v, nd):                                             # the --reward score, logged only when it is not bp/day
        return {} if args.reward == "mean" else dict(score=round(float(v), nd))

    def hof_key(kv):
        return -kv[1].get("score", kv[1]["reward_bpd"])

    def sname(st):                                             # combo-sweep: an invalid body has no combo
        return None if st is None else S.STATES[st]

    def dname(dr):
        return None if dr is None else S.DIRS[dr]

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    hof, n_eval, inval = {}, 0, collections.Counter()
    start_update, elapsed_prev, fmode = 0, 0.0, "w"
    ent = dict(coef=args.ent_coef, floor=None)
    seen_body, seen_beh = set(), {}                            # --dedup: bodies sent to the VM, behaviour hashes
    if ck is not None and args.dedup:
        seen_body = {tuple(b) for b in ck.get("seen_body", [])}
        seen_beh = {h: tuple(b) for h, b in ck.get("seen_beh", [])}
    if ck is not None:
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        hof = {tuple(tuple(x) if isinstance(x, list) else x for x in k): v for k, v in ck["hof"]}
        n_eval, inval, ent = int(ck["n_eval"]), collections.Counter(ck["inval"]), dict(ck["ent"])
        torch.set_rng_state(ck["torch_rng_state"])
        if device.type == "cuda" and ck.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng_state"], device)
        if pool is not None and ck.get("pool"):
            st = ck["pool"]
            for m in st["members"]:
                z, h = scorer.zhash(tuple(m["body"]))
                pool.members.append(dict(m, body=tuple(m["body"]), z=z))
            pool.score = float(st["score"])
        if logz is not None and ck.get("logz") is not None:
            logz.load_state_dict(ck["logz"])
            z_opt.load_state_dict(ck["z_opt"])
        start_update, elapsed_prev, fmode = int(ck["update"]) + 1, float(ck["elapsed_s"]), "a"
        for fn in ("BATCH_LOG.jsonl", "FORMULAS.jsonl"):
            TB._truncate_jsonl(os.path.join(out_dir, fn), int(ck["update"]))
        print("resumed from update %d" % start_update, flush=True)
    log_fh = open(os.path.join(out_dir, "BATCH_LOG.jsonl"), fmode, encoding="utf-8")
    form_fh = open(os.path.join(out_dir, "FORMULAS.jsonl"), fmode, encoding="utf-8")
    basin_of = sum(([b] * c for b, c in groups), [])
    t0, t_sess, stopped, vram = time.time() - elapsed_prev, time.time(), False, None
    win = collections.deque(maxlen=REPORT_EVERY)

    for update in range(start_update, args.updates):
        tm, t_ = {}, time.time()
        t_u = t_
        batch = sample(model, vm, tables, counts, device, sweep=args.combo_sweep)
        tm["sample"], t_ = time.time() - t_, time.time()
        n = args.batch
        cands, why = [None] * n, [None] * n
        for i, toks in enumerate(batch.formulas):
            body, h, st, dr = (tuple(toks), None, None, None) if args.combo_sweep else decode(vm, toks)
            cands[i] = (body, h, st, dr)
            g = G.check(vm, list(body))
            if g is not None:
                why[i] = "GRAMMAR_" + str(g)
        sweep_mis, n_dup = 0, collections.Counter()
        if args.dedup:                                          # a body seen before (or earlier in this batch): no VM call
            new_b = set()
            for i in range(n):
                if why[i] is None:
                    b = cands[i][0]
                    if b in seen_body or b in new_b:
                        why[i] = "DUP_BODY"
                    else:
                        new_b.add(b)
            seen_body |= new_b
        if args.combo_sweep:
            rb, n_vm, n_book = scorer.run_sweep([c[0] for c, w in zip(cands, why) if w is None], tm, args.reward,
                                                args.dd_max, seen_beh if args.dedup else None)
            res = {}
            for i in range(n):
                if why[i] is not None:
                    continue
                r = rb[cands[i][0]]
                if isinstance(r, str):
                    res[cands[i]] = r
                    continue
                combo, rows_c, s_rows, s_sweep = r
                cands[i] = (cands[i][0],) + tuple(combo)
                res[cands[i]] = rows_c
                sweep_mis += int(s_rows != s_sweep)
        else:
            res, n_vm, n_book = scorer.run([c for c, w in zip(cands, why) if w is None], tm)
        t_ = time.time()
        rew, bp, net, turn = (np.full(n, np.nan) for _ in range(4))   # rew = the --reward score, bp = cost0 bp/day
        valid = np.zeros(n, dtype=bool)
        for i in range(n):
            if why[i] is None:
                r = res[cands[i]]
                if isinstance(r, str):
                    why[i] = r[len("INVALID:"):]
                else:
                    rew[i], bp[i], net[i] = S.reward(r, args.reward, args.dd_max), S.bpd(r), S.bpd(r, "net")
                    turn[i] = sum(x["turn"] for x in r) / sum(x["days"] for x in r)
                    if not (np.isfinite(rew[i]) and np.isfinite(net[i])):
                        why[i] = "NONFINITE"
            if why[i] is not None:
                inval[why[i].split("_")[0]] += 1
                if why[i].startswith("DUP_"):
                    n_dup[why[i]] += 1
                continue
            valid[i] = True
        prow, gain, pev, pobj = None, None, {}, {}
        if pool is not None:                                    # evening item 5: marginal gain against the pool
            t_p = time.time()
            for i in np.flatnonzero(valid):
                b = cands[i][0]
                if b not in pobj:
                    z, hsh = scorer.zhash(b)
                    pobj[b] = dict(body=b, z=z, hash=hsh, cost0=[float(x["cost0"]) for x in res[cands[i]]],
                                   expr=vm.describe(list(b), list(M.TERMINALS)), single=float(rew[i]),
                                   first_s=round(time.time() - t0, 1))
            ub = list(pobj)
            pev = dict(zip(ub, pool.evaluate([pobj[b] for b in ub])))
            gain = np.full(n, np.nan)
            for i in np.flatnonzero(valid):
                e = pev[cands[i][0]]
                if isinstance(e, str):
                    valid[i], why[i] = False, e
                    inval[e.split("_")[0]] += 1
                else:
                    gain[i] = e["gain"]
            tm["pool"] = time.time() - t_p
        nv = int(valid.sum())
        if args.frozen:
            skip = "frozen"
        elif nv < 20:
            skip = "n_valid=%d<20" % nv
        elif pool is not None or args.gfn is not None:
            skip = None
        else:
            skip = "reward_std==0" if rew[valid].std() == 0.0 else None
        upd, fam_ent = {}, [None] * len(groups)
        coef_used = ent["coef"]
        if skip is None and args.gfn is not None:
            upd, fam_ent = GF.tb_update(model, optimizer, logz, z_opt, batch, GF.log_rewards(rew, valid, args.gfn), groups)
        elif skip is None:
            if pool is not None:                                # no batch mean (correction): gain / std, +-3, invalid -1
                g = gain[valid]
                sd = float(g.std())
                adv = np.full(n, -1.0)
                adv[valid] = 0.0 if sd < DD.STD_EPS else np.clip(g / sd, -D.ZCLIP, D.ZCLIP)
            else:
                adv = (DD.zfam if args.dedup else D.zfam)(rew, valid, counts)
                adv[~valid] = -1.0
            upd, fam_ent = TB.dense_update(model, optimizer, batch, np.arange(n), adv, groups, device,
                                           ent_coef=ent["coef"])
            me = upd.get("mean_entropy")
            if me is not None and np.isfinite(me):
                if ent["floor"] is None:
                    ent["floor"] = args.ent_floor_frac * float(me)
                if me < ent["floor"]:
                    ent["coef"] = min(ent["coef"] * ENT_UP, ENT_CAP)
                else:
                    ent["coef"] = max(args.ent_coef, ent["coef"] / ENT_DOWN)
        if pool is not None:
            t_p = time.time()
            done = None
            if nv:
                ib = int(np.flatnonzero(valid)[np.nanargmax(gain[valid])])
                if gain[ib] > 0:
                    b = cands[ib][0]
                    done = pool.execute(pobj[b], pev[b])
            acts = collections.Counter(e["act"] for e in pev.values() if not isinstance(e, str))
            prow = dict(update=update, size=len(pool.members), score=round(pool.score, 6), done=done,
                        best_gain=None if not nv else round(float(np.nanmax(gain[valid])), 6),
                        n_gain_pos=int((gain[valid] > 0).sum()), cand_act=dict(acts),
                        n_same=sum(1 for e in pev.values() if e == "POOL_SAME"))
            with open(os.path.join(out_dir, "POOL_LOG.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(prow) + "\n")
            tm["pool"] += time.time() - t_p
        tm["update"], t_ = time.time() - t_, time.time()
        for i in np.flatnonzero(valid if pool is None else np.isfinite(rew) & np.array([w is None or w.startswith("POOL_")
                                                                                           for w in why])):
            c = cands[i]
            if c not in hof:
                hof[c] = dict(basin=basin_of[i], expr=vm.describe(list(c[0]), list(M.TERMINALS)), h=c[1],
                              state=S.STATES[c[2]], dir=S.DIRS[c[3]], reward_bpd=round(float(bp[i]), 4),
                              net_bpd=round(float(net[i]), 4), turn_day=round(float(turn[i]), 4), n=0, first=update,
                              first_s=round(time.time() - t0, 1), **sc(rew[i], 4))
            hof[c]["n"] += 1
        if len(hof) > 8000:
            hof = dict(sorted(hof.items(), key=hof_key)[:5000])
        for i in range(n):
            c = cands[i]
            ok = valid[i] or (pool is not None and why[i] is not None and why[i].startswith("POOL_"))
            form_fh.write(json.dumps(dict(update=update, basin=basin_of[i], expr=vm.describe(list(c[0]), list(M.TERMINALS)),
                                          h=c[1], state=sname(c[2]), dir=dname(c[3]),
                                          reward_bpd=round(float(bp[i]), 6) if ok else None, **(sc(rew[i], 6) if ok else {}),
                                          net_bpd=round(float(net[i]), 6) if ok else None,
                                          turn_day=round(float(turn[i]), 6) if ok else None,
                                          **(dict(gain=None if not valid[i] else round(float(gain[i]), 6))
                                             if pool is not None else {}),
                                          invalid=why[i])) + "\n")
        n_eval += n
        win.append(cands)
        fam, off = {}, 0
        for k, (b, c) in enumerate(groups):
            m = valid[off:off + c]
            sl = slice(off, off + c)
            top = int(off + np.nanargmax(np.where(m, rew[sl], -np.inf))) if m.any() else None
            fam[b] = dict(n_valid=int(m.sum()), reward_bpd=round(float(np.mean(bp[sl][m])), 4) if m.any() else None,
                          **(sc(np.mean(rew[sl][m]), 4) if m.any() else {}),
                          entropy=round(fam_ent[k], 4) if fam_ent[k] is not None else None,
                          top=None if top is None else dict(expr=vm.describe(list(cands[top][0]), list(M.TERMINALS)),
                                                            h=cands[top][1], state=S.STATES[cands[top][2]],
                                                            dir=S.DIRS[cands[top][3]], reward_bpd=round(float(bp[top]), 4),
                                                            **sc(rew[top], 4)))
            off += c
        tm["log"] = time.time() - t_
        sec = time.time() - t_u
        if device.type == "cuda":
            vram = torch.cuda.max_memory_allocated() / 2 ** 30
        row = dict(update=update, seconds=round(sec, 2), times={k: round(v, 2) for k, v in tm.items()},
                   n_valid=nv, n_unique=len({c for c, w in zip(cands, why) if w is None}),
                   n_unique_body=len({c[0] for c in cands}), n_vm=n_vm, n_book=n_book, skipped=skip,
                   reward_mean_bpd=round(float(np.mean(bp[valid])), 4) if nv else None,
                   reward_pos_frac=round(float((bp[valid] > 0).mean()), 4) if nv else None,
                   **(sc(np.mean(rew[valid]), 4) if nv else {}),
                   ent_coef_used=round(coef_used, 6), ent_floor=ent["floor"], hof_size=len(hof), families=fam,
                   **(dict(sweep_rescore_mismatch=sweep_mis) if args.combo_sweep else {}),
                   **(dict(dup=dict(n_dup), n_behav=len(seen_beh)) if args.dedup else {}),
                   vram_gib=round(vram, 3) if vram else None,
                   **{k: upd.get(k) for k in ("pg_loss", "entropy_loss", "gradient_norm", "mean_entropy",
                                              "skipped_nonfinite_update") if k in upd})
        log_fh.write(json.dumps(row) + "\n")
        log_fh.flush()
        form_fh.flush()
        json.dump(dict(pid=os.getpid(), update=update, updates_total=args.updates, seconds=round(sec, 2),
                       updated=time.strftime("%Y-%m-%dT%H:%M:%S%z")), open(os.path.join(out_dir, "HEARTBEAT.json"), "w"))
        if update < start_update + 4 or (update + 1) % 16 == 0 or update == args.updates - 1:
            eta = (time.time() - t_sess) / (update + 1 - start_update) * (args.updates - update - 1)
            print("upd %4d %5.1fs %s valid %d uniq %d vm %d book %d | %s | ent %s coef %.3f eta %s"
                  % (update, sec, row["times"], nv, row["n_unique"], n_vm, n_book,
                     " ".join("%s r%s e%s" % (b, f["reward_bpd"], f["entropy"]) for b, f in fam.items()),
                     None if upd.get("mean_entropy") is None else round(upd["mean_entropy"], 3), coef_used,
                     TB._hms(eta)), flush=True)
        if (update + 1) % REPORT_EVERY == 0:
            allc = [c for cs in win for c in cs]
            rep = dict(update=update, window=len(win), distinct_share=round(len(set(allc)) / max(len(allc), 1), 4),
                       distinct_body_share=round(len({c[0] for c in allc}) / max(len(allc), 1), 4), families={})
            off = 0
            for b, c in groups:
                fc = [cs[i] for cs in win for i in range(off, off + c)]
                mc = collections.Counter(fc).most_common(1)[0]
                rep["families"][b] = dict(mode=dict(expr=vm.describe(list(mc[0][0]), list(M.TERMINALS)), h=mc[0][1],
                                                    state=sname(mc[0][2]), dir=dname(mc[0][3]),
                                                    share=round(mc[1] / len(fc), 4)),
                                          distinct_share=round(len(set(fc)) / len(fc), 4))
                off += c
            top100 = sorted(hof.items(), key=hof_key)[:100]            # the hall's chosen combos (supplement 10-15)
            rep["hof_top100_combo"] = {"%s/%s/%s" % k: v for k, v in collections.Counter(
                (c[1], S.STATES[c[2]], S.DIRS[c[3]]) for c, _ in top100).most_common(8)}
            with open(os.path.join(out_dir, "PROGRESS_64.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rep) + "\n")
            print("  [64] distinct %.3f body %.3f | %s | hof100 %s" % (rep["distinct_share"], rep["distinct_body_share"],
                  " | ".join("%s %.2f %s H%s %s %s" % (b, f["mode"]["share"], f["mode"]["expr"][:60], f["mode"]["h"],
                                                      f["mode"]["state"], f["mode"]["dir"]) for b, f in rep["families"].items()),
                  rep["hof_top100_combo"]), flush=True)
        over = bool(args.max_hours) and (time.time() - t0) >= args.max_hours * 3600.0
        if (update + 1) % args.checkpoint_every == 0 or over or update == args.updates - 1:
            TB._save_checkpoint(os.path.join(out_dir, "checkpoint_%04d.pt" % update), model=model.state_dict(),
                                optimizer=optimizer.state_dict(), update=update, args=vars(args), modules=mods,
                                hof=list(hof.items()), n_eval=n_eval, inval=dict(inval), ent=dict(ent),
                                **(dict(seen_body=[list(b) for b in seen_body],
                                        seen_beh=[(h, list(b)) for h, b in seen_beh.items()]) if args.dedup else {}),
                                **(dict(pool=pool.state()) if pool is not None else {}),
                                **(dict(logz=logz.state_dict(), z_opt=z_opt.state_dict()) if logz is not None else {}),
                                torch_rng_state=torch.get_rng_state(),
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
    scorer.close()
    if pool is not None:
        st = pool.state()
        st["members"] = [dict(m, body=list(m["body"])) for m in st["members"]]
        json.dump(dict(schema="SEL1_POOL_V1", run=args.out_name, mode=args.mode, rho_max=POOL_RHO, stopped=stopped,
                       last_update=update, **st), open(os.path.join(out_dir, "POOL.json"), "w", encoding="utf-8"), indent=1)
    if stopped and not args.receipt_on_stop:
        return
    top = sorted(hof.items(), key=hof_key)[:500]
    json.dump(dict(schema="SEL1_HOF_V1", run=args.out_name, mode=args.mode, rule="A-half cost0 bp/day, top 500" if args.reward == "mean" else
                   "A-half score %s (dd_max %s), top 500; reward_bpd = cost0 bp/day" % (args.reward, args.dd_max),
                   entries=[dict(body=list(c[0]), st=c[2], dr=c[3], **e) for c, e in top]),
              open(os.path.join(out_dir, "HOF.json"), "w", encoding="utf-8"), indent=1)
    json.dump(dict(schema="SEL1_TRAIN_RECEIPT_V1", run=args.out_name, mode=args.mode, args=vars(args), modules=mods,
                   n_eval=n_eval, invalid=dict(inval), hof_size=len(hof), peak_vram_gib=round(vram, 3) if vram else None,
                   stopped=stopped, last_update=update,
                   compute_seconds=round(time.time() - t0, 1), dev_read=False, oos_read=False,
                   created=time.strftime("%Y-%m-%dT%H:%M:%S%z")),
              open(os.path.join(out_dir, "TRAIN_RECEIPT_V1.json"), "w"), indent=1)
    print("done updates %d n_eval %d seconds %.0f" % (args.updates, n_eval, time.time() - t0), flush=True)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
