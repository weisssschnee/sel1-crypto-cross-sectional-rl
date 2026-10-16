"""XS_SEL1 tree GP v2, ARMS v2 GA-fix (supplement 10-16 noon). New file, a copy of sel1_ga.py (sha16 of that file in
the manifest; it is not modified, ARMS v1 ran it) with the fixes below; the trainer's Scorer / sampler / world are
imported from sel1_trainer_x as there.

  islands     two sub-islands per basin (T1a T1b T2a T2b T3a T3b; terminals stay basin-locked), POP 150 each.
  steady      each step of an island makes REPL = 20 % of POP new valid children, adds them and drops the worst REPL
              (rank key below); islands take steps in turn.
  parents     half uniform over the island's MAP-Elites cells (one cell per combo, the cell's best formula), half
              tournament 5 on the population.
  child       crossover p 0.5 (as v1), otherwise one mutation: point, subtree (rand_tree), hoist (a value subtree
              becomes the whole formula), shrink (a value subtree replaced by one of its own proper value subtrees,
              or by a terminal when it has none), window +-1 step (a window token to the neighbouring ladder value).
              Legality exactly as v1 (sampler mask + grammar + window minimum + length).
  parsimony   rank key (score', -length): a tie goes to the shorter formula; score' = score, or with --pars-mult m < 1
              and length > --pars-len L: score * m when score > 0, score / m when score < 0. ARMS v2: ties only (m 1).
  reset       every --reset-min (30) minutes, per basin: the sub-island with the lower best score' is reset to the
              top 10 % of the other one's population (FunSearch), its archive rebuilt from those.
  fitness / dedup / hall / outputs as v1 (Scorer.run_sweep, behaviour hash, HOF.json in the trainer's schema).
    python -I sel1_ga2.py OUT --mode pcnull --alpha A --reward t_dd --dd-max D [--plant2 F] [--max-hours H]
    python -I sel1_ga2.py OUT ... --selftest [--inject-plant] [--cpu-select]   # operators 100 % legal; plant t_A
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import pickle
import random
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import sel1_trainer_x as TX
import sel1_common as S
import sel1_dedup as DD
import sel1_world2 as W2
import sel1_sweep as W
import m2h_common as M
import m2h_dense_common as D
import m2h_b5_grammar as G
import m2h_rgp as RG
import m2h_b5_trainer_dbg_memo as TB
import cr1_common as Q
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import (OPERATOR_ARITIES, OPERATOR_NAMES, OPERATOR_MIN_WINDOW,
                                                         WINDOWED_OPERATORS)
from okx_alpha_discovery.adapters.xs_tscs_sampler_v2 import build_legality_table

POP, TOURN, P_CX, MAX_TRY, REFILL, ELITE_TOP, MINE_TOP = 150, 5, 0.5, 50, 4, 60, 50
REPL_FRAC, P_ARCH, RESET_TOP = 0.2, 0.5, 0.1
SUBS = ("a", "b")
ISLANDS = tuple(b + s for b in M.BASIN_NAMES for s in SUBS)
MUTS = (("point", 0.25), ("subtree", 0.25), ("hoist", 0.15), ("shrink", 0.15), ("win", 0.2))
PARS = dict(len=32, mult=1.0)                                   # set from --pars-len / --pars-mult


def fam_of(isl):
    return isl[:-1]


def eff(score, n):
    """Parsimony-adjusted score' (see header)."""
    m = PARS["mult"]
    if m < 1.0 and n > PARS["len"]:
        return score * m if score > 0 else score / m
    return score


def rank_key(x):
    """Sort key, best first: (-score', length)."""
    return (-eff(x["score"], len(x["body"])), len(x["body"]))
ARITY = dict(OPERATOR_ARITIES)
WINDOWED = frozenset(WINDOWED_OPERATORS)


# --------------------------------------------------------------------------- postfix trees (token ints, END dropped)
def starts(body):
    """start index of the subtree ending at each position (a window is a leaf; it sits right before its op)."""
    vm = _VM[0]
    st, out = [], []
    for i, t in enumerate(body):
        if vm.is_terminal(t) or vm.is_window(t):
            st.append(i)
        else:
            k = ARITY[vm.operator_name(t)] + (1 if vm.operator_name(t) in WINDOWED else 0)
            if len(st) < k:
                raise ValueError("malformed")
            s = st[-k]
            del st[-k:]
            st.append(s)
        out.append(st[-1])
    return out


def value_nodes(body):
    vm = _VM[0]
    return [i for i, t in enumerate(body) if not vm.is_window(t)]


def subtree(body, i, s=None):
    s = starts(body) if s is None else s
    return list(body[s[i]:i + 1])


def replace(body, i, sub, s=None):
    s = starts(body) if s is None else s
    return list(body[:s[i]]) + list(sub) + list(body[i + 1:])


_VM = [None]
_TAB = {}                                                       # island -> (legality mask numpy bool, min_finish list)


def sampler_legal(body, fam):
    """True when the RL sampler (sample_batch_g: legality table of the island + grammar mask) could emit body + END:
    the GA searches exactly the policy's space."""
    vm = _VM[0]
    mask, mf = _TAB[fam]
    L = mask.shape[2] - 1
    seq = list(body) + [vm.end_token]
    if len(seq) > L:
        return False
    depth, window, rows = 0, 0, []
    for step, tk in enumerate(seq):
        rem = L - step
        if not mask[depth, window, rem, tk]:
            return False
        if tk == vm.end_token:
            return step == len(seq) - 1 and len(rows) == 1 and not rows[0][0] and not rows[0][3]
        nxt, bad = TB._g_step(rows, tk, vm)
        if bad or TB._g_need(nxt, mf) > rem - 1:
            return False
        rows = nxt
        if vm.is_terminal(tk):
            depth, window = depth + 1, 0
        elif vm.is_window(tk):
            window = 1
        else:
            depth, window = depth + 1 - ARITY[vm.operator_name(tk)], 0
    return False


def legal(body, fam):
    """body without END -> sampler-legal on the island, B5 grammar, and every windowed op at or above its minimum
    window (the VM refuses the rest as WINDOW_BELOW_OPERATOR_MINIMUM)."""
    vm = _VM[0]
    if not body or not sampler_legal(body, fam):
        return False
    for i, t in enumerate(body):
        if vm.is_operator(t) and vm.operator_name(t) in WINDOWED:
            if i == 0 or not vm.is_window(body[i - 1]) or vm.window_value(body[i - 1]) < OPERATOR_MIN_WINDOW[vm.operator_name(t)]:
                return False
    return G.check(vm, list(body) + [vm.end_token]) is None


class Ops:
    def __init__(self, vm, fam, rng, p_rgp=0.0):
        self.vm, self.fam, self.rng, self.p_rgp = vm, fam, rng, float(p_rgp)
        self.terms = list(M.basin_terms(fam))
        self.wins = [vm.window_base + k for k in range(len(M.WINDOWS))]
        self.vocab = RG.token_names(vm, fam)
        self.same_op = {}
        for n in OPERATOR_NAMES:
            self.same_op[n] = [vm.operator_base + k for k, m in enumerate(OPERATOR_NAMES)
                               if ARITY[m] == ARITY[n] and (m in WINDOWED) == (n in WINDOWED) and m != n]
        self.mined = []
        self.tries = collections.Counter()

    def rand_sub(self):
        names = RG.rand_tree(self.rng, self.vocab, self.fam, max_len=12, max_depth=3)
        return RG.names_to_tokens(self.vm, names)

    def point(self, body):
        i = self.rng.randrange(len(body))
        t, out = body[i], list(body)
        if self.vm.is_terminal(t):
            out[i] = self.rng.choice(self.terms)
        elif self.vm.is_window(t):
            out[i] = self.rng.choice(self.wins)
        else:
            alt = self.same_op[self.vm.operator_name(t)]
            if not alt:
                return None
            out[i] = self.rng.choice(alt)
        return out

    def mutate(self, body):
        s = starts(body)
        i = self.rng.choice(value_nodes(body))
        if self.mined and self.rng.random() < self.p_rgp:
            return replace(body, i, self.rng.choice(self.mined), s), "rgp"
        u, kind = self.rng.random(), MUTS[-1][0]
        for k, p in MUTS:
            if u < p:
                kind = k
                break
            u -= p
        if kind == "point":
            return self.point(body), kind
        if kind == "subtree":
            return replace(body, i, self.rand_sub(), s), kind
        if kind == "hoist":                                     # a proper value subtree becomes the formula
            inner = [j for j in value_nodes(body) if j != len(body) - 1]
            if not inner:
                return None, kind
            return subtree(body, self.rng.choice(inner), s), kind
        if kind == "shrink":                                    # subtree i -> one of its proper value subtrees / a terminal
            inner = [j for j in value_nodes(body) if s[i] <= j < i]
            sub = subtree(body, self.rng.choice(inner), s) if inner else [self.rng.choice(self.terms)]
            return replace(body, i, sub, s), kind
        wins = [j for j, t in enumerate(body) if self.vm.is_window(t)]   # window +-1 step on the ladder
        if not wins:
            return None, kind
        j = self.rng.choice(wins)
        k = body[j] - self.vm.window_base + self.rng.choice((-1, 1))
        if not 0 <= k < len(M.WINDOWS):
            return None, kind
        out = list(body)
        out[j] = self.vm.window_base + k
        return out, kind

    def crossover(self, a, b):
        i = self.rng.choice(value_nodes(a))
        j = self.rng.choice(value_nodes(b))
        return replace(a, i, subtree(b, j)), "cx"

    def child(self, pick):
        """pick() -> a parent body (no END). -> (legal body, kind) after at most MAX_TRY draws, or (None, kind)."""
        kind = None
        for _ in range(MAX_TRY):
            if self.rng.random() < P_CX:
                c, kind = self.crossover(pick(), pick())
            else:
                c, kind = self.mutate(pick())
            self.tries[kind] += 1
            if c is not None and legal(c, self.fam):
                self.tries[kind + "_ok"] += 1
                return c, kind
        return None, kind

    def mine(self, bodies):
        """m2h_rgp's miner on the island's best bodies: sub-expressions of 3..8 tokens, as legal token lists."""
        ex = RG.SubexpressionExtractor(RG.miner_arity())
        pre = [RG.postfix_to_prefix(RG.tokens_to_names(self.vm, b)) for b in bodies]
        out = []
        for sub in ex.extract(pre, 3, 8):
            try:
                toks = RG.names_to_tokens(self.vm, flat_prefix_to_postfix(list(sub)))
            except (ValueError, IndexError):
                continue
            if legal(toks, self.fam):
                out.append(toks)
        self.mined = sorted(out)
        return len(out)


def flat_prefix_to_postfix(seq):
    """The miner's sub-expressions are FLAT preorder name tuples (SubexpressionExtractor._preorder); m2h_rgp's
    prefix_to_postfix walks one level only, so a depth > 1 sub-expression would come out scrambled. Parsed here
    with the miner's arity (a windowed op's window is its last prefix child, as postfix_to_prefix writes it)."""
    ar = RG.miner_arity()

    def walk(k):
        tok = seq[k]
        out, k = [], k + 1
        for _ in range(ar.get(tok, 0)):
            sub, k = walk(k)
            out += sub
        return out + [tok], k
    out, k = walk(0)
    if k != len(seq):
        raise ValueError("leftover")
    return out


# --------------------------------------------------------------------------- world / init
def tournament(rng, pop):
    best = None
    for _ in range(TOURN):
        x = pop[rng.randrange(len(pop))]
        if best is None or rank_key(x) < rank_key(best):
            best = x
    return best["body"]


def parent(rng, pop, arch):
    """Half uniform over the archive's cells, half tournament on the population."""
    if arch and rng.random() < P_ARCH:
        return arch[sorted(arch)[rng.randrange(len(arch))]]["body"]
    return tournament(rng, pop)


def u0_bodies(model, vm, table, fam, n, device):
    with torch.no_grad():
        b = TB.sample_batch_g(TB.BasinView(model, vm.token_count, TB.basin_token(vm, fam)), vm, table, int(n), device=device)
    return [list(f[:-1]) if vm.is_end(f[-1]) else list(f) for f in b.formulas]


def build_model(vm, device, seed):
    """The trainer's model at u0 (same lines and seed order as sel1_trainer main)."""
    from train_basin_riskpg_v1 import make_basin_model, apply_output_bias
    import model_core.alphagpt as alphagpt_mod
    from model_core.config import ModelConfig
    from model_core.vocab import FormulaVocab
    ModelConfig.DEVICE = device
    torch.manual_seed(seed)
    op_names = (list(OPERATOR_NAMES) + [f"W{w}" for w in M.WINDOWS] + ["END"] + M.HOLD_NAMES
                + list(M.BASIN_TOKEN_NAMES) + list(TX.STATE_NAMES) + list(TX.DIR_NAMES))
    model, fvocab = make_basin_model(alphagpt_mod=alphagpt_mod, ModelConfig=ModelConfig, FormulaVocab=FormulaVocab,
                                     feature_names=list(M.TERMINALS), operator_token_names=op_names,
                                     max_len=M.MAX_PROGRAM_LEN + 3, device=device)
    apply_output_bias(model, len(M.TERMINALS), len(OPERATOR_NAMES), len(M.WINDOWS), TX.OP_BIAS, TX.END_BIAS)
    return model


def tables_for(vm, device):
    out = {}
    for b in M.BASIN_NAMES:
        t = build_legality_table(vm, max_len=M.MAX_PROGRAM_LEN, available_terminals=M.basin_terms(b),
                                 available_windows=M.WINDOWS)
        _TAB[b] = (torch.isfinite(t.mask).numpy(), t.min_finish.tolist())
        out[b] = t.to(device)
    return out


# --------------------------------------------------------------------------- run
class Run:
    """State of a GA run (pickled each generation): populations, per-island archives, hall, seen sets."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.pop = {b: [] for b in ISLANDS}                     # list of dict(body, score, combo)
        self.arch = {b: {} for b in ISLANDS}                    # combo -> dict(body, score, expr)
        self.hof, self.seen_body, self.seen_beh = {}, set(), {}
        self.gen, self.u, self.n_eval, self.elapsed = 0, 0, 0, 0.0
        self.inval = collections.Counter()
        self.tries = {b: collections.Counter() for b in M.BASIN_NAMES}
        self.last_reset, self.resets = 0.0, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_name")
    ap.add_argument("--mode", required=True, choices=S.MODES)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--reward", default="t_dd", choices=("t", "t_dd"))
    ap.add_argument("--dd-max", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--plant2", default=None)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--pop", type=int, default=POP)
    ap.add_argument("--rgp", type=float, default=0.0)            # RGP sub-expression injection probability (off)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-hours", type=float, default=0.0)
    ap.add_argument("--gens", type=int, default=10 ** 6)
    ap.add_argument("--budget", type=int, default=0)             # stop at this many behaviour-distinct scored formulas
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--inject-plant", action="store_true")       # self-check: the plant formula in T1's u0 population
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pars-len", type=int, default=32)          # parsimony: length above which --pars-mult applies
    ap.add_argument("--pars-mult", type=float, default=1.0)      # 1.0 = ties-to-shorter only
    ap.add_argument("--reset-min", type=float, default=30.0)     # island reset period, minutes
    ap.add_argument("--cpu-select", action="store_true")         # selection on the CPU path (bit-identical; CPU tests)
    args = ap.parse_args()
    PARS.update(len=args.pars_len, mult=args.pars_mult)
    if args.reward == "t_dd" and not args.dd_max > 0:
        raise SystemExit("--reward t_dd needs --dd-max > 0")
    out_dir = os.path.join(S.OUT_ROOT, args.out_name)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)
    torch.set_num_threads(4)
    t_build = time.time()
    vm = Q.make_vm()
    _VM[0] = vm
    TX.sel1_fastts.install()
    S.K = args.k
    lg = lambda m: print(m, flush=True)
    ctx = (W2.build(args.mode, device, vm, args.alpha, args.plant2, log=lg) if args.plant2 else
           S.build(args.mode, device, vm, alpha=args.alpha, log=lg))
    tabs = tables_for(vm, device)
    model = build_model(vm, device, args.seed)
    if not G.selftest(vm)["ok"]:
        raise SystemExit("grammar selftest failed")
    scorer = TX.Scorer(ctx, vm, out_dir, args.workers, 0, "update", 0.0, True, not args.cpu_select)
    ops = {b: Ops(vm, b, None, args.rgp) for b in M.BASIN_NAMES}
    if args.selftest:
        return selftest(args, vm, ctx, scorer, model, tabs, ops, out_dir, device)
    ck_path = os.path.join(out_dir, "checkpoint.pkl")
    if args.resume:
        st = pickle.load(open(ck_path, "rb"))
        old = {k: v for k, v in st["args"].items() if k not in ("max_hours", "gens", "budget", "resume", "workers")}
        new = {k: v for k, v in vars(args).items() if k not in ("max_hours", "gens", "budget", "resume", "workers")}
        if old != new:
            raise SystemExit("--resume refused, args differ: %s" % {k: (old.get(k), v) for k, v in new.items() if old.get(k) != v})
        run = st["run"]
        for fn in ("FORMULAS.jsonl", "BATCH_LOG.jsonl"):
            TB._truncate_jsonl(os.path.join(out_dir, fn), run.u - 1)
    else:
        run = Run(args.seed)
    for o in ops.values():
        o.rng = run.rng
    mods = dict(TX.modules(), sel1_ga2=TB._sha(os.path.abspath(__file__)), m2h_rgp=TB._sha(RG.__file__),
                sel1_ga_v1=TB._sha(os.path.join(HERE, "sel1_ga.py")))
    manifest = dict(schema="SEL1_TRAIN_MANIFEST_V1", run=args.out_name, order="XS_SEL1", method="GA2", args=vars(args),
                    modules=mods, python=sys.executable, trainer=__doc__,
                    world=dict(mode=args.mode, grid=TX.X.GRID, alpha=ctx["alpha"], months=len(ctx["months"]),
                               null_seed=S.NULL_SEED if args.mode in ("null", "pcnull") else None, k=S.K,
                               plant2=ctx.get("plant2")),
                    pop=args.pop, islands=list(ISLANDS), repl_frac=REPL_FRAC, p_archive_parent=P_ARCH, mutations=dict(MUTS),
                    reset_top=RESET_TOP, tournament=TOURN, p_cx=P_CX, build_seconds=round(time.time() - t_build, 1),
                    dev_read=False, oos_read=False, written_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(manifest, open(os.path.join(out_dir, "TRAIN_MANIFEST.json" if not args.resume else
                                          "RESUME_MANIFEST_%s.json" % time.strftime("%Y%m%dT%H%M%S")), "w"),
              indent=1, default=str)
    fmode = "a" if args.resume else "w"
    form_fh = open(os.path.join(out_dir, "FORMULAS.jsonl"), fmode, encoding="utf-8")
    log_fh = open(os.path.join(out_dir, "BATCH_LOG.jsonl"), fmode, encoding="utf-8")
    t0 = time.time() - run.elapsed

    def evaluate(isl, items):
        """isl: island; items: list of (body, kind) legal, unseen bodies -> list of valid dict(body, score, combo); logs one
        'update' (BATCH_LOG row) per call."""
        t_ = time.time()
        fam = fam_of(isl)
        bodies = [tuple(b) + (vm.end_token,) for b, _ in items]
        run.seen_body |= set(bodies)
        tm = {}
        rb, n_vm, n_book = scorer.run_sweep(bodies, tm, args.reward, args.dd_max, run.seen_beh)
        out, inv = [], collections.Counter()
        for (b, kind), bb in zip(items, bodies):
            r = rb[bb]
            rec = dict(update=run.u, gen=run.gen, basin=fam, island=isl, kind=kind, expr=vm.describe(list(bb), list(M.TERMINALS)))
            if isinstance(r, str):
                why = r[len("INVALID:"):]
                inv[why.split("_")[0]] += 1
                run.inval[why.split("_")[0]] += 1
                form_fh.write(json.dumps(dict(rec, h=None, state=None, dir=None, reward_bpd=None, invalid=why)) + "\n")
                continue
            combo, rows, s_rows, _ = r
            sc = float(S.reward(rows, args.reward, args.dd_max))
            bp, net = S.bpd(rows), S.bpd(rows, "net")
            turn = sum(x["turn"] for x in rows) / sum(x["days"] for x in rows)
            if not (np.isfinite(sc) and np.isfinite(net)):
                inv["NONFINITE"] += 1
                run.inval["NONFINITE"] += 1
                form_fh.write(json.dumps(dict(rec, h=None, state=None, dir=None, reward_bpd=None, invalid="NONFINITE")) + "\n")
                continue
            c = (bb,) + tuple(combo)
            if c not in run.hof:
                run.hof[c] = dict(basin=fam, expr=rec["expr"], h=combo[0], state=S.STATES[combo[1]], dir=S.DIRS[combo[2]],
                                  reward_bpd=round(bp, 4), net_bpd=round(net, 4), turn_day=round(turn, 4), n=0,
                                  first=run.u, first_s=round(time.time() - t0, 1), score=round(sc, 4), kind=kind)
            run.hof[c]["n"] += 1
            form_fh.write(json.dumps(dict(rec, h=combo[0], state=S.STATES[combo[1]], dir=S.DIRS[combo[2]],
                                          reward_bpd=round(bp, 6), score=round(sc, 6), net_bpd=round(net, 6),
                                          turn_day=round(turn, 6), invalid=None)) + "\n")
            x = dict(body=list(b), score=sc, combo=tuple(combo))
            out.append(x)
            cell = run.arch[isl].get(tuple(combo))
            if cell is None or rank_key(x) < rank_key(cell):
                run.arch[isl][tuple(combo)] = dict(body=list(b), score=sc, expr=rec["expr"])
        if len(run.hof) > 8000:
            run.hof = dict(sorted(run.hof.items(), key=lambda kv: -kv[1]["score"])[:5000])
        run.n_eval += len(items)
        sec = time.time() - t_
        sc_v = [x["score"] for x in out]
        log_fh.write(json.dumps(dict(update=run.u, gen=run.gen, basin=fam, island=isl, seconds=round(sec, 2),
                                     times={k: round(v, 2) for k, v in tm.items()}, n=len(items), n_valid=len(out),
                                     n_vm=n_vm, n_book=n_book, invalid=dict(inv), n_behav=len(run.seen_beh),
                                     score=round(float(np.mean(sc_v)), 4) if sc_v else None,
                                     score_max=round(float(np.max(sc_v)), 4) if sc_v else None,
                                     cells=len(run.arch[isl]), hof_size=len(run.hof),
                                     elapsed_s=round(time.time() - t0, 1))) + "\n")
        log_fh.flush()
        form_fh.flush()
        run.u += 1
        return out

    def fresh(isl, n, make):
        """Up to n legal bodies never sent before, from make() (None = no more)."""
        out, keys = [], set()
        for _ in range(n * MAX_TRY):
            if len(out) >= n:
                break
            r = make()
            if r is None or r[0] is None:
                continue
            b, kind = r
            k = tuple(b) + (vm.end_token,)
            if k in run.seen_body or k in keys or not legal(b, fam_of(isl)):
                continue
            keys.add(k)
            out.append((b, kind))
        return out

    def fill(isl, have, make, target):
        """have (valid list) + evaluated fresh bodies until target valid or REFILL rounds."""
        for _ in range(REFILL):
            need = target - len(have)
            if need <= 0:
                break
            items = fresh(isl, need, make)
            if not items:
                break
            have = have + evaluate(isl, items)
        return have

    def stop_now():
        if args.max_hours and time.time() - t0 >= args.max_hours * 3600.0:
            return "max_hours"
        if args.budget and len(run.seen_beh) >= args.budget:
            return "budget"
        return None

    def best_of(isl):
        return min((rank_key(x) for x in run.pop[isl]), default=(np.inf, 0))

    def reset():
        """Per basin: the sub-island with the worse best is refounded from the top RESET_TOP of the other."""
        for fam in M.BASIN_NAMES:
            a, b = [fam + x for x in SUBS]
            good, bad = (a, b) if best_of(a) <= best_of(b) else (b, a)
            keep = sorted(run.pop[good], key=rank_key)[:max(1, int(round(RESET_TOP * args.pop)))]
            run.pop[bad] = [dict(x) for x in keep]
            run.arch[bad] = {}
            for x in keep:
                cell = run.arch[bad].get(tuple(x["combo"]))
                if cell is None or rank_key(x) < rank_key(cell):
                    run.arch[bad][tuple(x["combo"])] = dict(body=list(x["body"]), score=x["score"],
                                                            expr=vm.describe(list(x["body"]) + [vm.end_token],
                                                                             list(M.TERMINALS)))
            run.resets.append(dict(u=run.u, elapsed_s=round(time.time() - t0, 1), basin=fam, reset=bad, source=good,
                                   best_source=-best_of(good)[0]))
        run.last_reset = time.time() - t0

    why_stop = None
    n_new = max(1, int(round(REPL_FRAC * args.pop)))
    if run.gen == 0 and not any(run.pop.values()):              # step 0: the u0 policy, POP per island
        for isl in ISLANDS:
            fam = fam_of(isl)
            bag = collections.deque()

            def u0(fam=fam, bag=bag):
                if not bag:
                    for f in u0_bodies(model, vm, tabs[fam], fam, 256, device):
                        bag.append((f, "u0"))
                return bag.popleft()
            first = [(list(D.plant_body(vm))[:-1], "plant")] if args.inject_plant and isl == "T1a" else []
            have = evaluate(isl, first) if first else []
            run.pop[isl] = fill(isl, have, u0, args.pop)
        run.gen = 1
        run.last_reset = time.time() - t0
        save(ck_path, args, run, t0)
        why_stop = stop_now()
    while why_stop is None and run.gen < args.gens:             # one round = one steady-state step per island
        for isl in ISLANDS:
            fam = fam_of(isl)
            o = ops[fam]
            if args.rgp > 0:
                top = sorted(((e["score"], c[0]) for c, e in run.hof.items() if e["basin"] == fam), reverse=True)[:MINE_TOP]
                o.mine([list(b[:-1]) for _, b in top])
            pop = run.pop[isl]
            if not pop:
                continue
            arch = run.arch[isl]
            want = n_new + max(0, args.pop - len(pop))          # a refounded island grows back to POP
            kids = fill(isl, [], lambda: o.child(lambda: parent(run.rng, pop, arch)), want)
            run.pop[isl] = sorted(pop + kids, key=rank_key)[:args.pop]
            for k, v in o.tries.items():
                run.tries[fam][k] += v
            o.tries.clear()
            why_stop = stop_now()
            if why_stop:
                break
        if why_stop is None:
            run.gen += 1
            if time.time() - t0 - run.last_reset >= args.reset_min * 60.0:
                reset()
        save(ck_path, args, run, t0)
        best = {f: round(-best_of(f)[0], 4) for f in ISLANDS}
        print("round %d u %d elapsed %s behav %d hof %d resets %d best %s" % (run.gen, run.u, TB._hms(time.time() - t0),
              len(run.seen_beh), len(run.hof), len(run.resets), best), flush=True)
        why_stop = why_stop or stop_now()
    form_fh.close()
    log_fh.close()
    scorer.close()
    finish(args, vm, run, out_dir, mods, t0, why_stop)


def save(path, args, run, t0):
    run.elapsed = time.time() - t0
    tmp = path + ".tmp"
    pickle.dump(dict(args=vars(args), run=run), open(tmp, "wb"))
    os.replace(tmp, path)


def finish(args, vm, run, out_dir, mods, t0, why_stop):
    top = sorted(run.hof.items(), key=lambda kv: -kv[1]["score"])[:500]
    json.dump(dict(schema="SEL1_HOF_V1", run=args.out_name, mode=args.mode, method="GA",
                   rule="A-half score %s (dd_max %s), top 500; reward_bpd = cost0 bp/day" % (args.reward, args.dd_max),
                   entries=[dict(body=list(c[0]), st=c[2], dr=c[3], **e) for c, e in top]),
              open(os.path.join(out_dir, "HOF.json"), "w", encoding="utf-8"), indent=1)
    cells = {}
    for isl in ISLANDS:
        for k, v in run.arch[isl].items():
            if k not in cells or v["score"] > cells[k]["score"]:
                cells[k] = dict(v, basin=fam_of(isl), island=isl)
    json.dump(dict(schema="SEL1_GA_ARCHIVE_V1", run=args.out_name, cells=[dict(h=k[0], state=S.STATES[k[1]], dir=S.DIRS[k[2]], **v)
                                                                       for k, v in sorted(cells.items())],
                   per_island={f: len(run.arch[f]) for f in ISLANDS}, resets=run.resets),
              open(os.path.join(out_dir, "ARCHIVE.json"), "w", encoding="utf-8"), indent=1)
    json.dump(dict(schema="SEL1_TRAIN_RECEIPT_V1", run=args.out_name, mode=args.mode, method="GA2", args=vars(args),
                   modules=mods, n_eval=run.n_eval, n_behav=len(run.seen_beh), generations=run.gen, updates=run.u,
                   invalid=dict(run.inval), operator_tries={f: dict(v) for f, v in run.tries.items()},
                   hof_size=len(run.hof), stopped=why_stop, compute_seconds=round(time.time() - t0, 1),
                   dev_read=False, oos_read=False, created=time.strftime("%Y-%m-%dT%H:%M:%S%z")),
              open(os.path.join(out_dir, "TRAIN_RECEIPT_V1.json"), "w"), indent=1)
    print("done gens %d updates %d n_eval %d behav %d seconds %.0f (%s)" % (run.gen, run.u, run.n_eval, len(run.seen_beh),
          time.time() - t0, why_stop), flush=True)


# --------------------------------------------------------------------------- self-check
N_CHILD = 2000


def selftest(args, vm, ctx, scorer, model, tabs, ops, out_dir, device):
    """1. Operators: per island N_CHILD children of a u0 population (crossover / point / subtree, and RGP splices
    when --rgp > 0): every returned child passes m2h_b5_grammar.check, the sampler mask and the VM's arity check
    (validate_and_order); the raw draw legality rate is reported. 2. --inject-plant: the plant body through the
    fitness (Scorer.run_sweep): combo and A-half score."""
    rng = random.Random(args.seed)
    rep = dict(schema="SEL1_GA_SELFTEST_V1", islands={}, n_child=N_CHILD)
    ok_all = True
    for fam in M.BASIN_NAMES:
        o = ops[fam]
        o.rng = rng
        pop = u0_bodies(model, vm, tabs[fam], fam, 256, device)
        u0_smp = sum(sampler_legal(b, fam) for b in pop)          # 256 = sampler_legal mirrors sample_batch_g
        u0_ok = sum(legal(b, fam) for b in pop)                   # minus bodies the VM refuses (window minimum)
        pop = [b for b in pop if legal(b, fam)]
        if args.rgp > 0:
            o.mine(pop[:MINE_TOP])
        kids, bad, kinds = 0, [], collections.Counter()
        for _ in range(N_CHILD):
            c, kind = o.child(lambda: pop[rng.randrange(len(pop))])
            if c is None:
                kinds["none"] += 1
                continue
            kids += 1
            kinds[kind] += 1
            body = list(c) + [vm.end_token]
            if (G.check(vm, body) is not None or RG.validate_and_order(vm, list(c), fam) is None
                    or not sampler_legal(c, fam) or len(body) > M.MAX_PROGRAM_LEN):
                bad.append(vm.describe(body, list(M.TERMINALS)))
        tr = dict(o.tries)
        raw = {k: round(tr.get(k + "_ok", 0) / tr[k], 4) for k in ("cx", "point", "subtree", "hoist", "shrink", "win", "rgp") if tr.get(k)}
        rep["islands"][fam] = dict(u0_sampler_legal=u0_smp, u0_legal=u0_ok, u0_n=256, children=kids, illegal_children=len(bad),
                                   illegal_examples=bad[:5], kinds=dict(kinds), raw_legal_rate=raw, mined=len(o.mined),
                                   distinct_children=None)
        ok_all = ok_all and not bad and kids > 0 and u0_smp == 256
        print("selftest %s: u0 sampler-legal %d/256 legal %d, children %d, illegal %d, raw legal %s"
              % (fam, u0_smp, u0_ok, kids, len(bad), raw),
              flush=True)
    rep["operators_ok"] = bool(ok_all)
    if args.inject_plant:
        body = tuple(D.plant_body(vm))
        tm = {}
        rb, _, _ = scorer.run_sweep([body], tm, args.reward, args.dd_max, {})
        r = rb[body]
        if isinstance(r, str):
            rep["plant"] = dict(invalid=r)
            ok_p = False
        else:
            combo, rows, s_rows, s_sweep = r
            rep["plant"] = dict(expr=vm.describe(list(body), list(M.TERMINALS)), h=combo[0], state=S.STATES[combo[1]],
                                dir=S.DIRS[combo[2]], score_A=float(s_rows), sweep_score=float(s_sweep),
                                t_A=float(S.reward(rows, "t")), bpd_A=float(S.bpd(rows)), seconds=tm)
            ok_p = combo[0] == 4 and combo[1] == 0 and combo[2] == 0 and round(float(S.reward(rows, "t")), 2) == 5.88
            print("selftest plant: %s/%s/%s t_A %.4f score %.4f" % (combo[0], S.STATES[combo[1]], S.DIRS[combo[2]],
                  S.reward(rows, "t"), s_rows), flush=True)
        rep["plant_ok"] = bool(ok_p)
        ok_all = ok_all and ok_p
    rep["passed"] = bool(ok_all)
    rep["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(rep, open(os.path.join(out_dir, "SEL1_GA_SELFTEST.json"), "w", encoding="utf-8"), indent=1, default=str)
    scorer.close()
    sys.exit(0 if ok_all else 7)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
