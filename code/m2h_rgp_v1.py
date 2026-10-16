"""TV1 RGP search layer (TV1_ORDER s7 second branch: "R2 not found -> more tiles, or the RGP
search layer"). Gradient-free: no policy gradient, no advantage, no entropy. Fitness is the
ruler's TRAIN gross bp/day, evaluated directly.

    python m2h_rgp.py --mode p32|p8|null --out RGP_P8_V1 [--rounds 8] [--pop 256]
                      [--keep 64] [--seed 20261011] [--max-hours 12]

What is reused and what is not
  reused   RGP's SubexpressionExtractor (core/exploitation.py, 154 lines of pure Python, no
           deap / numba / pysr) verbatim: preorder traversal, unique sub-expressions of
           length [min_L, max_L]. RGP's own loop shape: each round re-scores the hall of
           fame, counts symbol / sub-expression frequency, injects the top ones back in.
  ours     the grammar (m2h_common.BASIN_TERMS, 24 operators, 7 windows), the VM, the ruler
           and the plant / null world. deap / numba / pysr are NOT installed in m1cu128 and
           AGENTS.md pins that venv, so nothing from those is needed.

Fitness is deliberately the mean over all 50 TRAIN tiles, not a drawn sample: the R1 vs R2
comparison showed a drawn-sample advantage estimate is the thing that loses the p8 signal, and
RGP can afford the exact number because it evaluates far fewer candidates than PG did.

Seed candidates: the frozen top-5 of a finished round (--seed-formulas, default the R2d p32
run, which reached batch corr 0.331), converted to this grammar's token names. Generation 0 is
those plus random trees, so the search starts where the policy got to instead of from zero.

Every accepted formula is re-scored on the full 50 tiles and written to RGP_LOG.jsonl; the
final hall of fame is FROZEN_RGP_TOP5.json in the same shape tv1_round.py emits, so the two
paths compare 1:1. TRAIN only. DEV and OOS are never read here.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
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
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import OPERATOR_ARITIES, OPERATOR_NAMES, CandidateLocalInvalid

# RGP core/exploitation.py, copied verbatim (the only file reused; pure Python, no deps).
# --------------------------------------------------------------------------- subexpression miner
class TreeNode:
    def __init__(self, val: str):
        self.val = val
        self.children = []

    def __repr__(self):
        return f"TreeNode({self.val})"

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return isinstance(other, TreeNode) and id(self) == id(other)


class SubexpressionExtractor:
    def __init__(self, symbol_arity):
        self.symbol_arity = symbol_arity
        self._cache_preorder = {}
        self._cache_tree = {}

    def _build_tree(self, prefix):
        key = tuple(prefix)
        if key in self._cache_tree:
            return self._cache_tree[key]

        def _parse(rem):
            if not rem:
                raise ValueError("Input is empty")
            token = rem[0]
            arity = self.symbol_arity.get(token, 2)
            node = TreeNode(token)
            remaining = rem[1:]
            for _ in range(arity):
                if not remaining:
                    raise ValueError(f"Symbol '{token}' requires {arity} operands")
                child, remaining = _parse(remaining)
                node.children.append(child)
            return node, remaining

        root, leftover = _parse(prefix)
        if leftover:
            raise ValueError(f"Expression not fully parsed, remaining: {leftover}")
        self._cache_tree[key] = root
        return root

    def _preorder(self, node):
        if node is None:
            return []
        if node in self._cache_preorder:
            return self._cache_preorder[node][:]
        seq = [node.val]
        for child in node.children:
            seq.extend(self._preorder(child))
        self._cache_preorder[node] = seq[:]
        return seq

    def _extract_from_tree(self, root, min_len, max_len):
        seen = set()

        def _traverse(node):
            if node is None:
                return
            seq = tuple(self._preorder(node))
            if min_len <= len(seq) <= max_len:
                seen.add(seq)
            for child in node.children:
                _traverse(child)

        _traverse(root)
        return seen

    def extract(self, expressions, min_len, max_len):
        out = set()
        for expr in expressions:
            try:
                out |= self._extract_from_tree(self._build_tree(list(expr)), min_len, max_len)
            except (ValueError, RecursionError):
                continue
        return out


# --------------------------------------------------------------------------- grammar
ARITY = dict(OPERATOR_ARITIES)
MIN_WIN = {"LAG": 1, "DELTA": 1, "TS_MEAN": 1, "TS_STD": 2, "TS_MAX": 1, "TS_MIN": 1,
           "TS_ZSCORE": 2, "TS_RANK": 2, "TS_SLOPE": 2, "EMA": 1, "TS_CORR": 2}
WINDOWED = {n for n in MIN_WIN}


def miner_arity():
    """Arity as the sub-expression miner's parser must see it: terminals and window selectors are
    leaves, operators take their value arity PLUS one for their window when windowed.

    Both halves of that are needed. Terminals are absent from OPERATOR_ARITIES and the miner's
    parser defaults an unknown symbol to arity 2, so a terminal parsed as binary. And the
    converted prefix form is `TS_MEAN W168 log_quote_volume`, so the miner itself must consume
    two children for TS_MEAN, not one.
    """
    a = {n: ARITY[n] + (1 if n in WINDOWED else 0) for n in OPERATOR_NAMES}
    a.update({f"W{w}": 0 for w in M.WINDOWS})
    a.update({t: 0 for t in M.TERMINALS})
    return a


def postfix_to_prefix(body, arity=None):
    """Our programs are POSTFIX (`x y ADD`, `x W168 TS_MEAN`); RGP's SubexpressionExtractor
    parses PREFIX (`ADD x y`). That mismatch is why it mined nothing: even a plain
    `log_close log_quote_volume ADD` fails to parse. Convert into the notation the miner reads.

    A windowed op consumes only its value arity here; its window is the sibling subtree that
    precedes it in the postfix stream, so it becomes one more child node of the op.
    """
    arity = arity or miner_arity()
    stack = []
    for tok in body:
        # miner_arity already charges a windowed op its window, which is exactly the extra stack
        # item our postfix form leaves for it (`x W168 TS_MEAN`).
        a = arity.get(tok, 0)
        if a == 0:
            stack.append([tok])
            continue
        args = []
        for _ in range(a):
            if not stack:
                raise ValueError("stack underflow")
            args.append(stack.pop())
        # args are popped newest-first, so reversing restores our operand order: the first
        # prefix child is our first postfix operand (SUB / DIV / GATE / PER_ASSET_THRESH).
        stack.append([tok] + args[::-1])
    if len(stack) != 1:
        raise ValueError("stack not empty")
    return _flatten_prefix(stack[0])


def _flatten_prefix(node):
    """Nested-list prefix tree -> the flat token sequence the miner parses. The miner rebuilds
    its own tree from the sequence and keys its cache on it, so the value must be hashable; a
    nested list is not. Recursive, because a node's children are themselves nested lists."""
    out = []

    def walk(n):
        out.append(n[0])
        for c in n[1:]:
            walk(c)

    walk(node)
    return out


def prefix_to_postfix(pre, arity=None):
    """A mined sub-expression -> our postfix notation.

    SubexpressionExtractor.extract() yields NESTED tuples ((tok, child...), ...), not a flat
    sequence, so this walks that shape directly. Same window accounting as postfix_to_prefix,
    which is already inside miner_arity.
    """
    arity = arity or miner_arity()
    out = []

    def walk(node):
        # extract() yields nested tuples of STRINGS: a leaf is a bare name, a branch is
        # (op, child, ...). Take the name off whatever node we get.
        if isinstance(node, str):
            out.append(node)
            return
        tok = node[0]
        kids = node[1:1 + arity.get(tok, 0)]
        # prefix children are already in our operand order (postfix_to_prefix reverses the pop
        # order), so children are emitted in the order they appear.
        for c in kids:
            walk(c)
        out.append(tok)

    walk(tuple(pre) if not isinstance(pre[0], (tuple, list)) else pre)
    return out


def token_names(vm, fam):
    """This grammar's symbol vocabulary as name strings: terminals, window selectors (W{w}),
    operators. The RGP miner works on names, so the whole pipeline is name-level and the VM
    token ints are produced only at scoring time. END is not in the vocabulary: a mutation
    target must be a token the program can still be built from."""
    terms = [M.TERMINALS[i] for i in M.basin_terms(fam)]
    wins = [f"W{w}" for w in M.WINDOWS]
    return terms + wins + list(OPERATOR_NAMES)


def rand_tree(rng, vocab, fam, max_len, max_depth):
    """A random RPN body in the family grammar. Windowed operators take a trailing window
    selector; every terminal and window is legal, operators are restricted to the ones
    that keep the value in the same dimension class loosely (we do not enforce the sampler's
    dimension rules here; the scorer rejects anything the VM or the ruler refuses).

    A windowed op is paired with a window that satisfies its minimum, otherwise the VM refuses
    the program as WINDOW_BELOW_OPERATOR_MINIMUM and the draw is wasted. TS_STD/TS_ZSCORE/
    TS_RANK/TS_SLOPE/TS_CORR need >= 2h and TS_SLOPE/TS_STD are the tightest, so the window is
    drawn from the op's valid range rather than uniformly from all seven."""
    terms = [M.TERMINALS[i] for i in M.basin_terms(fam)]
    ops = [n for n in OPERATOR_NAMES]

    def win_for(name):
        lo = MIN_WIN[name]
        ok = [w for w in M.WINDOWS if w >= lo]
        return "W%d" % rng.choice(ok or list(M.WINDOWS))

    out = []

    def emit(d):
        if len(out) >= max_len - 1 or d >= max_depth:
            out.append(rng.choice(terms))
            return
        r = rng.random()
        if r < 0.45:
            out.append(rng.choice(terms))
        else:
            name = rng.choice(ops)
            ar = ARITY[name]
            if name in WINDOWED:
                out.append(win_for(name))
                ar -= 1
            for _ in range(ar):
                emit(d + 1)
            out.append(name)

    emit(0)
    return out


def validate_and_order(vm, body, fam):
    """VM-level legality: the body must be a well-formed RPN program in the family's terminals
    that executes without CandidateLocalInvalid. Returns the int-token body or None."""
    terms = set(M.basin_terms(fam))
    stack = []
    for t in body:
        if vm.is_window(t):
            stack.append(("w", None))
            continue
        if vm.is_terminal(t):
            if t not in terms:
                return None
            stack.append(("v", t))
            continue
        name = vm.operator_name(t)
        ar = ARITY[name]
        need = ar - 1 if name in WINDOWED else ar
        if name in WINDOWED:
            if not stack or stack[-1][0] != "w":
                return None
            stack.pop()
        if len(stack) < need:
            return None
        for _ in range(need):
            if stack.pop()[0] != "v":
                return None
        stack.append(("v", None))
        if len(stack) != 1:
            return None
    return body if len(stack) == 1 else None


def names_to_tokens(vm, body):
    """Name list -> VM token ints (terminals / W{w} / operators). vm.operator_base is
    feature_count and vm.window_base is feature_count + len(OPERATOR_NAMES), both from the VM
    constructor; there is no feature_index helper. A trailing END is dropped: mutated trees may
    carry one, and the VM treats END as a terminator anyway."""
    out = []
    for n in body:
        if n == "END":
            continue
        if n in M.TERMINALS:
            out.append(M.TERMINALS.index(n))
        elif n.startswith("W") and n[1:].isdigit():
            out.append(vm.window_base + list(M.WINDOWS).index(int(n[1:])))
        else:
            out.append(vm.operator_base + list(OPERATOR_NAMES).index(n))
    return out


def tokens_to_names(vm, toks):
    out = []
    for t in toks:
        if vm.is_end(t):
            continue
        if vm.is_terminal(t):
            out.append(M.TERMINALS[t])
        elif vm.is_window(t):
            out.append(f"W{list(M.WINDOWS)[t - vm.window_base]}")
        else:
            out.append(vm.operator_name(t))
    return out
def world(mode):
    """The TRAIN world this search sees: real, p8/p32 planted, or the shuffled null, built
    exactly as m2h_plant builds it so RGP's numbers compare with the RL rounds'."""
    syms, ts, grids = M.load_train_grids(DT.PANEL)
    w = M.build_world(grids)
    base = w["shard"]
    if mode == "real":
        sh = base
    elif mode == "null":
        sh = D.null_shard(base, w["univ"], D.PLANT_SEED)
    else:
        cal = json.load(open(PL.ALPHA_FILE, encoding="utf-8"))[mode]
        s = np.load(PL.S_FILE)
        if PL.s_sha(s) != cal["s_sha16"]:
            raise RuntimeError("s differs from the calibrated one")
        sh = D.plant_shard(base, s, cal["alpha"])
    return w, sh


def score(vm, body_names, slices, starts, univ, shard, want_sig=False):
    """Mean TRAIN gross bp/day over all 50 tiles. None if the formula is invalid. The ruler
    is the only scorer; nothing here re-implements it."""
    toks = names_to_tokens(vm, body_names)
    try:
        res = D.score_dense(vm, toks, slices, starts, univ, shard, policy_mod, holds=(168,),
                            costs=("flat6", "cost0"), want_ic=True, want_sig=want_sig)
    except (M.ZeroCoverage, CandidateLocalInvalid):
        return None
    g = np.asarray(res["per_hold"][168]["cost0"], dtype=np.float64)
    n = np.asarray(res["per_hold"][168]["flat6"], dtype=np.float64)
    if not (np.isfinite(g).all() and np.isfinite(n).all()):
        return None
    return dict(gross=D.bpd(g.mean()), net=D.bpd(n.mean()), ic=float(np.mean(res["ic"])),
                gross_tiles=g, sig=res["sig"] if want_sig else None)


def seed_pool(vm, run, fam, limit=5):
    """The frozen top-5 of a finished RL round, as name lists, so generation 0 starts where
    the policy reached. Empty list if the report is absent. TV1 reports store the frozen body
    as token ints and the expression as text; the token ints go through tokens_to_names."""
    p = os.path.join(DT.ROOT, "artifacts", "m2h_v1", run, "ROUND_REPORT.json")
    if not os.path.exists(p):
        return []
    rep = json.load(open(p, encoding="utf-8"))
    return [tokens_to_names(vm, [int(x) for x in f["body"]]) for f in rep["frozen"].get(fam, [])[:limit]]


def mutate(rng, names, vocab, injected, p_inject, max_len):
    """One point mutation: with prob p_inject splice a mined sub-expression in at a random
    position, otherwise rewrite one token. Sub-expression injection is RGP's mechanism: the
    building blocks that recur in good formulas get re-used instead of rediscovered."""
    out = list(names)
    if injected and rng.random() < p_inject:
        sub = list(rng.choice(sorted(injected)))
        pos = rng.randrange(0, max(1, len(out) - 1))
        out = out[:pos] + sub + out[pos:]
    else:
        i = rng.randrange(len(out))
        out[i] = rng.choice(vocab)
    return out[:max_len]


def replace_subtree(rng, names, injected, p_inject, max_len):
    """Crossover-ish: replace a whole subtree of length [3, 8] with a mined one."""
    out = list(names)
    if not (injected and rng.random() < p_inject):
        return out
    lo = rng.randrange(0, max(1, len(out) - 3))
    hi = min(len(out), lo + rng.randrange(3, 8))
    sub = list(rng.choice(sorted(injected)))
    return (out[:lo] + sub + out[hi:])[:max_len]


def search(args):
    t_all = time.time()
    rng = random.Random(args.seed)
    vm = ST.make_vm()
    w, shard = world(args.mode)
    univ = w["univ"]
    starts = M.tile_starts(w["feats"].shape[2])
    assert len(starts) == 50, len(starts)
    slices = M.tile_slices(M.pad_feats(w["feats"]), starts, torch.device(args.device))
    s_plant = np.load(PL.S_FILE)
    out_dir = os.path.join(DT.ROOT, "artifacts", "m2h_v1", args.out)
    os.makedirs(out_dir, exist_ok=True)
    log_fh = open(os.path.join(out_dir, "RGP_LOG.jsonl"), "w", encoding="utf-8")
    extractor = SubexpressionExtractor(miner_arity())
    best = {fam: [] for fam in M.BASIN_NAMES}          # list of (gross, names, rec)
    total_eval = 0

    def evaluate(fam, cand, injected):
        """Score a batch of candidates. The per-tile signal is needed only for the corr with
        the plant signal s, so it is measured here and dropped: keeping it would cost ~1.3 MB
        per formula per 50 tiles."""
        nonlocal total_eval
        seen, pool = set(), []
        for c in cand:
            k = tuple(c)
            if k in seen:
                continue
            seen.add(k)
            rec = score(vm, c, slices, starts, univ, shard, want_sig=True)
            total_eval += 1
            if rec is None:
                continue
            if validate_and_order(vm, names_to_tokens(vm, c), fam) is None:
                continue
            cs = [D.corr_with_s(sg, s_plant, s0) for sg, s0 in zip(rec["sig"], starts)]
            rec["corr_s"] = float(np.mean(cs))
            rec.pop("sig", None)
            pool.append((rec["gross"], list(c), rec))
            log_fh.write(json.dumps(dict(round=state["round"], fam=fam, gross=round(rec["gross"], 3),
                                        net=round(rec["net"], 3), ic=round(rec["ic"], 5),
                                        corr_s=round(rec["corr_s"], 4),
                                        n_tokens=len(c), n_subexpr=len(injected),
                                        expr=vm.describe(names_to_tokens(vm, c), list(M.TERMINALS)))) + "\n")
            best[fam].append((rec["gross"], list(c), rec))
        return pool

    state = dict(round=0)
    print("world %s tiles 50 device %s" % (args.mode, args.device), flush=True)
    for rnd in range(args.rounds):
        state["round"] = rnd
        t0 = time.time()
        # 1. mine sub-expressions from what has scored well so far (RGP's exploitation step)
        injected = set()
        for fam in M.BASIN_NAMES:
            # Mine from the top of the hall of fame per LENGTH BUCKET, not the global top-K.
            # RGP's mechanism is sub-expression reuse, and a 1-token formula like `log_close`
            # has no sub-expression of length [3, 8] to contribute; a purely greedy top-K on a
            # weak-signal world is all 1-token, which starves the miner (smoke run: 0 mined).
            # Bucketing keeps the deepest good formulas represented without weakening selection.
            top = sorted(best[fam], key=lambda x: -x[0])[:args.keep]
            by_len = collections.defaultdict(list)
            for g, names, _ in top:
                by_len[len(names)].append(names)
            for bucket in by_len.values():
                # postfix -> prefix for the verbatim miner, and back for the injection pool
                pref = []
                for names in bucket[:max(1, args.keep // 8)]:
                    try:
                        pref.append(postfix_to_prefix(names))
                    except ValueError:
                        continue
                if pref:
                    got = extractor.extract(pref, args.exploit_min_L, args.exploit_max_L)
                    injected |= {tuple(prefix_to_postfix(list(g))) for g in got}
        if time.time() - t_all > args.max_hours * 3600:
            break
        # 2. generate and score this round's candidates
        n_new = 0
        for fam in M.BASIN_NAMES:
            vocab = token_names(vm, fam)
            cand = []
            if rnd == 0:                                # generation 0 = seeds + random trees
                cand += seed_pool(vm, args.seed_run, fam, args.keep)
                cand += [rand_tree(rng, vocab, fam, args.max_len, args.max_depth)
                         for _ in range(args.pop)]
            else:                                       # mutate the hall of fame
                top = sorted(best[fam], key=lambda x: -x[0])[:args.keep]
                for _, names, _ in top:
                    for _ in range(max(1, args.pop // max(len(top), 1))):
                        c = mutate(rng, names, vocab, injected, args.p_inject, args.max_len)
                        c2 = replace_subtree(rng, names, injected, args.p_inject, args.max_len)
                        cand += [c, c2]
                cand += [rand_tree(rng, vocab, fam, args.max_len, args.max_depth)
                         for _ in range(args.pop // 4)]
            pool = evaluate(fam, cand, injected)
            n_new += len(pool)
            best[fam] = sorted(best[fam], key=lambda x: -x[0])[:max(args.keep * 4, 64)]
        log_fh.flush()
        tops = {fam: round(max((g for g, _, _ in best[fam]), default=float("nan")), 3)
                for fam in M.BASIN_NAMES}
        print("round %d new %d total %d best_gross %s top_subexpr %d %.0fs"
              % (rnd, n_new, total_eval, tops, len(injected), time.time() - t0), flush=True)
        if time.time() - t_all > args.max_hours * 3600:
            break

    log_fh.close()
    # freeze: dedup on the 50-tile net vector, top 5 per family, same shape as tv1_round
    frozen = {}
    for fam in M.BASIN_NAMES:
        scored = []
        for g, names, rec in sorted(best[fam], key=lambda x: -x[0]):
            scored.append((g, names, rec))
        kept, seen_net = [], []
        for g, names, rec in scored:
            v = rec["gross_tiles"]
            if any(np.array_equal(v, k) for k in seen_net):
                continue
            seen_net.append(v)
            kept.append(dict(rank=len(kept) + 1, names=names,
                             body=names_to_tokens(vm, names), hold=168,
                             expr=vm.describe(names_to_tokens(vm, names), list(M.TERMINALS)),
                             train50_gross_bpd=round(g, 3), train50_net_bpd=round(rec["net"], 3),
                             ic=round(rec["ic"], 5), n_tokens=len(names),
                             corr_s=round(rec["corr_s"], 4),
                             level_flag=bool(D.level_flag(vm, names_to_tokens(vm, names)))))
            if len(kept) == 5:
                break
        frozen[fam] = kept
    blob = json.dumps(frozen, sort_keys=True).encode()
    out = dict(schema="TV1_RGP_FROZEN_V1", run=args.out, mode=args.mode, rounds=state["round"] + 1,
               pop=args.pop, keep=args.keep, seed=args.seed, total_evaluated=total_eval,
               seed_run=args.seed_run, sha256=hashlib.sha256(blob).hexdigest(),
               max_net_corr_s=max((f["corr_s"] for fam in frozen.values() for f in fam), default=None),
               families=frozen, seconds=round(time.time() - t_all, 1),
               written=time.strftime("%Y-%m-%dT%H:%M:%S%z"), dev_read=False, oos_read=False)
    json.dump(out, open(os.path.join(out_dir, "FROZEN_RGP_TOP5.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    print("RGP done %s evaluated %d sha %s %.0fs" % (args.out, total_eval, out["sha256"][:12],
                                                    time.time() - t_all), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("p32", "p8", "null", "real"), required=True)
    ap.add_argument("--seed-run", default="TV1_R2D_DENSE_P32")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--pop", type=int, default=256)
    ap.add_argument("--keep", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=M.MAX_PROGRAM_LEN)
    ap.add_argument("--max-depth", type=int, default=8)
    ap.add_argument("--p-inject", type=float, default=0.5)
    ap.add_argument("--exploit-min-L", type=int, default=3)
    ap.add_argument("--exploit-max-L", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    search(ap.parse_args())


if __name__ == "__main__":
    main()