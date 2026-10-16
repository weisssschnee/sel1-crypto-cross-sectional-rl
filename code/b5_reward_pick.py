"""BASIN5 step 1: pick the training reward by re-scoring only, no training (BASIN5_ORDER s1).

    python b5_reward_pick.py

 wait     BASIN5 s0: starts only after RGP_P8_V2 and RGP_NULL_V2 both wrote FROZEN_RGP_TOP5.json
          and the RGP chain logged "END null" (one GPU, one job).
 pool     T1 of each run, distinct expr from RGP_LOG.jsonl, parsed back to tokens (round trip
          through vm.describe checked), re-scored in that run's own world (p8 planted / shuffled
          null, built as m2h_plant does) on the 50 TRAIN tiles at H168; per-tile scores kept.
 rewards  a gross mean; b gross tile t = tile mean / tile sd; c IC mean (current); d IC tile t;
          e residual IC: per hour the 168 h forward return is cs-ranked and OLS-residualised on
          the cs-ranks of mom168 / vol672 / size168 (gbm_xs_h2.resid_label), then the same
          Spearman as D.ic_tile.
 report   per reward (1) rank in the P8 pool of the P8 formula with the highest corr_s;
          (2) does the NULL pool's best score beat that formula's score.
 pick     among rewards with (2) = no, the best (1); tie -> simpler (a, c, b, d, e).
          None -> "p8 below the luck line on 2 years of TRAIN", stop.
Writes B5_REWARD_SCORES.npz and REWARD_PICK.json in artifacts\\m2h_v1\\BASIN5_20261012.
TRAIN only; DEV and OOS are never read.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m2h_selftest as ST            # sys.path for m1_* / okx_alpha_discovery
import m2h_dense_trainer as DT       # legacy v4 roots
import m2h_common as M
import m2h_dense_common as D
import m2h_plant as PL
import m2h_rgp as RG
from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid

ART = os.path.join(DT.ROOT, "artifacts", "m2h_v1")
OUT = os.path.join(ART, "BASIN5_20261012")
RUNS = dict(p8="RGP_P8_V2", null="RGP_NULL_V2")
CHAIN_LOG = r"D:\ChengboRemote\runtime\basin3_h1\RGP_V2_CHAIN.log"
REWARDS = ("a", "b", "c", "d", "e")
SIMPLE = ("a", "c", "b", "d", "e")           # tie-break order: simpler first
NAMES = dict(a="毛收益均值", b="毛收益段间 t", c="IC 均值（现行）", d="IC 段间 t", e="残差 IC 均值")
MAX_WAIT_H = 14.0                            # the null run's own cap is 12 h
SMOKE_N = 0


def wait_for_rgp():
    t0 = time.time()
    while True:
        frozen = all(os.path.exists(os.path.join(ART, r, "FROZEN_RGP_TOP5.json")) for r in RUNS.values())
        ended = os.path.exists(CHAIN_LOG) and b"=== END null" in open(CHAIN_LOG, "rb").read()
        if frozen and ended:
            return
        if time.time() - t0 > MAX_WAIT_H * 3600:
            raise SystemExit("RGP chain not finished after %.0f h" % MAX_WAIT_H)
        time.sleep(60)


# ---- gbm_xs_h1 helpers (lag / roll / plog / cs_rank), copied so this needs no lightgbm
def lag(x, k):
    y = np.full_like(x, np.nan)
    y[:, k:] = x[:, :-k]
    return y


def roll(x, w, fn="mean"):
    r = pd.DataFrame(x.T).rolling(w, min_periods=max(2, w // 2))
    return getattr(r, fn)().to_numpy().T


def plog(x):
    return np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan)


def cs_rank(x, univ):
    x = np.where(univ & np.isfinite(x), x, np.nan)
    return pd.DataFrame(x.T).rank(axis=1, pct=True).to_numpy().T - 0.5


def controls(grids, univ):
    """gbm_xs_h2.resid_label's regressors: cs-ranks of ret168, vol672, size168."""
    lc, qv = plog(grids["trade_close"]), grids["trade_quote_volume"]
    r1 = lc - lag(lc, 1)
    return [cs_rank(lc - lag(lc, 168), univ), cs_rank(roll(r1, 672, "std"), univ),
            cs_rank(plog(roll(qv, 168)), univ)]


def resid_future(shard, univ, Z, s0, cache):
    """ic_tile's (rows, future) with the future replaced, per hour, by the OLS residual of its
    cs-rank (pool at the signal hour s0 - LAG + row) on [1, Z]; < 20 coins -> NaN row (IC 0)."""
    if s0 not in cache:
        rows, fut = D.ic_future_rows(shard, s0)
        out = np.full(fut.shape, np.nan)
        for i, t in enumerate(s0 - M.LAG + rows):
            m = univ[:, t] & np.isfinite(fut[i])
            if m.sum() < D.IC_MIN_COINS:
                continue
            y = np.full(m.shape, np.nan)
            y[m] = pd.Series(fut[i][m]).rank(pct=True).to_numpy() - 0.5
            X = np.column_stack([np.ones(m.shape)] + [z[:, t] for z in Z])
            ok = m & np.isfinite(X).all(1)
            if ok.sum() < D.IC_MIN_COINS:
                continue
            b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
            out[i, ok] = y[ok] - X[ok] @ b
        cache[s0] = (rows, out)
    return cache[s0]
_TOK = re.compile(r"\s*(?:(w=)|([A-Za-z_][A-Za-z0-9_]*)|(\d+)|([(),]))")


def parse_expr(text):
    """vm.describe text -> postfix name list (`NAME(args, w=N)` -> args, W{N}, NAME)."""
    text, toks, pos = text.strip(), [], 0
    while pos < len(text):
        m = _TOK.match(text, pos)
        if not m or m.end() == pos:
            raise ValueError(text[pos:pos + 20])
        toks.append(next(g for g in m.groups() if g))
        pos = m.end()
    out, i = [], 0

    def node():
        nonlocal i
        name = toks[i]
        i += 1
        if i < len(toks) and toks[i] == "(":
            i += 1
            win = None
            while True:
                if toks[i] == "w=":
                    win = int(toks[i + 1])
                    i += 2
                else:
                    node()
                i += 1
                if toks[i - 1] == ")":
                    break
                if toks[i - 1] != ",":
                    raise ValueError(text)
            if win is not None:
                out.append(f"W{win}")
        out.append(name)

    node()
    if i != len(toks):
        raise ValueError(text)
    return out


def tstat(v):
    sd = float(np.std(v, ddof=1))
    return float(np.mean(v)) / sd if sd > 0 else float("nan")


def score_pool(vm, run, slices, starts, univ, sh, s, rf):
    # a line without its newline is one the (smoke-time) still-running search is writing
    rows = [json.loads(l) for l in open(os.path.join(ART, run, "RGP_LOG.jsonl"), encoding="utf-8")
            if l.endswith("\n")]
    first = {}
    for r in rows:
        if r["fam"] == "T1" and r["expr"] not in first:
            first[r["expr"]] = r
    if SMOKE_N:                              # the logged top-corr one first, so the target is in
        order = sorted(first, key=lambda e: -first[e]["corr_s"])
        first = {e: first[e] for e in order[:1] + order[len(order) - (SMOKE_N - 1):]}
    keep = dict(expr=[], n_tokens=[], logged_gross=[], logged_corr=[], gross=[], net=[], ic=[], ric=[], corr=[])
    bad_parse = invalid = 0
    t0 = time.time()
    for k, (expr, r) in enumerate(first.items()):
        try:
            names = parse_expr(expr)
            toks = RG.names_to_tokens(vm, names)
            if vm.describe(toks, list(M.TERMINALS)) != expr or RG.validate_and_order(vm, toks, "T1") is None:
                raise ValueError(expr)
        except (ValueError, IndexError):
            bad_parse += 1
            continue
        try:
            res = D.score_dense(vm, toks, slices, starts, univ, sh, policy_mod, holds=(168,),
                                costs=("flat6", "cost0"), want_ic=True, want_sig=True)
        except (M.ZeroCoverage, CandidateLocalInvalid):
            invalid += 1
            continue
        ric = [float(D.cs_spearman_rows(sg[rf[s0][0]], rf[s0][1]).mean()) for sg, s0 in zip(res["sig"], starts)]
        for key, val in (("expr", expr), ("n_tokens", len(names)), ("logged_gross", r["gross"]),
                         ("logged_corr", r["corr_s"]),
                         ("gross", [D.bpd(x) for x in res["per_hold"][168]["cost0"]]),
                         ("net", [D.bpd(x) for x in res["per_hold"][168]["flat6"]]),
                         ("ic", res["ic"]), ("ric", ric),
                         ("corr", float(np.mean([D.corr_with_s(sg, s, s0) for sg, s0 in zip(res["sig"], starts)])))):
            keep[key].append(val)
        if k % 200 == 0:
            print("%s %d/%d %.0fs" % (run, k, len(first), time.time() - t0), flush=True)
    out = {k: np.asarray(v) for k, v in keep.items()}
    g, ic = out["gross"], out["ic"]
    out["rw"] = dict(a=g.mean(1), b=np.array([tstat(x) for x in g]), c=ic.mean(1),
                     d=np.array([tstat(x) for x in ic]), e=out["ric"].mean(1))
    match = float(np.mean(np.abs(g.mean(1) - out["logged_gross"]) < 0.01))
    print("%s distinct %d scored %d bad_parse %d invalid %d gross==log %.3f" %
          (run, len(first), len(g), bad_parse, invalid, match), flush=True)
    out.update(n_distinct=len(first), bad_parse=bad_parse, invalid=invalid, log_match=match)
    return out


def rank_of(v, i):
    v = np.where(np.isfinite(v), v, -np.inf)
    return int((v > v[i]).sum()) + 1


def main():
    global OUT, SMOKE_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="score only N formulas per pool, on CPU, "
                    "into BASIN5_20261012\\smoke; no wait for the RGP chain")
    args = ap.parse_args()
    if args.smoke:
        OUT = os.path.join(OUT, "smoke")
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(os.path.join(OUT, "REWARD_PICK.json")):
        raise SystemExit("REWARD_PICK.json exists")
    if not args.smoke:
        wait_for_rgp()
    t_all = time.time()
    dev = torch.device("cpu" if args.smoke else "cuda")
    SMOKE_N = args.smoke
    vm = ST.make_vm()
    syms, ts, grids = M.load_train_grids(DT.PANEL)
    w = M.build_world(grids)
    univ, base = w["univ"], w["shard"]
    starts = M.tile_starts(w["feats"].shape[2])
    assert len(starts) == 50, len(starts)
    slices = M.tile_slices(M.pad_feats(w["feats"]), starts, dev)
    s = np.load(PL.S_FILE)
    cal = json.load(open(PL.ALPHA_FILE, encoding="utf-8"))["p8"]
    if PL.s_sha(s) != cal["s_sha16"]:
        raise RuntimeError("s differs from the calibrated one")
    Z = controls(grids, univ)
    del grids
    make = dict(p8=lambda: D.plant_shard(base, s, cal["alpha"]),        # as m2h_rgp.world
                null=lambda: D.null_shard(base, univ, D.PLANT_SEED))
    P = {}
    for key, run in RUNS.items():
        sh, rf = make[key](), {}
        for s0 in starts:
            resid_future(sh, univ, Z, s0, rf)
        P[key] = score_pool(vm, run, slices, starts, univ, sh, s, rf)
        del sh, rf
    np.savez_compressed(os.path.join(OUT, "B5_REWARD_SCORES.npz"),
                        **{f"{k}_{f}": P[k][f] for k in P for f in ("expr", "n_tokens", "gross", "net", "ic", "ric", "corr")})
    p8, nu = P["p8"], P["null"]
    i = int(np.argmax(p8["corr"]))
    table = {}
    for x in REWARDS:
        v, vn = p8["rw"][x], nu["rw"][x]
        j = int(np.nanargmax(vn))
        table[x] = dict(name=NAMES[x], target_rank=rank_of(v, i), of=len(v),
                        target_score=round(float(v[i]), 5), null_best=round(float(vn[j]), 5),
                        null_beats_target=bool(vn[j] > v[i]), null_best_expr=str(nu["expr"][j]),
                        p8_top_expr=str(p8["expr"][int(np.nanargmax(v))]))
    ok = [x for x in REWARDS if not table[x]["null_beats_target"]]
    pick = min(ok, key=lambda x: (table[x]["target_rank"], SIMPLE.index(x))) if ok else None
    reason = ("p8 在 2 年 TRAIN 上低于运气线：5 种奖励下 NULL 池第一名都高于植入近亲" if pick is None else
              "%s：NULL 第一名 %.4g 不高于植入近亲 %.4g，近亲排第 %d / %d（符合②的奖励里最好）"
              % (NAMES[pick], table[pick]["null_best"], table[pick]["target_score"],
                 table[pick]["target_rank"], table[pick]["of"]))
    out = dict(schema="B5_REWARD_PICK_V1", runs=RUNS, table=table, pick=pick, reason=reason,
               target=dict(expr=str(p8["expr"][i]), corr_s=round(float(p8["corr"][i]), 4),
                           n_tokens=int(p8["n_tokens"][i]),
                           gross_bpd=round(float(p8["rw"]["a"][i]), 3)),
               pools={k: dict(distinct=P[k]["n_distinct"], scored=len(P[k]["expr"]), bad_parse=P[k]["bad_parse"],
                              invalid=P[k]["invalid"], gross_matches_log=P[k]["log_match"]) for k in P},
               seconds=round(time.time() - t_all, 1), written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               dev_read=False, oos_read=False)
    json.dump(out, open(os.path.join(OUT, "REWARD_PICK.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("REWARD_PICK %s %.0fs" % (pick, time.time() - t_all), flush=True)


if __name__ == "__main__":
    main()
