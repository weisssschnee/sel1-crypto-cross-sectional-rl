"""CR1b step 0 and pre-run checks (QuantMind 10-14 CR1b approval, points 1-2; AGENTS.md veto rule). V4 TRAIN
only; no DEV, no OOS. Refuses if CR1B_ALPHA.json exists. Written before any CR1b number.

  orth     per hour corr of s_perp8 with C and with each active cell's z over the regression sample:
           max mean |corr| < 0.05 (the residual is orthogonal by construction; this checks the code).
  alpha    s_perp8 alone, full book, cost0, mean over the 30 A tiles: lift over the real world = +8 / +32
           bp/day (within 2 %), secant from CR1's p8 alpha.
  visible  p8 world, the plant formula as the residualised ninth cell, B half: marginal net mean > 0,
           t >= 2, corr_s(B) > 0.7.
  rank     p8 world, B half: the plant formula's rank among the 50 CR1 PC candidates
           (CR1_PC_P8\\ROUND_PC.json, a pool a marginal-reward trainer actually reached in a p8 world)
           re-scored under the CR1b marginal, same book: rank <= 10 (rank = 1 + candidates strictly above).
  book     decisions on concentrated books only (QuantMind E3). Declared now: rank_book = conc_k10 if K10
           passes visible and rank; else conc_k20 if K20 passes both; else no launch, hand to the user
           (no other fallback, no alpha / pool / threshold change).
  reported the plant row in real / null / p8 / p32 x K10 / K20 / full x A / B; the candidates' max in the
           NULL world (a proxy of the NULL bar, not a gate).

    python -I cr1b_calib.py     rc 0 launch, 6 alpha, 7 not visible, 8 orthogonality, 9 rank > 10
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cr1b_common as Q
import numpy as np
import torch
import m2h_common as M
import m2h_dense_common as D
import v4_world as X
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid

PREREG = dict(written="2026-10-14 (77o), before any CR1b number", p_visible_k10_or_k20=0.35,
              p_rank_le10_given_visible=0.6, p_pc_found_given_launch=0.35, p_real_above_bar_given_found=0.1,
              why="CR1 p8 on B: K10 t -0.45, full t 2.27; residualising removes the premium / momentum reweights "
                  "that filled CR1's top 10, but a 1/9 cell is a small perturbation of a 10-coin book, so K10 "
                  "may still not see +8 bp/day.")
BOOKS_GATE = ("conc_k10", "conc_k20")


def log(m):
    print(m, flush=True)


def pcorr(a, c, min_coins=20):
    out = []
    for j in range(a.shape[1]):
        m = np.isfinite(a[:, j]) & np.isfinite(c[:, j])
        if m.sum() >= min_coins:
            out.append(float(np.corrcoef(a[m, j], c[m, j])[0, 1]))
    return np.asarray(out)


def gross_alone(sh, sp, starts):
    return float(np.mean([X.tile(X.tile_rows(sp, s0), sh, s0)["gross"] for s0 in starts]))


def score(ctx, vm, body, st, sl, corr=True):
    try:
        r = Q.marginal(ctx, vm, list(body), st, sl, books=Q.BOOKS3, corr_tiles=len(st) if corr else 0)
    except (M.ZeroCoverage, CandidateLocalInvalid) as e:
        return dict(invalid=str(e))
    out = dict(corr_s=round(r["corr"], 4), corr_raw=round(r["corr_raw"], 4))
    for b in ("conc_k10", "conc_k20", "full"):
        v = [Q.bpd(x) for x in r["books"][b]["net"]]
        out[b] = dict(mean=round(float(np.mean(v)), 4), t=round(Q.tstat(v), 2), n_pos=int(np.sum(np.array(v) > 0)))
    return out


def world(ctx, mode, alpha=None):
    base = ctx["base"]
    if mode == "real":
        sh = base
    elif mode == "null":
        sh = X.install_layer(D.null_shard(base, ctx["univ"], D.PLANT_SEED), ctx["dead"])
    else:
        sh = X.install_layer(D.plant_shard(base, ctx["s_perp"], alpha), ctx["dead"])
    return Q.set_world(ctx, sh)
def orth_check(ctx, step=24):
    cs, cz = [], {k: [] for k in ctx["fit"]["active"]}
    for s0 in ctx["A"] + ctx["B"]:
        sp = X.tile_rows(ctx["s_perp"], s0)[M.LAG::step].T          # (A, n)
        C = X.tile_rows(ctx["Cg"], s0)[M.LAG::step].T
        cs.append(pcorr(sp, C))
        for j, k in enumerate(ctx["fit"]["active"]):
            z = np.where(ctx["Z"][s0][j][M.LAG::step] != 0, ctx["Z"][s0][j][M.LAG::step], np.nan).T
            cz[k].append(pcorr(sp, z))
    out = dict(combo=round(float(np.abs(np.concatenate(cs)).mean()), 5),
               cells={k: round(float(np.abs(np.concatenate(v)).mean()), 5) for k, v in cz.items()},
               raw_vs_perp=round(float(np.nanmean([D.corr_with_s(X.tile_rows(ctx["s_raw"], s0), ctx["s_perp"], s0)
                                                   for s0 in ctx["A"]])), 4))
    out["max"] = max([out["combo"]] + list(out["cells"].values()))
    return out


def main():
    if os.path.exists(Q.ALPHA_JSON):
        raise SystemExit("%s exists: refusing" % Q.ALPHA_JSON)
    os.makedirs(Q.OUT, exist_ok=True)
    warnings.simplefilter("ignore", RuntimeWarning)
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vm = Q.make_vm()
    ctx = Q.build("real", device, vm, log=log, need_cal=False)
    np.save(Q.S_PERP_NPY, ctx["s_perp"])
    rep = dict(schema="CR1B_ALPHA_V1", prereg=PREREG, conventions=Q.__doc__, checks=__doc__,
               s_perp_sha16=ctx["s_perp_sha16"], fit=ctx["fit"], dev_read=False, oos_read=False)
    rep["orth"] = orth = orth_check(ctx)
    log("orth %s" % json.dumps(orth))
    # alpha: s_perp8 alone, full book, cost0, A tiles, lift over real
    base, A, B, sp = ctx["base"], ctx["A"], ctx["B"], ctx["s_perp"]
    g0 = gross_alone(base, sp, A)
    a0 = json.load(open(Q1_ALPHA, encoding="utf-8"))["p8"]["alpha"]
    for mode, tgt in Q.TARGETS.items():
        hist, a = [], a0 * tgt / 8.0
        for _ in range(6):
            lift = gross_alone(X.install_layer(D.plant_shard(base, sp, a), ctx["dead"]), sp, A) - g0
            hist.append([a, round(lift, 4)])
            if abs(lift / tgt - 1) < 0.02:
                break
            a = a * tgt / lift if lift > 0 else a * 2
        a, lift = hist[-1]                         # the alpha whose lift was measured
        rep[mode] = dict(alpha=a, lift_bpd=lift, history=hist, target=tgt)
        log("alpha %s %.4g lift %.3f (%d it)" % (mode, a, lift, len(hist)))
    alpha_ok = all(abs(rep[m]["lift_bpd"] / t - 1) < 0.02 for m, t in Q.TARGETS.items())
    # plant row in every world; the 50 CR1 PC candidates in p8 and null on B
    pool = json.load(open(Q.PC_POOL, encoding="utf-8"))
    pb = list(D.plant_body(vm))
    cands = [c for c in pool["candidates"] if list(c["body"]) != pb]
    slB = M.tile_slices(ctx["feats_pad"], B, device)
    slA = M.tile_slices(ctx["feats_pad"], A, device)
    rep["plant"], rep["pool"] = {}, dict(source=Q.PC_POOL, n=len(cands), dropped_plant_dupes=len(pool["candidates"]) - len(cands))
    ts = time.time()
    for mode in ("real", "null", "p8", "p32"):
        world(ctx, mode, rep[mode]["alpha"] if mode in Q.TARGETS else None)
        rep["plant"][mode] = dict(A=score(ctx, vm, pb, A, slA), B=score(ctx, vm, pb, B, slB))
        log("plant %s B %s" % (mode, json.dumps(rep["plant"][mode]["B"])))
        if mode in ("p8", "null"):
            rows = []
            for c in cands:
                r = score(ctx, vm, c["body"], B, slB)
                r.update(expr=c["expr"], basin=c["basin"])
                rows.append(r)
            rep["pool"][mode] = rows
            log("pool %s scored %d in %.0fs" % (mode, len(rows), time.time() - ts))
    world(ctx, "p8", rep["p8"]["alpha"])
    pl = rep["plant"]["p8"]["B"]
    gate = {}
    for b in BOOKS_GATE:
        vals = [r[b]["mean"] for r in rep["pool"]["p8"] if b in r]
        visible = bool(pl[b]["mean"] > 0 and pl[b]["t"] >= Q.RULER_T_MIN and pl["corr_s"] > Q.PLANT_CORR_MIN)
        rank = 1 + int(sum(v > pl[b]["mean"] for v in vals))
        nvals = [r[b]["mean"] for r in rep["pool"]["null"] if b in r]
        gate[b] = dict(visible=visible, mean=pl[b]["mean"], t=pl[b]["t"], corr_s=pl["corr_s"], rank=rank,
                       of=len(vals) + 1, rank_ok=rank <= Q.RANK_MAX, tenth=sorted(vals, reverse=True)[9] if len(vals) >= 10 else None,
                       null_pool_max=max(nvals) if nvals else None,
                       plant_above_null_pool_max=bool(nvals and pl[b]["mean"] > max(nvals)))
    rep["gate"] = gate
    rank_book = next((b for b in BOOKS_GATE if gate[b]["visible"] and gate[b]["rank_ok"]), None)
    rep.update(rank_book=rank_book, alpha_ok=alpha_ok, orth_ok=orth["max"] < Q.CORR_ORTH_MAX,
               seconds=round(time.time() - t0, 1), seconds_pool=round(time.time() - ts, 1),
               written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    rc = 0 if rank_book else (9 if any(g["visible"] for g in gate.values()) else 7)
    if not alpha_ok:
        rc = 6
    if not rep["orth_ok"]:
        rc = 8
    rep["rc"] = rc
    json.dump(rep, open(Q.ALPHA_JSON, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    log("CR1b calib rc %d rank_book %s gate %s %.0fs" % (rc, rank_book, json.dumps(gate), time.time() - t0))
    return rc


Q1_ALPHA = os.path.join(Q.A_ROOT, "CR1", "CR1_ALPHA.json")

if __name__ == "__main__":
    sys.exit(main())
