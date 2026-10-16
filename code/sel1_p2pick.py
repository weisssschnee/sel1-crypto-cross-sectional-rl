"""XS_SEL1 SEL1_ARMS second plant P2, by rule (supplement 10-15 evening). TRAIN only (NULL world + P1); no DEV, no OOS.
New file; sel1_common / sel1_world2 / sel1_round are imported, not modified. Writes SEL1_ARMS_RULE.json (rule + choice).

  pool      sel1_calib's real candidate pool (CR1_PC_P8 HOF top 50 + B5_DBG_REAL HOF_RANKED), basin T2 or T3 only
            (CR1 entry 'basin'; B5 family), deduplicated, the P1 body excluded, valid on the V4 world.
  order     |cross-sectional Spearman corr| with P1's score (sel1_round.spearman_rows, every 24th A-half hour) in the
            ARMS world as the trainer sees it (pcnull: NULL roll + P1); smallest first.
  book      P2's own book = P2 formula, hold 72, state all, direction both (the order: hold 72), K 5.
  strength  alpha2 so that the P2 book's A-half cost0 gain over the P1-only world is 20 bp/day (one linear step from
            ALPHA0, as sel1_calib; |gain - target| <= 10 % checked); P2 book B-half monthly t >= 2 in the two-plant world,
            else 40 bp/day; else the next formula.
    python -I sel1_p2pick.py [--device cpu|cuda]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import sel1_common as S
import sel1_fastts as F
import sel1_world2 as W2
import sel1_round as R
import cr1_common as Q
import m2h_common as M
import m2h_dense_common as D

OUT = os.path.join(S.OUT_ROOT, "SEL1_ARMS_RULE.json")
CR1_HOF = os.path.join(S.A_ROOT, "CR1_PC_P8", "HOF.json")
B5_HOF = os.path.join(S.A_ROOT, "B5_DBG_REAL", "HOF_RANKED.json")
ALPHA_PCNULL = 6.908190339580098e-05                             # P1, the V2 chain's pcnull alpha (_flags_PC_V2.txt)
P2_COMBO = (72, 0, 0)                                            # hold 72, all, both
TARGETS, ALPHA0, T_MIN, TOL = (20.0, 40.0), 1e-4, 2.0, 0.10


def log(m):
    print(m, flush=True)


def pool(vm, pb):
    out = []
    for e in json.load(open(CR1_HOF, encoding="utf-8"))["entries"][:50]:
        if e.get("basin") in ("T2", "T3"):
            out.append((tuple(e["body"]), e["basin"], "CR1_PC_P8"))
    for fam, ents in json.load(open(B5_HOF, encoding="utf-8"))["families"].items():
        if fam in ("T2", "T3"):
            out += [(tuple(e["body"]), fam, "B5_DBG_REAL") for e in ents]
    seen, res = set(), []
    for b, fam, src in out:
        if b in seen or b == pb:
            continue
        seen.add(b)
        res.append((b, fam, src))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    t0 = time.time()
    if os.path.exists(OUT):
        raise SystemExit("%s exists" % OUT)
    torch.set_num_threads(4)
    dev = torch.device(a.device)
    vm = Q.make_vm()
    F.install()
    pb = tuple(D.plant_body(vm))
    ctx = S.build("pcnull", dev, vm, alpha=ALPHA_PCNULL, log=log, keep_base=True)
    feats, univ = W2.load_feats()
    hours = np.concatenate([np.arange(m["m0"], m["m1"]) for m in ctx["months"] if m["half"] == "A"])[::R.HOUR_STEP]
    sp = S.signal(ctx, vm, pb)
    cands, bad = [], {}
    for b, fam, src in pool(vm, pb):
        ex = vm.describe(list(b), list(M.TERMINALS))
        try:
            sg = S.signal(ctx, vm, b)
        except Exception as e:                                      # not valid on the V4 world: out (as sel1_calib)
            bad[ex] = str(e)[:80]
            continue
        c = R.spearman_rows(sg[:, hours].T, sp[:, hours].T)
        cands.append(dict(body=list(b), expr=ex, basin=fam, src=src, corr_p1=c))
        log("%-4s %+.4f %s" % (fam, c, ex))
    cands.sort(key=lambda x: abs(x["corr_p1"]) if np.isfinite(x["corr_p1"]) else 9.0)
    h, st, dr = P2_COMBO
    tried, pick = [], None
    for cand in cands:
        b = tuple(cand["body"])
        sg = S.signal(ctx, vm, b)
        z2 = W2.shifted_rankz(vm, feats, univ, dev, ctx["offsets"], b)
        g_base = S.bpd(S.score(sg, ctx, h, st, dr))

        def gain(alpha):
            return S.bpd(S.score(sg, W2.with_plant2(ctx, z2, alpha), h, st, dr)) - g_base
        g0 = gain(ALPHA0)
        for tgt in TARGETS:
            al = ALPHA0 * tgt / g0 if g0 > 0 else float("nan")
            if not np.isfinite(al):
                tried.append(dict(cand, target=tgt, alpha2=None, why="gain0 %.3f <= 0" % g0))
                break
            w = W2.with_plant2(ctx, z2, al)
            g1 = S.bpd(S.score(sg, w, h, st, dr)) - g_base
            rb = S.score(sg, w, h, st, dr, halves=("B",))
            tb = S.tstat([1e4 * r["cost0"] / r["days"] for r in rb])
            ok = bool(abs(g1 - tgt) <= TOL * tgt and tb >= T_MIN)
            row = dict(cand, target=tgt, alpha2=al, gain0=g0, gain=g1, tB=tb, B=S.summary(rb), ok=ok)
            tried.append(row)
            log("try %s target %.0f alpha2 %.3e gain %.2f B t %.2f -> %s" % (cand["expr"], tgt, al, g1, tb, ok))
            if ok:
                pick = row
                break
        if pick:
            break
    rep = dict(schema="SEL1_ARMS_RULE_V1", order="XS_SEL1 supplement 10-15 evening (SEL1_ARMS)", rule=__doc__,
               world=dict(mode="pcnull", alpha1=ALPHA_PCNULL, p1_body=list(pb), p1_combo="4/all/both (V2 PC)"),
               p2_combo=dict(h=h, state=S.STATES[st], dir=S.DIRS[dr]), candidates=cands, invalid=bad, tried=tried,
               p2=None if pick is None else dict(body=pick["body"], expr=pick["expr"], basin=pick["basin"],
                                                  alpha=pick["alpha2"], target_bpd=pick["target"], corr_p1=pick["corr_p1"],
                                                  tB=pick["tB"]),
               arms=dict(hours_each=3.0, list=["V2 RL (--combo-sweep)", "V2 + --dedup", "V2 + --gfn 1", "GA (sel1_ga)",
                                               "frozen u0 (--frozen)", "V2 + --pool 20"],
                         rank=["plants recovered (hall or pool member with corr > 0.7 to P1 / P2, B-hour Spearman)",
                               "time to recover (within 20 % = tie)", "composite book B t (sel1_combo)"],
                         report_only="behaviour-distinct formulas with A-half t > 3"),
               dev_read=False, oos_read=False, seconds=round(time.time() - t0, 1),
               written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=1, default=float)
    log("P2 -> %s" % (rep["p2"] and "%s alpha2 %.3e (%.0f bp/day) corr %.3f B t %.2f"
                      % (rep["p2"]["expr"], rep["p2"]["alpha"], rep["p2"]["target_bpd"], rep["p2"]["corr_p1"], rep["p2"]["tB"])))
    sys.exit(0 if pick else 7)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
