"""XS_SEL1 SEL1_ARMS ranking (supplement 10-15 evening). TRAIN, ARMS world (pcnull + P1 + P2, SEL1_ARMS_RULE.json);
no DEV, no OOS. New file; sel1_round / sel1_combo / sel1_pool are imported, not modified.

  arms      PREFIX + RL / DEDUP / GFN / GA / FROZEN / POOL (finished, receipt present; a max-hours stop writes one).
  recovery  every hall entry (HOF.json, top 500) and, for POOL, every POOL.json member is re-run in the ARMS world;
            a plant is recovered when |corr| > 0.7 (sel1_round.spearman_rows, every 24th B hour) with that plant's
            own signal. time = seconds (the entry's first_s) at which the last recovered plant was first recovered.
  rank      1 plants recovered (more first) 2 time: arms within 20 % of the best time in the group tie 3 composite
            book B t (sel1_combo on the hall: greedy <= 20 members, rank z x sign t_A, combo by A t_dd). The POOL
            arm also gets its pool composite (members sign +1), reported, not ranked.
  report    behaviour-distinct hall entries with A-half t > 3 (sel1_dedup hash of the A selections), not ranked.
    python -I sel1_arms_rank.py [--prefix SEL1_ARMS_] [--top N] [--device cuda]
-> <prefix>RESULT.json; rc 0 V2 RL (PREFIX RL) first, 10 another arm first, 7 an arm missing.
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
import sel1_round as R
import sel1_combo as C
import sel1_pool as P
import sel1_dedup as DD
import sel1_world2 as W2
import cr1_common as Q
import m2h_common as M
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid

RULE_PATH = os.path.join(S.OUT_ROOT, "SEL1_ARMS_RULE.json")
ARMS = (("V2 RL", "RL"), ("V2 + --dedup", "DEDUP"), ("V2 + --gfn 1", "GFN"), ("GA", "GA"),
        ("V2 + --frozen", "FROZEN"), ("V2 + --pool 20", "POOL"))
T_TIE, T_GOOD = 1.2, 3.0
NONE_T = float("inf")


def log(m):
    print(m, flush=True)


def scan(entries, ctx, vm, plants, hours):
    """entries (body, first_s, and h/st/dr for the hall) -> per plant the earliest first_s (inf = not recovered),
    the recovering exprs, and the behaviour-distinct count with A t > 3 (hall entries only)."""
    first = [NONE_T] * len(plants)
    hits = [[] for _ in plants]
    good, cache = set(), {}
    for e in entries:
        b = tuple(e["body"])
        if b not in cache:
            try:
                s = P.masked_signal(ctx, vm, b)
            except (S.Invalid, CandidateLocalInvalid, M.ZeroCoverage):
                cache = {b: None}
                continue
            sg = s.cpu().numpy()
            cache = {b: (s, [abs(R.spearman_rows(sg[:, hours].T, p[:, hours].T)) for p in plants])}
            del sg
        if cache[b] is None:
            continue
        s, cs = cache[b]
        t_s = float(e.get("first_s", NONE_T))
        for k, c in enumerate(cs):
            if c > R.CORR_MIN:
                hits[k].append(dict(expr=e["expr"], corr=round(c, 4), first_s=t_s))
                first[k] = min(first[k], t_s)
        if "h" in e:
            sels = P.select_signal(ctx, s, ("A",))
            if float(S.reward(S.score_sel(sels, ctx, e["h"], e["st"], e["dr"]), "t")) > T_GOOD:
                good.add(DD.behaviour_hash(sels))
    return first, hits, len(good)


def order(rows):
    """Plants desc; within equal plants, repeatedly: the best time t*, arms with time <= 1.2 t* tie and go by
    composite t_B desc."""
    out = []
    for n in sorted({r["plants"] for r in rows}, reverse=True):
        left = [r for r in rows if r["plants"] == n]
        while left:
            t_best = min(r["time_s"] for r in left)
            tied = [r for r in left if r["time_s"] <= T_TIE * t_best or t_best == NONE_T]
            tied.sort(key=lambda r: -r["composite_tB"])
            out += tied
            left = [r for r in left if r not in tied]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="SEL1_ARMS_")
    ap.add_argument("--top", type=int, default=0)               # tests only: first N hall entries
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(4)
    rule = json.load(open(RULE_PATH, encoding="utf-8"))
    vm = Q.make_vm()
    F.install()
    S.K = 5
    ctx = W2.build("pcnull", torch.device(a.device), vm, rule["world"]["alpha1"], RULE_PATH, log=log)
    hours = np.concatenate([np.arange(m["m0"], m["m1"]) for m in ctx["months"] if m["half"] == "B"])[::R.HOUR_STEP]
    plants = [S.signal(ctx, vm, tuple(rule["world"]["p1_body"])), S.signal(ctx, vm, tuple(rule["p2"]["body"]))]
    lay = P.Layout(ctx, ("A", "B"))
    rows, missing = [], []
    for label, key in ARMS:
        run = a.prefix + key
        d = os.path.join(S.OUT_ROOT, run)
        if not os.path.exists(os.path.join(d, "TRAIN_RECEIPT_V1.json")):
            missing.append(run)
            continue
        man = C.manifest(d)
        ctx["dd_max"] = float(man["args"]["dd_max"])
        hof = json.load(open(os.path.join(d, "HOF.json"), encoding="utf-8"))["entries"]
        hof = hof[:a.top] if a.top else hof
        first, hits, n_good = scan(hof, ctx, vm, plants, hours)
        pool_tb = None
        if key == "POOL":
            pm = json.load(open(os.path.join(d, "POOL.json"), encoding="utf-8"))["members"]
            pf, ph, _ = scan(pm, ctx, vm, plants, hours)
            first = [min(x, y) for x, y in zip(first, pf)]
            hits = [x + y for x, y in zip(hits, ph)]
            mem = [dict(z=lay.rankz(P.masked_signal(ctx, vm, tuple(m["body"]))).float(), sign=1.0) for m in pm]
            bk = C.book(mem, ctx, lay) if mem else None
            pool_tb = None if bk is None else bk["t_B"]
        chosen, _, _ = C.members(hof, ctx, vm, lay, 0)
        bk = C.book(chosen, ctx, lay) if chosen else None
        t_b = float(bk["t_B"]) if bk is not None and np.isfinite(bk["t_B"]) else -9999.0
        got = [f for f in first if f < NONE_T]
        rec = json.load(open(os.path.join(d, "TRAIN_RECEIPT_V1.json"), encoding="utf-8"))
        r = dict(arm=label, run=run, p1_found=first[0] < NONE_T, p2_found=first[1] < NONE_T, plants=len(got),
                 time_s=max(got) if got else NONE_T, first_s=first, composite_tB=t_b,
                 composite=None if bk is None else {k: bk[k] for k in ("h", "state", "dir", "t_A", "t_B", "trim_A",
                                                                       "trim_B", "years_against")},
                 composite_members=len(chosen), pool_composite_tB=pool_tb, behav_tA_above_3=n_good,
                 hall=len(hof), n_eval=rec.get("n_eval"), n_behav=rec.get("n_behav"),
                 compute_s=rec.get("compute_seconds"), p1_hits=hits[0][:10], p2_hits=hits[1][:10])
        rows.append(r)
        log("%-16s plants %d (P1 %s P2 %s) time %s composite t_B %.3f pool %s behav t_A>3 %d"
            % (label, r["plants"], r["p1_found"], r["p2_found"], "-" if not got else "%.0fs" % r["time_s"], t_b,
               pool_tb, n_good))
    rep = dict(schema="SEL1_ARMS_RESULT_V1", order="XS_SEL1 supplement 10-15 evening", prefix=a.prefix, top=a.top or None,
               rule=rule, missing=missing, dev_read=False, oos_read=False)
    rc = 7
    if not missing:
        ranked = order(rows)
        for i, r in enumerate(ranked):
            r["rank"] = i + 1
        rows = ranked
        rep.update(ranking=[r["arm"] for r in ranked], winner=ranked[0]["arm"], winner_run=ranked[0]["run"])
        rc = 0 if ranked[0]["run"] == a.prefix + "RL" else 10
        log("ranking: %s -> winner %s" % (" > ".join(r["arm"] for r in ranked), ranked[0]["arm"]))
    else:
        log("missing arms: %s" % missing)
    rep.update(arms=rows, seconds=round(time.time() - t0, 1), written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(rep, open(os.path.join(S.OUT_ROOT, "%sRESULT%s.json" % (a.prefix, "_TEST" if a.top else "")), "w",
                        encoding="utf-8"), indent=1, default=lambda x: None if x == NONE_T else float(x), ensure_ascii=False)
    sys.exit(rc)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
