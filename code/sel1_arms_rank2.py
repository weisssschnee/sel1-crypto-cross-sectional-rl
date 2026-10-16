"""XS_SEL1 ARMS merged ranking (supplement 10-16 noon): ARMS v1's V2 RL / DEDUP / FROZEN / POOL rows (taken from
SEL1_ARMS_RESULT.json as sel1_arms_rank.py wrote them; same world, same code) + the three ARMS v2 arms (PG-risk,
GFN-fix, GA-fix) scored here with sel1_arms_rank's own scan / members / book / order, imported, not modified.
v1's GFN and GA rows are superseded by their fixes and are listed as report-only ("v1 implementation only").
TRAIN, ARMS world; no DEV, no OOS. New file.
    python -I sel1_arms_rank2.py [--top N] [--device cuda]
-> SEL1_ARMS2_RESULT.json; rc 0 V2 RL first, 10 another arm first, 7 an arm missing.
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
import sel1_combo as C
import sel1_pool as P
import sel1_world2 as W2
import sel1_arms_rank as AR
import cr1_common as Q

V1_KEEP = ("SEL1_ARMS_RL", "SEL1_ARMS_DEDUP", "SEL1_ARMS_FROZEN", "SEL1_ARMS_POOL")
V1_REPORT = ("SEL1_ARMS_GFN", "SEL1_ARMS_GA")
V2_ARMS = (("PG-risk (--risk-seek 0.05 --pq 64 --ent-gamma 1.0)", "SEL1_ARMS2_PGRISK"),
           ("GFN-fix (--gfn2 1 --explore 0.05 --pq 64)", "SEL1_ARMS2_GFN"),
           ("GA-fix (sel1_ga2)", "SEL1_ARMS2_GA"))
NOTE = "v1 RL / GFN / GA rows only represent that version of the implementation (supplement 10-16 noon)"


def log(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(4)
    v1p = os.path.join(S.OUT_ROOT, "SEL1_ARMS_RESULT.json")
    v1 = json.load(open(v1p, encoding="utf-8")) if os.path.exists(v1p) else dict(arms=[])
    by_run = {r["run"]: r for r in v1.get("arms", [])}
    rows, missing = [], [r for r in V1_KEEP if r not in by_run]
    for r in V1_KEEP:
        if r in by_run:
            x = dict(by_run[r], source="ARMS v1 result")
            x.pop("rank", None)
            if x.get("time_s") is None:                         # the v1 json wrote inf (not recovered) as null
                x["time_s"] = AR.NONE_T
            if r == "SEL1_ARMS_RL":
                x["note"] = NOTE
            rows.append(x)
    rule = json.load(open(AR.RULE_PATH, encoding="utf-8"))
    vm = Q.make_vm()
    F.install()
    S.K = 5
    ctx = W2.build("pcnull", torch.device(a.device), vm, rule["world"]["alpha1"], AR.RULE_PATH, log=log)
    hours = np.concatenate([np.arange(m["m0"], m["m1"]) for m in ctx["months"] if m["half"] == "B"])[::AR.R.HOUR_STEP]
    plants = [S.signal(ctx, vm, tuple(rule["world"]["p1_body"])), S.signal(ctx, vm, tuple(rule["p2"]["body"]))]
    lay = P.Layout(ctx, ("A", "B"))
    for label, run in V2_ARMS:
        d = os.path.join(S.OUT_ROOT, run)
        if not os.path.exists(os.path.join(d, "TRAIN_RECEIPT_V1.json")):
            missing.append(run)
            continue
        man = C.manifest(d)
        ctx["dd_max"] = float(man["args"]["dd_max"])
        hof = json.load(open(os.path.join(d, "HOF.json"), encoding="utf-8"))["entries"]
        hof = hof[:a.top] if a.top else hof
        first, hits, n_good = AR.scan(hof, ctx, vm, plants, hours)
        chosen, _, _ = C.members(hof, ctx, vm, lay, 0)
        bk = C.book(chosen, ctx, lay) if chosen else None
        t_b = float(bk["t_B"]) if bk is not None and np.isfinite(bk["t_B"]) else -9999.0
        got = [f for f in first if f < AR.NONE_T]
        rec = json.load(open(os.path.join(d, "TRAIN_RECEIPT_V1.json"), encoding="utf-8"))
        r = dict(arm=label, run=run, source="ARMS v2", p1_found=first[0] < AR.NONE_T, p2_found=first[1] < AR.NONE_T,
                 plants=len(got), time_s=max(got) if got else AR.NONE_T, first_s=first, composite_tB=t_b,
                 composite=None if bk is None else {k: bk[k] for k in ("h", "state", "dir", "t_A", "t_B", "trim_A",
                                                                       "trim_B", "years_against")},
                 composite_members=len(chosen), pool_composite_tB=None, behav_tA_above_3=n_good,
                 hall=len(hof), n_eval=rec.get("n_eval"), n_behav=rec.get("n_behav"),
                 compute_s=rec.get("compute_seconds"), p1_hits=hits[0][:10], p2_hits=hits[1][:10])
        rows.append(r)
        log("%-50s plants %d (P1 %s P2 %s) time %s composite t_B %.3f behav t_A>3 %d"
            % (label, r["plants"], r["p1_found"], r["p2_found"], "-" if not got else "%.0fs" % r["time_s"], t_b, n_good))
    report = [dict(by_run[r], note=NOTE, ranked=False) for r in V1_REPORT if r in by_run]
    rep = dict(schema="SEL1_ARMS_RESULT_V1", variant="ARMS2 merged", order="XS_SEL1 supplement 10-16 noon",
               prefix="SEL1_ARMS", top=a.top or None, rule=rule, missing=missing, note=NOTE, report_only=report,
               v1_result=v1p, dev_read=False, oos_read=False)
    rc = 7
    if not missing:
        ranked = AR.order(rows)
        for i, r in enumerate(ranked):
            r["rank"] = i + 1
        rows = ranked
        rep.update(ranking=[r["arm"] for r in ranked], winner=ranked[0]["arm"], winner_run=ranked[0]["run"])
        rc = 0 if ranked[0]["run"] == "SEL1_ARMS_RL" else 10
        log("ranking: %s -> winner %s" % (" > ".join(r["arm"] for r in ranked), ranked[0]["arm"]))
    else:
        log("missing arms: %s" % missing)
    rep.update(arms=rows, seconds=round(time.time() - t0, 1), written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(rep, open(os.path.join(S.OUT_ROOT, "SEL1_ARMS2_RESULT%s.json" % ("_TEST" if a.top else "")), "w",
                        encoding="utf-8"), indent=1, default=lambda x: None if x == AR.NONE_T else float(x), ensure_ascii=False)
    sys.exit(rc)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
