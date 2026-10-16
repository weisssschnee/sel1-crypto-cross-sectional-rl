"""XS_SEL1 round (order XS_SEL1 s2 / s4, supplement 10-15 late night): the B half of a finished run. TRAIN (V4 grid)
only; no DEV, no OOS. New file.

  candidates  every entry of the run's HOF.json (top 500 by its A-half t_dd score), re-scored in the run's own world
              (TRAIN_MANIFEST: mode, alpha, K) on all 60 months with its own combo.
              t_A / t_B = S.reward 't' of the A / B months; corr = mean over B hours (every 24th) of the hourly
              cross-sectional Spearman correlation of the candidate's signal with the plant formula's (|corr|).
  pc          -> ROUND_PC.json; found_corr = any candidate with corr > 0.7. rc 0 found, 20 not.
  null        -> ROUND_NULL.json; T_null = max t_B over the NULL hall. PC verdict: a PC candidate with corr > 0.7 and
              t_B > T_null. rc 0 found, 21 not (chain stops, PC NOT FOUND).
  real        -> ROUND_REAL.json. Gates (order s2): 1 t_B > T_null; 2 t_A > 2 (same sign); 3 both halves cost0 > 0
              after the most favourable 1 % spells (S.trimmed_ok); 4 at most one year with data of the opposite sign
              (A + B months of the year summed). Dedup by behaviour: B monthly cost0 corr > 0.9 keeps the higher
              t_B. Passing list ordered by t_B; FROZEN_SEL1 (<= 5) and DEV are later steps. rc 0 any, 22 none.
    python -I sel1_round.py RUN --mode pc|null|real [--pc RUN] [--null RUN]
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
import pandas as pd
import torch
import sel1_common as S
import sel1_fastts as F
import cr1_common as Q
import m2h_common as M
import m2h_dense_common as D
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid

CORR_MIN, DEDUP_MAX, T_A_MIN, HOUR_STEP = 0.7, 0.9, 2.0, 24


def log(m):
    print(m, flush=True)


def spearman_rows(x, y):
    """Mean over rows of the Spearman correlation of x and y on the columns finite in both (>= 20)."""
    out = []
    for a, b in zip(x, y):
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < S.MIN_COINS:
            continue
        ra, rb = pd.Series(a[m]).rank().to_numpy(), pd.Series(b[m]).rank().to_numpy()
        ra, rb = ra - ra.mean(), rb - rb.mean()
        d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
        if d > 0:
            out.append(float((ra * rb).sum() / d))
    return float(np.mean(out)) if out else float("nan")


def trimmed(rows, frac=0.01):
    """S.trimmed_ok with months holding nothing (no 'held': cost0 0) counted in days but not in the spells."""
    ev = S.events([r for r in rows if "held" in r])
    if ev.size == 0:
        return float("nan")
    k = max(1, int(np.ceil(frac * ev.size)))
    return 1e4 * (sum(r["cost0"] for r in rows) - np.sort(ev)[::-1][:k].sum()) / sum(r["days"] for r in rows)


def years_against(rows, sign):
    by = {}
    for r in rows:
        if r["n_pos"] > 0:
            by[r["year"]] = by.get(r["year"], 0.0) + r["cost0"]
    return {str(y): v for y, v in sorted(by.items())}, sum(1 for v in by.values() if np.sign(v) == -sign)


def rescore(run, ctx, vm, with_corr, top=0):
    hof = json.load(open(os.path.join(S.OUT_ROOT, run, "HOF.json"), encoding="utf-8"))["entries"]
    hof = hof[:top] if top else hof
    hours = np.concatenate([np.arange(m["m0"], m["m1"]) for m in ctx["months"] if m["half"] == "B"])[::HOUR_STEP]
    sp = S.signal(ctx, vm, tuple(D.plant_body(vm))) if with_corr else None
    sigs, out = {}, []
    for e in hof:
        b = tuple(e["body"])
        if b not in sigs:
            try:
                sigs = {b: S.signal(ctx, vm, b)}                      # one body at a time (memory)
            except (S.Invalid, CandidateLocalInvalid, M.ZeroCoverage) as ex:
                sigs = {b: "INVALID:" + str(ex)}
        sg = sigs[b]
        if isinstance(sg, str):
            out.append(dict(expr=e["expr"], invalid=sg))
            continue
        rows = S.score_sel(S.select_months(sg, ctx, ("A", "B")), ctx, e["h"], e["st"], e["dr"], ("A", "B"), keep=True)
        ra, rb = [r for r in rows if r["half"] == "A"], [r for r in rows if r["half"] == "B"]
        r = dict(expr=e["expr"], body=e["body"], h=e["h"], st=e["st"], dr=e["dr"], state=S.STATES[e["st"]],
                 dir=S.DIRS[e["dr"]], score_A_hof=e.get("score"), t_A=S.reward(ra, "t"), t_B=S.reward(rb, "t"),
                 bpd_A=S.bpd(ra), bpd_B=S.bpd(rb), net_bpd_B=S.bpd(rb, "net"),
                 turn_day_B=sum(x["turn"] for x in rb) / sum(x["days"] for x in rb),
                 trim_A=trimmed(ra), trim_B=trimmed(rb), B_months=[x["cost0"] for x in rb],
                 rows_all=[dict(year=x["year"], cost0=x["cost0"], n_pos=x["n_pos"]) for x in rows])
        if with_corr:
            r["corr"] = abs(spearman_rows(sg[:, hours].T, sp[:, hours].T))
        out.append(r)
    return [r for r in out if "invalid" not in r], [r for r in out if "invalid" in r]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--mode", required=True, choices=("pc", "null", "real"))
    ap.add_argument("--pc")
    ap.add_argument("--null")
    ap.add_argument("--top", type=int, default=0)            # tests only: first N HOF entries
    a = ap.parse_args()
    t0 = time.time()
    sfx = "_TEST" if a.top else ""                             # test rounds read / write ROUND_*_TEST.json
    d = os.path.join(S.OUT_ROOT, a.run)
    mf = os.path.join(d, "TRAIN_MANIFEST.json")
    if not os.path.exists(mf):                                   # a resumed run (speed checks) has resume manifests only
        mf = os.path.join(d, sorted(f for f in os.listdir(d) if f.startswith("RESUME_MANIFEST_"))[-1])
    man = json.load(open(mf, encoding="utf-8"))
    if not os.path.exists(os.path.join(d, "TRAIN_RECEIPT_V1.json")):
        raise SystemExit("%s has no receipt" % a.run)
    S.K = int(man["world"].get("k", 5))                    # manifests before --k ran K 5
    vm = Q.make_vm()
    F.install()
    ctx = S.build(man["world"]["mode"], torch.device("cuda"), vm, alpha=man["world"]["alpha"], log=log)
    ok, bad = rescore(a.run, ctx, vm, a.mode == "pc", a.top)
    ok.sort(key=lambda r: -(r["t_B"] if np.isfinite(r["t_B"]) else -1e9))
    lite = lambda r: {k: v for k, v in r.items() if k not in ("B_months", "rows_all", "body")}
    rep = dict(schema="SEL1_ROUND_V1", run=a.run, mode=a.mode, top=a.top or None, world=man["world"], candidates=len(ok), invalid=len(bad),
               dev_read=False, oos_read=False)
    rc = 0
    if a.mode == "pc":
        hits = [r for r in ok if r.get("corr", 0) > CORR_MIN]
        rep.update(found_corr=bool(hits), corr_hits=[lite(r) for r in hits], top20=[lite(r) for r in ok[:20]])
        rc = 0 if hits else 20
        log("round pc %s: %d candidates, corr > %.1f: %d -> found_corr %s" % (a.run, len(ok), CORR_MIN, len(hits), bool(hits)))
    elif a.mode == "null":
        tn = max(r["t_B"] for r in ok if np.isfinite(r["t_B"]))
        pc = json.load(open(os.path.join(S.OUT_ROOT, a.pc, "ROUND_PC%s.json" % sfx), encoding="utf-8"))
        found = [r for r in pc["corr_hits"] if r["t_B"] > tn]
        rep.update(T_null=tn, pc_run=a.pc, pc_found=bool(found), pc_found_list=found, top20=[lite(r) for r in ok[:20]])
        rc = 0 if found else 21
        log("round null %s: T_null %.3f | PC %s corr hits %d, above T_null %d -> found %s"
            % (a.run, tn, a.pc, len(pc["corr_hits"]), len(found), bool(found)))
    else:
        tn = json.load(open(os.path.join(S.OUT_ROOT, a.null, "ROUND_NULL%s.json" % sfx), encoding="utf-8"))["T_null"]
        passed, kept = [], []
        for r in ok:
            yrs, against = years_against(r["rows_all"], np.sign(r["t_B"]))
            g = dict(g1=bool(r["t_B"] > tn), g2=bool(r["t_A"] > T_A_MIN and np.sign(r["t_A"]) == np.sign(r["t_B"])),
                     g3=bool(r["trim_A"] > 0 and r["trim_B"] > 0), g4=bool(against <= 1))
            r.update(gates=g, years=yrs, years_against=against, passed=all(g.values()))
            if not r["passed"]:
                continue
            v = np.asarray(r["B_months"])
            dup = next((k["expr"] for k in kept if np.corrcoef(v, np.asarray(k["B_months"]))[0, 1] > DEDUP_MAX), None)
            if dup is None:
                kept.append(r)
            else:
                r["dup_of"] = dup
            passed.append(r)
        prog = os.path.join(d, "PROGRESS_64.jsonl")
        last = [json.loads(x) for x in open(prog, encoding="utf-8") if x.strip()][-1] if os.path.exists(prog) else None
        rep.update(T_null=tn, null_run=a.null, gates_passed=len(passed), passing=[lite(r) for r in kept],
                   duplicates=[dict(expr=r["expr"], dup_of=r["dup_of"]) for r in passed if "dup_of" in r],
                   top20=[lite(r) for r in ok[:20]], converged_to=last)
        rc = 0 if kept else 22
        log("round real %s: T_null %.3f | gates passed %d, after dedup %d" % (a.run, tn, len(passed), len(kept)))
    rep["seconds"] = round(time.time() - t0, 1)
    rep["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(rep, open(os.path.join(d, "ROUND_%s%s.json" % (a.mode.upper(), sfx)), "w", encoding="utf-8"), indent=1,
              default=float, ensure_ascii=False)
    sys.exit(rc)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
