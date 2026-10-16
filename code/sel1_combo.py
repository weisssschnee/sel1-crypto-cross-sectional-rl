"""XS_SEL1 composite book from a finished hall (supplement 10-15 afternoon item 1). TRAIN (V4 grid) only; no DEV, no
OOS. New file; sel1_common / sel1_sweep / sel1_round / sel1_pool are imported, not modified. Rule: SEL1_COMBO_RULE.json.

  members    the run's HOF.json in its order (A-half t_dd, high to low), re-scored in the run's own world with its own
             combo on the A months; greedy: a candidate joins when its A monthly cost0 corr (sel1_pool.month_corr, nan =
             no evidence = allowed) is <= 0.7 with every member chosen so far; at most 20; t_A 0 / invalid skipped.
  composite  per signal hour: sum over members of sign(t_A_j) * rank z_j (sel1_pool.Layout over the A + B months).
  book       K, beta hedge as the chain; combo = the best of 60 by A-half t_dd (sel1_sweep.sweep, the trainer's rule);
             then A + B once with that combo (S.score_sel, keep) -> t_A, t_B, trimmed, years.
  modes      pc    -> COMBO_PC.json: members' |corr| with the plant (sel1_round's B-hour Spearman). Self-check:
                      a member with corr > 0.7 and composite t_B >= 2 (rc 0, else 30).
             null  -> COMBO_NULL.json: T_combo_null = composite t_B of the NULL hall.
             real  -> COMBO_REAL.json: gates g1 t_B > max(2, T_combo_null); g2 t_A > 2, same sign; g3 trimmed A and B
                      > 0; g4 at most one opposite-sign year. rc 0 pass, 32 not.
    python -I sel1_combo.py RUN --mode pc|null|real [--null RUN] [--device cpu|cuda] [--top N]
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
import sel1_sweep as W
import sel1_round as R
import sel1_pool as P
import cr1_common as Q
import m2h_common as M
import m2h_dense_common as D
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid

N_MAX, RHO_MAX, T_FLOOR = 20, 0.7, 2.0


def log(m):
    print(m, flush=True)


def manifest(d):
    mf = os.path.join(d, "TRAIN_MANIFEST.json")
    if not os.path.exists(mf):
        mf = os.path.join(d, sorted(f for f in os.listdir(d) if f.startswith("RESUME_MANIFEST_"))[-1])
    return json.load(open(mf, encoding="utf-8"))


def members(hof, ctx, vm, lay, top, plant=None, hours=None):
    """Greedy pick -> (chosen list of dict, considered count, max |hall score - re-scored score|)."""
    chosen, seen, d_max = [], 0, 0.0
    for e in hof[:top] if top else hof:
        if len(chosen) >= N_MAX:
            break
        seen += 1
        b = tuple(e["body"])
        try:
            s = P.masked_signal(ctx, vm, b)
        except (S.Invalid, CandidateLocalInvalid, M.ZeroCoverage):
            continue
        ra = S.score_sel(P.select_signal(ctx, s, ("A",)), ctx, e["h"], e["st"], e["dr"])
        c0 = [float(r["cost0"]) for r in ra]
        if e.get("score") is not None:
            d_max = max(d_max, abs(float(S.reward(ra, "t_dd", ctx["dd_max"])) - float(e["score"])))
        t_a = float(S.reward(ra, "t"))
        if not np.isfinite(t_a) or t_a == 0.0:
            continue
        rho = [P.month_corr(c0, m["cost0"]) for m in chosen]
        if any(np.isfinite(r) and r > RHO_MAX for r in rho):
            continue
        m = dict(expr=e["expr"], body=list(b), h=e["h"], state=S.STATES[e["st"]], dir=S.DIRS[e["dr"]],
                 score_A_hof=e.get("score"), t_A=t_a, sign=float(np.sign(t_a)), cost0=c0,
                 rho_max=max([r for r in rho if np.isfinite(r)], default=None), z=lay.rankz(s).float())
        if plant is not None:
            sg = s.cpu().numpy()
            m["corr"] = abs(R.spearman_rows(sg[:, hours].T, plant[:, hours].T))
        chosen.append(m)
        del s
    return chosen, seen, d_max


def book(chosen, ctx, lay):
    """Composite -> dict(combo, t_A, t_B, ...) or None (no finite combo)."""
    comp = lay.composite([m["z"] for m in chosen], [m["sign"] for m in chosen])
    sels = lay.select(comp)
    sa = [x for x, m in zip(sels, lay.months) if m["half"] == "A"]
    sw = W.sweep(sa, ctx, "t_dd", ctx["dd_max"])
    if sw is None:
        return None
    c = sw[0]
    rows = S.score_sel(sels, ctx, *c, halves=("A", "B"), keep=True)
    ra, rb = [r for r in rows if r["half"] == "A"], [r for r in rows if r["half"] == "B"]
    t_b = float(S.reward(rb, "t"))
    yrs, against = R.years_against([dict(year=x["year"], cost0=x["cost0"], n_pos=x["n_pos"]) for x in rows], np.sign(t_b))
    return dict(h=c[0], state=S.STATES[c[1]], dir=S.DIRS[c[2]], score_A=float(sw[2]), t_A=float(S.reward(ra, "t")),
                t_B=t_b, bpd_A=S.bpd(ra), bpd_B=S.bpd(rb), net_bpd_B=S.bpd(rb, "net"), trim_A=R.trimmed(ra),
                trim_B=R.trimmed(rb), years=yrs, years_against=against, B_months=[x["cost0"] for x in rb])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--mode", required=True, choices=("pc", "null", "real"))
    ap.add_argument("--null")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--top", type=int, default=0)                # tests only: first N hall entries
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(4)
    sfx = "_TEST" if a.top else ""
    d = os.path.join(S.OUT_ROOT, a.run)
    man = manifest(d)
    if not os.path.exists(os.path.join(d, "TRAIN_RECEIPT_V1.json")):
        raise SystemExit("%s has no receipt" % a.run)
    S.K = int(man["world"].get("k", 5))
    vm = Q.make_vm()
    F.install()
    w = man["world"]
    if w.get("plant2"):
        import sel1_world2 as W2
        sp = os.path.join(d, "_plant2.json")
        json.dump(dict(body=w["plant2"]["body"], alpha=w["plant2"]["alpha"]), open(sp, "w"))
        ctx = W2.build(w["mode"], torch.device(a.device), vm, w["alpha"], sp, log=log)
    else:
        ctx = S.build(w["mode"], torch.device(a.device), vm, alpha=w["alpha"], log=log)
    ctx["dd_max"] = float(man["args"]["dd_max"])
    lay = P.Layout(ctx, ("A", "B"))
    hof = json.load(open(os.path.join(d, "HOF.json"), encoding="utf-8"))["entries"]
    plant = hours = None
    if a.mode == "pc":
        hours = np.concatenate([np.arange(m["m0"], m["m1"]) for m in ctx["months"] if m["half"] == "B"])[::R.HOUR_STEP]
        plant = S.signal(ctx, vm, tuple(D.plant_body(vm)))
    chosen, seen, d_max = members(hof, ctx, vm, lay, a.top, plant, hours)
    bk = book(chosen, ctx, lay) if chosen else None
    lite = [dict((k, v) for k, v in m.items() if k not in ("z", "cost0", "body")) for m in chosen]
    rep = dict(schema="SEL1_COMBO_V1", run=a.run, mode=a.mode, top=a.top or None, world=w, device=a.device,
               hall=len(hof), considered=seen, members=lite, member_bodies=[m["body"] for m in chosen],
               rescore_max_abs_diff=d_max, book=bk, dev_read=False, oos_read=False)
    rc = 0
    if a.mode == "pc":
        hit = [m["expr"] for m in chosen if m.get("corr", 0) > R.CORR_MIN]
        ok = bool(hit) and bk is not None and bk["t_B"] >= T_FLOOR
        rep.update(plant_members=hit, selfcheck_passed=ok)
        rc = 0 if ok else 30
        log("combo pc %s: %d members (of %d), plant members %d, composite %s t_A %s t_B %s -> self-check %s"
            % (a.run, len(chosen), seen, len(hit), bk and "%s/%s/%s" % (bk["h"], bk["state"], bk["dir"]),
               bk and round(bk["t_A"], 3), bk and round(bk["t_B"], 3), ok))
    elif a.mode == "null":
        rep["T_combo_null"] = bk["t_B"] if bk is not None else None
        log("combo null %s: %d members, T_combo_null %s" % (a.run, len(chosen), rep["T_combo_null"]))
    else:
        nl = json.load(open(os.path.join(S.OUT_ROOT, a.null, "COMBO_NULL%s.json" % sfx), encoding="utf-8"))
        tn = nl["T_combo_null"]
        thr = max(T_FLOOR, tn if tn is not None and np.isfinite(tn) else T_FLOOR)
        if bk is None:
            g = dict(g1=False, g2=False, g3=False, g4=False)
        else:
            g = dict(g1=bool(bk["t_B"] > thr), g2=bool(bk["t_A"] > R.T_A_MIN and np.sign(bk["t_A"]) == np.sign(bk["t_B"])),
                     g3=bool(bk["trim_A"] > 0 and bk["trim_B"] > 0), g4=bool(bk["years_against"] <= 1))
        rep.update(T_combo_null=tn, threshold=thr, null_run=a.null, gates=g, passed=all(g.values()))
        rc = 0 if rep["passed"] else 32
        log("combo real %s: %d members, t_B %s vs threshold %.3f, gates %s -> %s"
            % (a.run, len(chosen), bk and round(bk["t_B"], 3), thr, g, rep["passed"]))
    rep["seconds"] = round(time.time() - t0, 1)
    rep["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(rep, open(os.path.join(d, "COMBO_%s%s.json" % (a.mode.upper(), sfx)), "w", encoding="utf-8"), indent=1,
              default=float, ensure_ascii=False)
    sys.exit(rc)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
