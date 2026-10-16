"""XS_SEL1 补充 10-15 夜 / 深夜: reward rule (DD_MAX) and the plant check in the NULL world (check only, no training;
V4 TRAIN only, no DEV, no OOS). New file.

World    NULL (sel1_common null: per-coin circular roll >= 720 h, NULL_SEED); plant = the trainer's pcnull world:
         fwd[:, t] += alpha zp[:, t-1], zp = the plant formula's TOP100 rank z as the NULL world sees it
         (S.plant_signal_shifted). pcnull(alpha 0) must equal null bit for bit (checked).
Pool     sel1_calib's (CR1_PC_P8 HOF top 50 + B5_DBG_REAL HOF_RANKED, dedup, plant body removed), bodies invalid in
         the NULL world dropped; + the plant body for the check.
Reward   S.reward t_dd: t = mean / (sd ddof 1 / sqrt n) of the 30 A-month cost0 log NAVs (a month holding nothing
         is 0 and counts); 0 when the A-half max drawdown (cost0 log NAV marked at month ends from 0) > DD_MAX[K].
Stages
  ddmax  NULL world, pool x 60 combos, A-half max drawdown per K in {5,10,20}; DD_MAX[K] = q95 (numpy linear).
         -> SEL1_REWARD_RULE.json (written once, never rewritten).
  pipe   alpha 0, K 5, one cell through the full check code; mechanics only -> SEL1_NULLPLANT_PIPE.json.
  grid   SEL1_NULLPLANT_RULE.json first; strength {20, 40} bp/day x K {5, 10, 20} -> SEL1_NULLPLANT.json.
Strength alpha by one linear step as sel1_calib: gain = A-half cost0 bp/day of the plant book (plant formula, H72,
         all, long, K 5 = gross 1) in pcnull(alpha) minus in null; |gain - target| <= 10 %; one alpha per strength
         for every K (sel1_calib / sel1_power convention).
Cell     every pool body and the plant take the combo of 60 with the highest A-half t_dd at that K (ties: first in
         sel1_calib.COMBOS order); B-half t (same t, B months) of that combo.
         visible  plant B t >= 2;  rankable  plant B t rank <= 10 among pool + plant.  passed = both.
         The fixed plant book 72/all/long is reported too (B t and its rank), not judged.
Choice   passing cells: smallest strength, then smallest K. None -> hand back (rc 7), nothing re-tuned.
    python -I sel1_nullplant.py ddmax|pipe|grid
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import sel1_common as S
import sel1_calib as C
import cr1_common as Q
import m2h_dense_common as D

REWARD_RULE = os.path.join(S.OUT_ROOT, "SEL1_REWARD_RULE.json")
GRID_RULE = os.path.join(S.OUT_ROOT, "SEL1_NULLPLANT_RULE.json")
OUT = dict(pipe=os.path.join(S.OUT_ROOT, "SEL1_NULLPLANT_PIPE.json"), grid=os.path.join(S.OUT_ROOT, "SEL1_NULLPLANT.json"))
KS = (5, 10, 20)
STRENGTHS = (20.0, 40.0)
ALPHA0 = 1e-4
DD_Q = 0.95
WORKERS = 20
RULE = dict(
    schema="SEL1_NULLPLANT_RULE_V1",
    world="NULL (per-coin circular roll, NULL_SEED) + plant fwd[:, t] += alpha zp[:, t-1], zp = plant formula TOP100 "
          "rank z in the NULL world (trainer --mode pcnull)",
    grid=dict(strength_bpd=list(STRENGTHS), K=list(KS), t="monthly"),
    strength="alpha by one linear step from 1e-4: A-half cost0 bp/day gain of the plant book (plant formula, H72, all, "
             "long, K 5) pcnull vs null, within 10 %; the same alpha at every K",
    combo="every body takes the combo of 60 with the highest A-half t_dd (SEL1_REWARD_RULE DD_MAX[K]); ties first in order",
    t="mean / (sd ddof 1 / sqrt n) of monthly cost0 log NAV, months holding nothing count as 0",
    visible="plant B-half t (its t_dd combo) >= 2",
    rankable="plant B-half t rank <= 10 among the NULL-valid sel1_calib pool + plant",
    fixed="plant book 72/all/long B-half t and rank reported, not judged",
    select="cells passing both: smallest strength, then smallest K; PC uses that strength and K, NULL / REAL that K",
    none_pass="no cell passes -> hand back, nothing re-tuned",
)


def log(m):
    print(m, flush=True)


def t_of(rows):
    return S.reward(rows, "t")


def cut(sels, k):
    return [(hi[:, :k], lo[:, :k], n) for hi, lo, n in sels]


def write_once(path, rule):
    if os.path.exists(path):
        old = json.load(open(path, encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "written"} != rule:
            raise SystemExit("%s exists with a different rule" % path)
        return
    json.dump(dict(rule, written=time.strftime("%Y-%m-%dT%H:%M:%S%z")), open(path, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)


# --------------------------------------------------------------------------- score workers
def dd_task(task):
    """(key, sels_A at 20) -> {K: [A-half max drawdown per combo]}."""
    key, sa = task
    v, out = S._WS["view"], {}
    for k in KS:
        S.K = k
        a = cut(sa, k)
        out[k] = [S.max_dd(S.score_sel(a, v, h, st, dr)) for h, st, dr in C.COMBOS]
    return key, out


def cell_task(task):
    """(key, ks, dd_max {K: line}, sels_A, sels_B at 20, fixed) -> {K: t_dd combo, its A score and B t}."""
    key, ks, dd, sa, sb, fixed = task
    v, out = S._WS["view"], {}
    for k in ks:
        S.K = k
        a, b = cut(sa, k), cut(sb, k)
        best = None
        for c in C.COMBOS:
            ra = S.score_sel(a, v, *c)
            s = S.reward(ra, "t_dd", dd[k])
            if np.isfinite(s) and (best is None or s > best[0]):
                best = (s, c, t_of(ra), S.max_dd(ra))
        if best is None:
            out[k] = None
            continue
        s, c, ta, da = best
        rb = S.score_sel(b, v, *c, halves=("B",))
        r = dict(score_A=s, t_A=ta, dd_A=da, h=c[0], state=S.STATES[c[1]], dir=S.DIRS[c[2]], t_B=t_of(rb),
                 B_cost0_bpd=S.bpd(rb))
        if fixed:
            rf = S.score_sel(b, v, *C.PLANT_COMBO, halves=("B",))
            r["fixed"] = dict(t_B=t_of(rf), B_cost0_bpd=S.bpd(rf))
        out[k] = r
    return key, out


def pool_run(fn, tasks, world, name):
    cdir = S.write_cache(world, os.path.join(S.OUT_ROOT, "_nullplant_cache_" + name))
    with mp.get_context("spawn").Pool(WORKERS, initializer=S.init_score_worker, initargs=(cdir,)) as pool:
        return dict(pool.imap_unordered(fn, tasks))


# --------------------------------------------------------------------------- worlds and selections
def setup(vm, device):
    """NULL world (keep_base: zp as pcnull builds it), pool + plant selections at K 20 (independent of fwd)."""
    null = S.build("null", device, vm, log=log, keep_base=True)
    pb = tuple(D.plant_body(vm))
    bodies = [tuple(e["body"]) for e in json.load(open(C.CR1_HOF, encoding="utf-8"))["entries"][:50]]
    for fam in json.load(open(C.B5_HOF, encoding="utf-8"))["families"].values():
        bodies += [tuple(e["body"]) for e in fam]
    bodies = [b for b in dict.fromkeys(bodies) if b != pb]
    S.K = max(KS)
    sels, bad = {}, {}
    for b in bodies + [pb]:
        try:
            sg = S.signal(null, vm, b)
        except Exception as e:                                      # invalid in the NULL world: not in the pool
            bad[vm.describe(list(b), list(S.M.TERMINALS))] = str(e)[:80]
            continue
        sels[b] = (S.select_months(sg, null, ("A",)), S.select_months(sg, null, ("B",)))
    if pb not in sels:
        raise SystemExit("plant body invalid in the NULL world")
    log("pool %d of %d bodies valid in NULL (+ plant)" % (len(sels) - 1, len(bodies)))
    return null, pb, sels, bad


def world(null, alpha):
    """pcnull(alpha) from the NULL world: the trainer's build('pcnull', alpha) shard."""
    w = dict(null)
    w["shard"] = S.no_winsor_shard(null["base"], null["dead"], fwd=null["base"]["fwd"] + float(alpha) * S.plant_add(null["zp"]))
    w["mode"], w["alpha"] = "pcnull", float(alpha)
    return w


def plant_book_bpd(sels_a, w):
    S.K = 5
    return S.bpd(S.score_sel(cut(sels_a, 5), w, *C.PLANT_COMBO))


def cells(vm, w, name, pb, sels, dd, ks):
    tasks = [(b, ks, dd, sa, sb, b == pb) for b, (sa, sb) in sels.items()]
    res = pool_run(cell_task, tasks, w, name)
    out = []
    for k in ks:
        rows = [(b, r[k]) for b, r in res.items() if r[k] is not None]
        key = lambda x: -(x[1]["t_B"] if np.isfinite(x[1]["t_B"]) else -1e9)
        rs = sorted(rows, key=key)
        rank = next(i + 1 for i, (b, _) in enumerate(rs) if b == pb)
        pr = rs[rank - 1][1]
        others = [r["t_B"] for b, r in rs if b != pb and np.isfinite(r["t_B"])]
        ft = pr["fixed"]["t_B"]
        row = dict(K=k, plant=pr, plant_rank=rank, pool=len(rs) - 1, visible=bool(pr["t_B"] >= C.T_MIN),
                   rankable=bool(rank <= C.RANK_MAX),
                   fixed=dict(pr["fixed"], rank=1 + sum(1 for t in others if t > ft) if np.isfinite(ft) else None),
                   top3=[dict(expr=vm.describe(list(b), list(S.M.TERMINALS)), t_B=r["t_B"], h=r["h"],
                              state=r["state"], dir=r["dir"]) for b, r in rs[:3]])
        row["passed"] = row["visible"] and row["rankable"]
        out.append(row)
    return out


# --------------------------------------------------------------------------- stages
def stage_ddmax(vm, device):
    t0 = time.time()
    null, pb, sels, bad = setup(vm, device)
    tasks = [(b, sa) for b, (sa, _) in sels.items() if b != pb]
    res = pool_run(dd_task, tasks, null, "ddmax")
    dd = {k: np.asarray([v for r in res.values() for v in r[k]]) for k in KS}
    line = {k: float(np.quantile(dd[k], DD_Q)) for k in KS}
    rule = dict(schema="SEL1_REWARD_RULE_V1", reward="t_dd", trainer_flags="--reward t_dd --dd-max DD_MAX[K] --k K",
                t="A half, 30 odd months: mean / (sd ddof 1 / sqrt 30) of the monthly cost0 beta-hedged log NAV; "
                  "a month holding nothing is 0 and counts; sd 0 -> 0 (S.reward 't')",
                dd="A-half max drawdown: the 30 A months' cost0 log NAV joined in order, cumulative from 0, marked at "
                   "month ends, largest fall from the running peak (S.max_dd)",
                t_dd="0 when dd > DD_MAX[K], else t (S.reward 't_dd')",
                dd_max={str(k): line[k] for k in KS},
                calibration=dict(world="NULL (no plant), NULL_SEED %d" % S.NULL_SEED, pool="sel1_calib's pool, NULL-valid",
                                 bodies=len(tasks), combos=len(C.COMBOS), quantile=DD_Q, method="numpy linear",
                                 invalid_in_null=bad,
                                 per_k={str(k): dict(n=int(dd[k].size), zeros=int((dd[k] == 0).sum()),
                                                     median=float(np.median(dd[k])), q95=line[k]) for k in KS}),
                combo_rule="every check and the trainer's combo token use the same t_dd score",
                batch_norm="trainer in-batch standardisation (D.zfam) unchanged")
    write_once(REWARD_RULE, rule)
    log("DD_MAX %s from %d bodies x %d combos (%.0fs)" % ({k: round(v, 4) for k, v in line.items()}, len(tasks),
                                                          len(C.COMBOS), time.time() - t0))


def load_dd():
    r = json.load(open(REWARD_RULE, encoding="utf-8"))
    return {k: float(r["dd_max"][str(k)]) for k in KS}


def stage_pipe(vm, device):
    t0 = time.time()
    if os.path.exists(OUT["pipe"]):
        raise SystemExit("%s exists" % OUT["pipe"])
    dd = load_dd()
    tb = S.build("pcnull", device, vm, alpha=0.0, log=log)       # the trainer's own pcnull world at alpha 0
    tb = dict(zp=tb["zp"], **{k: tb["shard"][k] for k in ("fwd_wx", "fi", "fr")})
    torch.cuda.empty_cache()
    null, pb, sels, bad = setup(vm, device)
    w0 = world(null, 0.0)
    eq = {k: bool(np.array_equal(w0["shard"][k], null["shard"][k], equal_nan=True)
                  and np.array_equal(tb[k], null["shard"][k], equal_nan=True)) for k in ("fwd_wx", "fi", "fr")}
    eq["zp"] = bool(np.array_equal(tb["zp"], null["zp"], equal_nan=True))
    del tb
    g = plant_book_bpd(sels[pb][0], w0) - plant_book_bpd(sels[pb][0], null)
    rows = cells(vm, w0, "pipe", pb, sels, dd, (5,))
    ok = all(eq.values()) and g == 0.0 and rows[0]["pool"] > 0
    json.dump(dict(schema="SEL1_NULLPLANT_PIPE_V1", alpha=0.0, K=5, pcnull0_eq_null=eq, gain_alpha0=g,
                   pool=rows[0]["pool"], invalid_in_null=bad, cell=rows[0], passed=ok, note="pipeline only, not read",
                   seconds=round(time.time() - t0, 1), dev_read=False, oos_read=False),
              open(OUT["pipe"], "w", encoding="utf-8"), indent=1, default=float, ensure_ascii=False)
    log("pipe alpha 0 K 5: pcnull0==null %s gain0 %s pool %d -> mechanics ok %s (%.0fs)"
        % (all(eq.values()), g == 0.0, rows[0]["pool"], ok, time.time() - t0))
    sys.exit(0 if ok else 7)


def stage_grid(vm, device):
    t0 = time.time()
    if os.path.exists(OUT["grid"]):
        raise SystemExit("%s exists" % OUT["grid"])
    dd = load_dd()
    write_once(GRID_RULE, dict(RULE, dd_max={str(k): dd[k] for k in KS}))
    null, pb, sels, bad = setup(vm, device)
    base = plant_book_bpd(sels[pb][0], null)
    g0 = plant_book_bpd(sels[pb][0], world(null, ALPHA0)) - base
    table, alphas = [], {}
    for target in STRENGTHS:
        alpha = ALPHA0 * target / g0
        w = world(null, alpha)
        g1 = plant_book_bpd(sels[pb][0], w) - base
        alphas[target] = dict(alpha=alpha, gain_bpd=g1, ok=bool(abs(g1 - target) <= 0.1 * target))
        log("strength %.0f: alpha %.3e gain %.2f ok %s" % (target, alpha, g1, alphas[target]["ok"]))
        for row in cells(vm, w, "s%d" % int(target), pb, sels, dd, KS):
            row.update(strength_bpd=target, alpha=alpha, alpha_ok=alphas[target]["ok"])
            row["passed"] = row["passed"] and row["alpha_ok"]
            table.append(row)
            log("s %2.0f K %2d plant t_B %5.2f rank %3d/%d fixed t_B %5.2f -> %s" % (
                target, row["K"], row["plant"]["t_B"], row["plant_rank"], row["pool"], row["fixed"]["t_B"], row["passed"]))
    ok = [r for r in table if r["passed"]]
    chosen = min(ok, key=lambda r: (r["strength_bpd"], r["K"])) if ok else None
    rep = dict(schema="SEL1_NULLPLANT_V1", rule=RULE, dd_max=dd, plant_book_null_bpd=base, alpha0=ALPHA0, gain0_bpd=g0,
               alphas=alphas, invalid_in_null=bad, table=table, chosen=chosen,
               verdict=("chosen strength %.0f bp/day K %d" % (chosen["strength_bpd"], chosen["K"])) if chosen
               else "no cell passes -> hand back",
               dev_read=False, oos_read=False, seconds=round(time.time() - t0, 1),
               written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(rep, open(OUT["grid"], "w", encoding="utf-8"), indent=1, default=float, ensure_ascii=False)
    log("nullplant -> %s (%.0fs)" % (rep["verdict"], time.time() - t0))
    sys.exit(0 if chosen else 7)


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage not in ("ddmax", "pipe", "grid"):
        raise SystemExit(__doc__)
    device = torch.device("cuda")
    vm = Q.make_vm()
    dict(ddmax=stage_ddmax, pipe=stage_pipe, grid=stage_grid)[stage](vm, device)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
