"""XS_SEL1 补充 B: power table (check only, no training; V4 TRAIN only, no DEV, no OOS). New file.

Plant strength and plant formula unchanged (SEL1_ALPHA.json, alpha 7.98e-5; sel1_calib's world). Grid: K per side
in {5, 10, 20} x t measure in {monthly, weekly blocks + Newey-West lag 1}. Per cell:
  visible   the plant book (plant formula, H72, all, long; K per side) B-half t under the cell's measure, pc world
  rankable  every pool body (sel1_calib's pool) and the plant take the combo (60) with the best A-half cost0 in the
            pc world at that K; rank of the plant's B-half t (cell measure) among them
  real      the plant book in the real world (no plant), B half: cost0 bp/day and t
  se        SE of the visible book's B-half mean under the measure (bp/day)
The rule (SEL1_POWER_RULE.json) is written before anything is computed and never rewritten.
Selections come from one K=20 stable sort per body; the first K columns are exactly select_month at K (n >= 20 rows only).
    python -I sel1_power.py
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

RULE_JSON = os.path.join(S.OUT_ROOT, "SEL1_POWER_RULE.json")
OUT = os.path.join(S.OUT_ROOT, "SEL1_POWER.json")
KS = (5, 10, 20)
MEASURES = ("monthly", "weekly_nw1")
BLOCK_H = 168
RULE = dict(
    schema="SEL1_POWER_RULE_V1",
    grid=dict(K=list(KS), t=list(MEASURES)),
    plant="SEL1_ALPHA.json alpha and plant body unchanged; visible book = plant formula, H72, state all, long only",
    visible="plant book B-half t (cell measure) >= 2 in the pc world",
    rankable="pool = sel1_calib's (CR1_PC_P8 HOF top 50 + B5_DBG_REAL HOF_RANKED, dedup, V4-invalid dropped) + plant; "
             "each takes its best A-half cost0 combo of 60 at that K; plant B-half t (cell measure) rank <= 10",
    t_monthly="per B month cost0 bp/day; mean / (sd ddof 1 / sqrt n)",
    t_weekly_nw1="each B month cut into consecutive 168 h blocks from its first row, a tail under 84 h joins the previous "
                 "block; per block cost0 bp/day (log1p of hourly cost0 net, as m1_ruler.settle_multi); "
                 "mean / sqrt((g0 + g1) / n), g_k = sum(d_t d_t-k) / n (Newey-West, Bartlett weight, lag 1)",
    select="among cells passing both checks the smallest K; same K: monthly first. T_null = the NULL training run "
           "at the chosen configuration",
    none_pass="no cell passes -> hand back: the data does not carry sparse selection at this plant strength",
    excluded="real-pool B-top TS_RANK(EMA(log_close,72),72) h4 both (B t 4.2) used its B half already: reported, never nominated",
)


def log(m):
    print(m, flush=True)


def hourly(r):
    """Hourly cost0 log1p(net) of one settled window (settle_window(keep=True) row); zeros if it held nothing."""
    E = int(round(r["days"] * 24))
    if "held" not in r:
        return np.zeros(E)
    held, fw, fi, fr = r["held"], r["fw"], r["fi"], r["fr"]
    gross = np.nansum(held * np.nan_to_num(fw), axis=1)
    fund = np.nansum(held * np.where(fi, fr, 0.0), axis=1)
    ln = np.log1p(np.clip(gross - fund, -0.9, None))
    if abs(ln.sum() - r["cost0"]) > 1e-9:
        raise RuntimeError("hourly cost0 != settle_multi cost0")
    return ln


def week_blocks(rows):
    out = []
    for r in rows:
        ln = hourly(r)
        E = ln.size
        edges = list(range(0, E, BLOCK_H))
        if len(edges) > 1 and E - edges[-1] < BLOCK_H // 2:
            edges.pop()
        bounds = edges + [E]
        out += [1e4 * ln[a:b].sum() / ((b - a) / 24.0) for a, b in zip(bounds[:-1], bounds[1:])]
    return np.asarray(out)


def t_nw1(v):
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    d = v - v.mean()
    var = ((d * d).sum() + (d[1:] * d[:-1]).sum()) / n
    se = float(np.sqrt(var / n)) if var > 0 else float("nan")
    return float(v.mean() / se) if se > 0 else float("nan"), se


def measures(rb):
    m = np.asarray([1e4 * r["cost0"] / r["days"] for r in rb])
    w = week_blocks(rb)
    tw, sew = t_nw1(w)
    return dict(B_cost0_bpd=round(S.bpd(rb), 3),
                monthly=dict(t=S.tstat(m), se=float(m.std(ddof=1) / np.sqrt(m.size)), n=int(m.size)),
                weekly_nw1=dict(t=tw, se=sew, n=int(w.size)))


def cut(sels, k):
    return [(hi[:, :k], lo[:, :k], n) for hi, lo, n in sels]


def cell_task(task):
    """(key, k, sels_A at 20, sels_B at 20) -> best A combo at K and its B-half measures (score worker)."""
    key, k, sa, sb = task
    S.K = k
    v = S._WS["view"]
    sa, sb = cut(sa, k), cut(sb, k)
    best = None
    for h, st, dr in C.COMBOS:
        a = S.bpd(S.score_sel(sa, v, h, st, dr))
        if np.isfinite(a) and (best is None or a > best[0]):
            best = (a, h, st, dr)
    if best is None:
        return (key, k), None
    a, h, st, dr = best
    rb = S.score_sel(sb, v, h, st, dr, halves=("B",), keep=True)
    return (key, k), dict(A_cost0_bpd=round(a, 3), h=h, state=S.STATES[st], dir=S.DIRS[dr], **measures(rb))


def write_rule():
    if os.path.exists(RULE_JSON):
        old = json.load(open(RULE_JSON, encoding="utf-8"))
        if {k: v for k, v in old.items() if k != "written"} != RULE:
            raise SystemExit("%s exists with a different rule" % RULE_JSON)
        return
    json.dump(dict(RULE, written=time.strftime("%Y-%m-%dT%H:%M:%S%z")), open(RULE_JSON, "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)


def main():
    t0 = time.time()
    if os.path.exists(OUT):
        raise SystemExit("%s exists" % OUT)
    write_rule()
    device = torch.device("cuda")
    vm = Q.make_vm()
    pb = tuple(D.plant_body(vm))
    aj = json.load(open(S.ALPHA_JSON, encoding="utf-8"))
    if list(pb) != aj["plant_body"]:
        raise SystemExit("plant body differs from SEL1_ALPHA.json")
    alpha = float(aj["alpha"])
    real = S.build("real", device, vm, log=log, keep_base=True)
    add = np.zeros_like(real["shard"]["fwd"])
    add[:, 1:] = np.nan_to_num(real["zp"][:, :-1], nan=0.0)
    pc = dict(real)                                                  # sel1_calib.world(alpha)
    pc["shard"] = S.no_winsor_shard(real["shard"], real["dead"], fwd=real["shard"]["fwd"] + alpha * add)
    pc["mode"] = "pc"
    sig_r, sig_p = S.signal(real, vm, pb), S.signal(pc, vm, pb)
    h, st, dr = C.PLANT_COMBO
    cells = {}
    for k in KS:
        S.K = k
        vis = measures(S.score(sig_p, pc, h, st, dr, halves=("B",), keep=True))
        rl = measures(S.score(sig_r, real, h, st, dr, halves=("B",), keep=True))
        cells[k] = dict(visible_book=vis, real_book=rl)
        log("K %d visible t monthly %.2f weekly %.2f | real B %.2f bp/day" % (
            k, vis["monthly"]["t"], vis["weekly_nw1"]["t"], rl["B_cost0_bpd"]))
    # pool: sel1_calib's
    bodies = [tuple(e["body"]) for e in json.load(open(C.CR1_HOF, encoding="utf-8"))["entries"][:50]]
    for fam in json.load(open(C.B5_HOF, encoding="utf-8"))["families"].values():
        bodies += [tuple(e["body"]) for e in fam]
    bodies = [b for b in dict.fromkeys(bodies) if b != pb] + [pb]
    S.K = max(KS)
    sels, bad = {}, 0
    for b in bodies:
        try:
            sg = sig_p if b == pb else S.signal(pc, vm, b)
        except Exception:
            bad += 1
            continue
        sels[b] = (S.select_months(sg, pc, ("A",)), S.select_months(sg, pc, ("B",)))
    log("pool %d bodies (%d invalid on V4) + plant" % (len(sels) - 1, bad))
    cdir = S.write_cache(pc, os.path.join(S.OUT_ROOT, "_power_cache"))
    tasks = [(b, k, sa, sb) for k in KS for b, (sa, sb) in sels.items()]
    with mp.get_context("spawn").Pool(20, initializer=S.init_score_worker, initargs=(cdir,)) as pool:
        res = dict(pool.imap_unordered(cell_task, tasks))
    table = []
    for k in KS:
        rows = [(b, r) for (b, kk), r in res.items() if kk == k and r]
        for meas in MEASURES:
            key = lambda x: -(x[1][meas]["t"] if np.isfinite(x[1][meas]["t"]) else -1e9)
            rs = sorted(rows, key=key)
            rank = next(i + 1 for i, (b, _) in enumerate(rs) if b == pb)
            pr = rs[rank - 1][1]
            vis = cells[k]["visible_book"][meas]
            row = dict(K=k, t=meas, t_B=vis["t"], se=vis["se"], n=vis["n"], visible=bool(vis["t"] >= C.T_MIN),
                       plant_rank=rank, pool=len(rs) - 1, rankable=bool(rank <= C.RANK_MAX),
                       plant_best_combo=dict(h=pr["h"], state=pr["state"], dir=pr["dir"], t_B=pr[meas]["t"]),
                       real_B_cost0_bpd=cells[k]["real_book"]["B_cost0_bpd"], real_B_t=cells[k]["real_book"][meas]["t"],
                       top3=[dict(expr=vm.describe(list(b), list(S.M.TERMINALS)), h=r["h"], state=r["state"],
                                  dir=r["dir"], t_B=round(r[meas]["t"], 2)) for b, r in rs[:3]])
            row["passed"] = row["visible"] and row["rankable"]
            table.append(row)
            log("K %2d %-10s t_B %5.2f se %6.2f rank %3d/%d real B %7.2f bp/day t %5.2f -> %s" % (
                k, meas, row["t_B"], row["se"], rank, row["pool"], row["real_B_cost0_bpd"], row["real_B_t"], row["passed"]))
    ok = [r for r in table if r["passed"]]
    chosen = min(ok, key=lambda r: (r["K"], MEASURES.index(r["t"]))) if ok else None
    rep = dict(schema="SEL1_POWER_V1", rule=RULE, alpha=alpha, table=table, cells=cells, chosen=chosen,
               verdict=("chosen K %d / %s" % (chosen["K"], chosen["t"])) if chosen else "no cell passes -> hand back",
               dev_read=False, oos_read=False, seconds=round(time.time() - t0, 1),
               written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=1, default=float, ensure_ascii=False)
    log("power -> %s (%.0fs)" % (rep["verdict"], time.time() - t0))
    sys.exit(0 if chosen else 7)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
