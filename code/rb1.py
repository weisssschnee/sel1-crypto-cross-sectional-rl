"""RB1 rule book on V4 TRAIN (order ENR1 step 3; ADR-0012). CPU only, no DEV, no OOS. New weight layer; m1_ruler,
v4_world and enr1 are imported, never modified. Rules = the cells that passed ENR1_TABLE.json.

  rule      trigger of the cell's (signal, condition) at hour t on an eligible coin-hour whose t..t+h+1 stays in t's
            month (enr1 V_h) -> coin at d / 5 NAV on rows t+1..t+h, BTC at -d beta_i(t) / 5 NAV, both 0 from the
            delisting hour L on; exit row t+h+1.
  book      5 fixed slots of 1/5 NAV (equal weight, no rebalancing between positions); one position per coin; a
            trigger is taken only if a slot is free and the coin is not held. Same-hour ties: rules by |t_A|
            (descending), then coin index. A slot frees once its exit row is reached (next trigger t' >= t + h).
  settle    per calendar month (every position opens and closes inside one month, so each month starts and ends
            flat) through m1_ruler.settle_multi, flat6 / cost0, Rule 0 freezing; held and fwd column counts asserted.
            Two fwd grids: fwd_wx (ruler convention, 1/99 winsor + delisting layer) and fwd_x (no winsor) -- the
            ENR1 table showed the passing cells turn negative without the winsor, so both are reported.
  gate      (order) B net > 0 and >= 4 of 5 years net > 0, on fwd_wx. FROZEN_RB1 / DEV are NOT done here.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import enr1 as E

OUT = os.path.join(E.OUT_DIR, "RB1_TRAIN.json")
SLOTS = 5


def log(m):
    print(m, flush=True)


def build_held(D, ctx, rules):
    """(T, A) weights of the rule book; returns held and per-position records."""
    T, A, bi, Larr = ctx["T"], len(ctx["syms"]), ctx["bi"], ctx["Larr"]
    cand = []
    for pri, (k, c, h, d) in enumerate(rules):
        j = k * len(E.CONDS) + c
        P = D["PT"][D["PO"][j]:D["PO"][j + 1]]
        P = P[D["V%d" % h][P]]
        cand.append(np.stack([D["EH"][P], np.full(P.size, pri), D["CO"][P], P], axis=1))
    cand = np.concatenate(cand)
    cand = cand[np.lexsort((cand[:, 2], cand[:, 1], cand[:, 0]))]
    held = np.zeros((T, A))
    active = []                      # (free_at_trigger_hour, coin)
    pos = []
    for t, pri, a, p in cand:
        t, a = int(t), int(a)
        active = [x for x in active if x[0] > t]
        if len(active) >= SLOTS or any(x[1] == a for x in active):
            continue
        k, c, h, d = rules[pri]
        b = float(D["BETA"][p])
        end = min(t + h, int(Larr[a]) - 1)          # last held row (delisting: both legs out from L)
        held[t + 1:end + 1, a] += d / SLOTS
        held[t + 1:end + 1, bi] += -d * b / SLOTS
        active.append((t + h, a))
        pos.append((t, pri, a, h, d, b))
    return held, pos


def settle_months(held, ctx, R, key):
    sh, ts = ctx["sh"], ctx["ts"]
    specs = {c: R.cost_specs(R.NAV_USD)[c] for c in ("flat6", "cost0")}
    mon = ctx["mon"]
    i0 = int(ts.get_loc(pd.Timestamp("2021-01-01")))
    out = []
    for m in np.unique(mon[i0:]):
        rows = np.flatnonzero(mon == m)
        h = held[rows]
        fw = sh[key][:, rows].T
        fi, fr = sh["fi"][:, rows].T, sh["fr"][:, rows].T
        assert h.shape[1] == fw.shape[1] == fi.shape[1] == fr.shape[1], "held / fwd column count differs"
        assert np.all(h[0] == 0) and np.all(h[-1] == 0), "a month does not start / end flat"
        res, parts = R.settle_multi(h, fw, fi, fr, specs)
        parts.update(R.book_diagnostics(h, fw))
        fw0 = np.nan_to_num(fw)
        out.append(dict(month=str(ts[rows[0]])[:7], year=int(ts[rows[0]].year), half="A" if ts[rows[0]].month % 2 else "B",
                        days=len(rows) / 24.0, net=res["flat6"]["log_nav"], gross=res["cost0"]["log_nav"],
                        turnover=parts["turnover"], beta_ew=parts["beta_mkt"],
                        book_ret=(h * fw0).sum(axis=1), btc_ret=fw0[:, ctx["bi"]],
                        n_pos=(np.abs(np.delete(h, ctx["bi"], axis=1)) > 0).sum(axis=1),
                        net_exp=h.sum(axis=1), gross_exp=np.abs(h).sum(axis=1)))
    return out


def summarise(ms, sel):
    ms = [m for m in ms if sel(m)]
    if not ms:
        return None
    days = sum(m["days"] for m in ms)
    net = np.array([m["net"] for m in ms])
    br, bb = np.concatenate([m["book_ret"] for m in ms]), np.concatenate([m["btc_ret"] for m in ms])
    vb = float(bb.var())
    return dict(months=len(ms), net_bp_day=round(1e4 * net.sum() / days, 3),
                gross_bp_day=round(1e4 * sum(m["gross"] for m in ms) / days, 3),
                sharpe_monthly_ann=round(float(net.mean() / net.std(ddof=1) * np.sqrt(12)), 3) if len(ms) > 1 and net.std() > 0 else None,
                months_pos=int((net > 0).sum()),
                turnover_day=round(sum(m["turnover"] for m in ms) / days, 4),
                avg_positions=round(float(np.concatenate([m["n_pos"] for m in ms]).mean()), 3),
                avg_net_exposure=round(float(np.concatenate([m["net_exp"] for m in ms]).mean()), 4),
                avg_gross_exposure=round(float(np.concatenate([m["gross_exp"] for m in ms]).mean()), 4),
                beta_btc=round(float(np.cov(br, bb, bias=True)[0, 1] / vb), 4) if vb > 0 else None,
                beta_ew_mean=round(float(np.nanmean([m["beta_ew"] for m in ms])), 4))


def main():
    t0 = time.time()
    if os.path.exists(OUT):
        raise SystemExit("%s exists; not overwritten" % OUT)
    tab = json.load(open(E.TABLE, encoding="utf-8"))
    if tab.get("test") or tab["prereg_sha16"] != E.PREREG_SHA16:
        raise SystemExit("not the real ENR1 table")
    cells = sorted(tab["passed"], key=lambda k: -abs(tab["cells"][k]["A"]["t"]))
    if not cells:
        raise SystemExit("no passing cell; step 3 is not run")
    rules = []
    for key in cells:
        k, c, hi, s = tab["cells"][key]["idx"]
        if s != 0:
            raise SystemExit("state-conditional rule %s not implemented" % key)
        rules.append((k, c, E.HOLDS[hi], tab["cells"][key]["d"]))
    import m2h_verdict_dense as V
    import v4_world as X
    import m1_ruler as R
    D, ctx = E.precompute(V, X)
    held, pos = build_held(D, ctx, rules)
    log("book %d positions %.0fs" % (len(pos), time.time() - t0))
    rep = dict(schema="RB1_TRAIN_V1", enr1_table_sha16=E.sha16(E.TABLE), code_sha16=E.sha16(os.path.abspath(__file__)),
               rules=[dict(cell=key, signal=E.SIGS[r[0]], cond=E.CONDS[r[1]], hold_h=r[2], d=r[3]) for key, r in zip(cells, rules)],
               slots=SLOTS, n_positions=len(pos),
               positions_per_rule={key: sum(1 for p in pos if p[1] == i) for i, key in enumerate(cells)},
               dev_read=False, oos_read=False, conventions=__doc__)
    for key, nm in (("fwd_wx", "winsor_ruler"), ("fwd_x", "no_winsor")):
        ms = settle_months(held, ctx, R, key)
        r = dict(all=summarise(ms, lambda m: True), A=summarise(ms, lambda m: m["half"] == "A"),
                 B=summarise(ms, lambda m: m["half"] == "B"),
                 years={str(y): summarise(ms, lambda m, y=y: m["year"] == y) for y in E.YEARS},
                 per_month=[dict(month=m["month"], net_bp_day=round(1e4 * m["net"] / m["days"], 2)) for m in ms])
        yp = sum(1 for y in E.YEARS if r["years"][str(y)]["net_bp_day"] > 0)
        r["gate"] = dict(rule="B net > 0 and >= 4/5 years net > 0", B_net=r["B"]["net_bp_day"], years_pos=yp,
                         passed=bool(r["B"]["net_bp_day"] > 0 and yp >= 4))
        rep[nm] = r
        log("%s: all %.2f bp/day A %.2f B %.2f years+ %d/5 sharpe %s pos %.2f netexp %.3f beta_btc %s -> %s"
            % (nm, r["all"]["net_bp_day"], r["A"]["net_bp_day"], r["B"]["net_bp_day"], yp, r["all"]["sharpe_monthly_ann"],
               r["all"]["avg_positions"], r["all"]["avg_net_exposure"], r["all"]["beta_btc"], r["gate"]["passed"]))
    rep["frozen_dev"] = "not done: waiting for the user's decision (the passing cells depend on the 1/99 winsor)"
    rep["seconds"] = round(time.time() - t0, 1)
    rep["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    log("wrote %s %.0fs" % (OUT, time.time() - t0))


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
