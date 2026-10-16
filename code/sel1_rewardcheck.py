"""XS_SEL1 补充 10-15 夜 prep 1: --reward t / t_dd against a hand calculation on 3 formulas, production path.
Candidates: 3 from the hall of SEL1_SPEED_V4 checkpoint_0005 (null world, K 5), distinct bodies, at least one with a
state filter (months that hold nothing). Rows from the trainer's own Scorer (--fastts --gpu-select, 4 CPU score
workers) and, as a cross-check, in-process S.score_sel. Hand: plain Python over the 30 monthly cost0 log NAVs,
t = mean / (sd ddof 1 / sqrt n), max drawdown by a running peak from 0; t_dd line = the middle of the 3 drawdowns
(so one is kept at the boundary, one is zeroed). mean: S.reward(rows, "mean") == S.bpd(rows) (the old reward).
No returns beyond the A half of the null world. -> SEL1_REWARDCHECK.json, one line.
    python -I sel1_rewardcheck.py
"""
import json
import math
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import sel1_common as S
import sel1_trainer as TR
import cr1_common as Q
import m2h_common as M
import sel1_fastts as F

OUT = os.path.join(S.OUT_ROOT, "SEL1_REWARDCHECK.json")
REL_TOL = 1e-12


def hand_t(x):
    n = len(x)
    mu = sum(x) / n
    var = sum((v - mu) ** 2 for v in x) / (n - 1)
    return mu / math.sqrt(var / n) if var > 0 else 0.0


def hand_dd(x):
    nav = peak = dd = 0.0
    for v in x:
        nav += v
        peak = max(peak, nav)
        dd = max(dd, peak - nav)
    return dd


def pick(hof):
    out, bodies = [], set()
    for want_state in (lambda s: s == 0, lambda s: s != 0, lambda s: True):
        for c, _ in hof:
            if c[0] not in bodies and want_state(c[2]):
                out.append(c)
                bodies.add(c[0])
                break
    return out


def main():
    warnings.simplefilter("ignore", RuntimeWarning)
    ck = torch.load(os.path.join(S.OUT_ROOT, "SEL1_SPEED_V4", "checkpoint_0005.pt"), map_location="cpu",
                    weights_only=False)
    hof = [(tuple(tuple(x) if isinstance(x, list) else x for x in k), v) for k, v in ck["hof"]]
    vm = Q.make_vm()
    F.install()
    S.K = 5
    ctx = S.build("null", torch.device("cuda"), vm, log=lambda m: None)
    out_dir = os.path.join(S.OUT_ROOT, "_rewardcheck")
    sc = TR.Scorer(ctx, vm, out_dir, 4, 0, "global", 0.0, True, True)
    res, _, _ = sc.run([c for c, _ in hof], {})
    ok_hof = [(c, v) for c, v in hof if not isinstance(res[c], str)]
    cands = pick(ok_hof)
    sc.close()
    rows = {c: res[c] for c in cands}
    same_rows = all(S.rows_lite(S.score_sel(S.vm_select_gpu(ctx, vm, c[0]), ctx, *c[1:])) == rows[c] for c in cands)
    x = {c: [r["cost0"] for r in rows[c]] for c in cands}
    dds = {c: hand_dd(x[c]) for c in cands}
    line = sorted(dds.values())[1]
    rep, ok = [], same_rows
    for c in cands:
        r = rows[c]
        ht, hd = hand_t(x[c]), dds[c]
        pt, pdd = S.reward(r, "t"), S.max_dd(r)
        ptd, htd = S.reward(r, "t_dd", line), (0.0 if hd > line else ht)
        e = dict(expr=vm.describe(list(c[0]), list(M.TERMINALS)), h=c[1], state=S.STATES[c[2]], dir=S.DIRS[c[3]],
                 months=len(r), months_flat=sum(1 for v in x[c] if v == 0.0), t_prod=pt, t_hand=ht,
                 dd_prod=pdd, dd_hand=hd, t_dd_prod=ptd, t_dd_hand=htd, mean_prod=S.reward(r, "mean"), bpd=S.bpd(r))
        e["ok"] = bool(len(r) == 30 and abs(pt - ht) <= REL_TOL * max(1.0, abs(ht)) and pdd == hd
                       and abs(ptd - htd) <= REL_TOL * max(1.0, abs(htd)) and e["mean_prod"] == e["bpd"])
        ok = ok and e["ok"]
        rep.append(e)
    zeroed = sum(1 for e in rep if e["t_dd_prod"] == 0.0 and e["t_prod"] != 0.0)
    ok = ok and len(cands) == 3 and zeroed == 1
    json.dump(dict(schema="SEL1_REWARDCHECK_V1", rule=__doc__, dd_line_used=line, scorer_rows_eq_inprocess=same_rows,
                   candidates=rep, zeroed_by_line=zeroed, passed=ok, dev_read=False, oos_read=False),
              open(OUT, "w", encoding="utf-8"), indent=1, default=float, ensure_ascii=False)
    print("rewardcheck cands %d rows_eq %s zeroed %d | ok %s" % (len(cands), same_rows, zeroed,
          [e["ok"] for e in rep]), flush=True)
    sys.exit(0 if ok else 7)


if __name__ == "__main__":
    main()
