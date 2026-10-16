"""XS_SEL1 zero-diff check on an update range (supplement 10-15 evening, GPU self-checks). New file; sel1_logdiff
(open-ended, whole tail) is not modified. Two runs' BATCH_LOG / FORMULAS rows with FROM <= update <= TO, line by line;
the timing fields of sel1_logdiff are dropped, plus --drop keys. --only FORMULAS compares that file alone.
--gain-eq-score: every row of B with a gain must have gain == score (the empty pool's first update).
--near k1,k2: those keys are compared apart with relative tolerance --rtol (default 1e-6), the known last-digit
differences of the GPU backward (pg_loss and what follows from the updated weights); the max relative diff is reported.
    python -I sel1_logdiff2.py RUN_A RUN_B FROM TO [--drop k1,k2] [--near k1,k2] [--rtol R] [--only FORMULAS]
                               [--gain-eq-score] [--tag T]
-> LOGDIFF2_<tag>.json in RUN_B, one line; rc 0 zero diff, 9 not.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sel1_common as S
from sel1_logdiff import TIMING


def rows(run, fn, u0, u1, drop, near=()):
    out, raw, nv = [], [], []
    for ln in open(os.path.join(S.OUT_ROOT, run, fn), encoding="utf-8"):
        r = json.loads(ln)
        if u0 <= r["update"] <= u1:
            raw.append(dict(r))
            for k in TIMING + drop:
                r.pop(k, None)
            nv.append({k: r.pop(k) for k in near if k in r})
            out.append(json.dumps(r, sort_keys=True))
    return out, raw, nv


def rel(x, y):
    if x is None or y is None:
        return 0.0 if x is y else float("inf")
    return 0.0 if x == y else abs(x - y) / max(abs(x), abs(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("u0", type=int)
    ap.add_argument("u1", type=int)
    ap.add_argument("--drop", default="")
    ap.add_argument("--near", default="")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--only", default=None)
    ap.add_argument("--gain-eq-score", action="store_true")
    ap.add_argument("--tag", default="check")
    a = ap.parse_args()
    drop = tuple(k for k in a.drop.split(",") if k)
    near = tuple(k for k in a.near.split(",") if k)
    fns = [a.only + ".jsonl"] if a.only else ["BATCH_LOG.jsonl", "FORMULAS.jsonl"]
    res, ok = {}, True
    for fn in fns:
        ra, _, na = rows(a.a, fn, a.u0, a.u1, drop, near)
        rb, raw_b, nb = rows(a.b, fn, a.u0, a.u1, drop, near)
        dn = [max([rel(x.get(k), y.get(k)) for k in set(x) | set(y)], default=0.0) for x, y in zip(na, nb)]
        diff = [i for i, (x, y) in enumerate(zip(ra, rb)) if x != y or dn[i] > a.rtol]
        res[fn] = dict(n_a=len(ra), n_b=len(rb), n_diff=len(diff) + abs(len(ra) - len(rb)),
                       near_max_rel=max(dn, default=0.0) if near else None,
                       first=None if not diff else (ra[diff[0]][:200], rb[diff[0]][:200], na[diff[0]], nb[diff[0]]))
        ok = ok and res[fn]["n_diff"] == 0 and res[fn]["n_a"] > 0
        if a.gain_eq_score and fn == "FORMULAS.jsonl":
            g = [r for r in raw_b if r.get("gain") is not None]
            bad = [r for r in g if r["gain"] != r.get("score")]
            res["gain_eq_score"] = dict(n=len(g), n_bad=len(bad), first=None if not bad else json.dumps(bad[0])[:200])
            ok = ok and len(g) > 0 and not bad
    json.dump(dict(a=a.a, b=a.b, updates=[a.u0, a.u1], drop=list(drop), near=list(near), rtol=a.rtol, files=res, zero_diff=ok),
              open(os.path.join(S.OUT_ROOT, a.b, "LOGDIFF2_%s.json" % a.tag), "w"), indent=1, default=str)
    print("%s zero_diff %s | %s" % (a.tag, ok, " ".join("%s %s" % (k[:5], v.get("n_diff", v.get("n_bad")))
                                                      for k, v in res.items())))
    sys.exit(0 if ok else 9)


if __name__ == "__main__":
    main()
