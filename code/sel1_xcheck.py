"""XS_SEL1 CPU self-checks for the 10-15 supplement items that need no world (new file, nothing modified).

  zfam      sel1_dedup.zfam: a family of identical rewards whose numpy std is ~1e-16 -> 0 (D.zfam gives +-noise);
            otherwise equal to D.zfam.
  rankz     sel1_pool.Layout.rankz on a synthetic 2-month world = pandas average rank, demean, std ddof 0, per hour.
  book x2   composite of 2 / 3 members with signs, hand-computed per position (python loops), and the selection
            of the composite = S.select_month on the same matrix (the trainer's numpy rule).
  one       a one-member composite selects exactly what the raw signal selects (empty pool reward = V2 reward).
  pool      sel1_pool.Pool with a stub t_dd (sum of member weights): add, replace (rho > 0.7 -> pool - j + f),
            POOL_SAME, full-pool drop of the one whose removal leaves the highest score.
    python -I sel1_xcheck.py      -> SEL1_XCHECK.json next to this file; rc 0 all pass, 7 not
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import torch
import sel1_common as S
import sel1_dedup as DD
import sel1_pool as P
import m2h_dense_common as D

OUT = {}


def check(name, ok, **kw):
    OUT[name] = dict(ok=bool(ok), **kw)
    print("%-8s %s %s" % (name, "PASS" if ok else "FAIL", json.dumps(kw, default=float)[:160]), flush=True)


def zfam_check():
    r = np.array([0.1 + 0.2] * 6 + [1.0, 2.0, 3.0, 0.5], np.float64)
    r[:6] = [0.3, 0.1 + 0.2, 0.3, 0.1 + 0.2, 0.3, 0.30000000000000004]
    valid = np.ones(10, bool)
    groups = [6, 4]                                               # counts, as the trainer passes
    sd = float(r[:6].std())
    a, b = DD.zfam(r, valid, groups), D.zfam(r, valid, groups)
    check("zfam", sd > 0 and sd < DD.STD_EPS and np.all(a[:6] == 0) and np.array_equal(a[6:], b[6:])
          and np.abs(b[:6]).max() > 0.5, std=sd, old=b[:6].tolist(), new=a[:6].tolist())


def world(seed=7):
    """2 A months of 6 rows, 12 coins; coin 0 masked out (BTC), a few NaN, ties."""
    rng = np.random.default_rng(seed)
    A, T = 12, 14
    mask = np.ones((A, T), bool)
    mask[0] = False
    mask[3, 4:7] = False
    months = [dict(half="A", m0=1, m1=7, month="m1", year=2024), dict(half="A", m0=7, m1=13, month="m2", year=2024)]
    ctx = dict(mask=mask, months=months, feats_gpu=torch.zeros((A, 1, T)))
    sig = [np.round(rng.normal(size=(A, T)), 1) for _ in range(3)]
    for s in sig:
        s[rng.random((A, T)) < 0.12] = np.nan
    return ctx, [torch.from_numpy(np.where(mask, s, np.nan)) for s in sig]


def hand_rankz(X):
    out = np.full(X.shape, np.nan)
    for h in range(X.shape[0]):
        x = X[h]
        f = np.isfinite(x)
        if not f.any():
            continue
        rk = pd.Series(x[f]).rank().to_numpy()
        d = rk - rk.mean()
        sd = np.sqrt((d * d).mean())
        out[h, f] = d / sd if sd > 0 else 0.0
    return out


def rankz_check(ctx, sig):
    lay = P.Layout(ctx)
    errs = []
    for s in sig:
        X = lay.rows(s).numpy()
        z = lay.rankz(s).numpy()
        hz = hand_rankz(X)[lay.mask.numpy()]
        errs.append(float(np.nanmax(np.abs(z - hz))) if np.isfinite(hz).any() else 0.0)
        if not np.array_equal(np.isfinite(z), np.isfinite(hz)):
            errs.append(np.inf)
    check("rankz", max(errs) <= 1e-12, max_abs=max(errs))
    return lay


def hand_composite(zs, signs, mask):
    """python loops: per position, sum of sign * z over finite members; NaN where none finite."""
    out = []
    for p in range(len(zs[0])):
        acc, anyf = 0.0, False
        for z, sg in zip(zs, signs):
            v = float(z[p])
            if np.isfinite(v):
                acc += sg * v
                anyf = True
        out.append(acc if anyf else np.nan)
    return np.asarray(out)


def book_check(lay, ctx, sig, name, idx, signs):
    zs = [lay.rankz(sig[i]).float() for i in idx]
    comp = lay.composite(zs, signs).numpy()
    hand = hand_composite([z.numpy().astype(np.float64) for z in zs], signs, lay.mask)
    same_fin = np.array_equal(np.isfinite(comp), np.isfinite(hand))
    err = float(np.nanmax(np.abs(comp - hand)))
    full = np.full((int(lay.off[-1]), lay.A), np.nan)
    full[lay.mask.numpy()] = comp
    sig_mat = np.full(ctx["mask"].shape, np.nan)                   # back to (A, T): hours m0-1 .. m1-2
    sig_mat[:, lay.hrs.numpy()] = full.T
    ref = S.select_months(sig_mat, ctx)
    got = lay.select(torch.from_numpy(comp))
    sel_eq = all(np.array_equal(a, b) for r, g in zip(ref, got) for a, b in zip(r, g))
    check(name, same_fin and err <= 1e-12 and sel_eq, members=list(idx), signs=signs, max_abs=err, selection_equal=sel_eq,
          first_row_hi=got[0][0][1].tolist())


def one_check(lay, ctx, sig):
    ok = True
    for s in sig:
        a = P.select_signal(ctx, s)
        b = lay.select(lay.composite([lay.rankz(s).float()]))
        c = S.select_months(s.numpy(), ctx)
        ok = ok and all(np.array_equal(x, y) and np.array_equal(x, w)
                        for r1, r2, r3 in zip(a, b, c) for x, y, w in zip(r1, r2, r3))
    check("one", ok)


def pool_check():
    """members are weights; t_dd(set) = sum of weights (stub). cost0 lists make rho controllable."""
    lay = None
    tdd = lambda sets: [float(sum(float(z) for z in zs)) for zs in sets]
    pool = P.Pool(2, lay, tdd, 0.7)
    up, dn, ot = [1, 2, 3, 4], [4, 3, 2, 1], [1, 3, 2, 4]
    f = lambda w, c, h: dict(z=w, cost0=c, hash=h, expr=h, body=h, single=w)
    log = []
    a = f(1.0, up, "a")
    ev = pool.evaluate([a])[0]
    log.append(("a", ev))
    ok = ev["act"] == "add" and ev["gain"] == 1.0
    pool.execute(a, ev)
    b = f(0.5, [1.1, 2.0, 3.1, 4.2], "b")                          # corr with a ~0.998 > 0.7 -> replace a
    c = f(2.0, dn, "c")                                            # corr with a = -1 -> add
    same = f(9.0, up, "a")
    evs = pool.evaluate([b, c, same])
    log.append(("b c same", evs))
    ok = ok and evs[0]["act"] == "rep" and evs[0]["j"] == 0 and evs[0]["gain"] == 0.5 - 1.0
    ok = ok and evs[1]["act"] == "add" and evs[1]["gain"] == 2.0 and evs[2] == "POOL_SAME"
    pool.execute(c, evs[1])                                        # pool a, c score 3
    d = f(1.5, ot, "d")                                            # corr(ot, up) 0.8 -> replace a
    e = f(-0.5, [2, 1, 4, 3], "e")                                 # corr with a 0.6, with c -0.6 -> add, full -> drop
    evs = pool.evaluate([d, e])
    log.append(("d e", evs))
    ok = ok and evs[0]["act"] == "rep" and evs[0]["j"] == 0 and abs(evs[0]["T"] - 3.5) < 1e-12
    r = pool.execute(d, evs[0])
    ok = ok and r["act"] == "rep" and r["out"] == "a" and [m["hash"] for m in pool.members] == ["d", "c"] and pool.score == 3.5
    evs = pool.evaluate([e])
    ok = ok and evs[0]["act"] == "add"
    r = pool.execute(e, evs[0])                                    # d c e: drop the one leaving the max: e (3.5)
    ok = ok and r["act"] == "add_drop" and r["out"] == "e" and r["dropped_newcomer"] and pool.score == 3.5
    check("pool", ok, rho_b=P.month_corr(b["cost0"], up), rho_d=P.month_corr(ot, up), rho_e=P.month_corr([2, 1, 4, 3], up),
          final=[m["hash"] for m in pool.members], score=pool.score)


def main():
    torch.set_num_threads(4)
    zfam_check()
    ctx, sig = world()
    lay = rankz_check(ctx, sig)
    book_check(lay, ctx, sig, "book1", (0, 1), [1.0, 1.0])
    book_check(lay, ctx, sig, "book2", (0, 1, 2), [1.0, -1.0, 1.0])
    one_check(lay, ctx, sig)
    pool_check()
    ok = all(v["ok"] for v in OUT.values())
    json.dump(dict(schema="SEL1_XCHECK_V1", checks=OUT, passed=ok), open(os.path.join(HERE, "SEL1_XCHECK.json"), "w"),
              indent=1, default=float)
    print("passed %s" % ok)
    sys.exit(0 if ok else 7)


if __name__ == "__main__":
    main()
