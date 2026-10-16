"""ENR1b LS cells, one DEV read (order XS_SEL1 〇). CPU only; DEV 2026-01..06 read once; no OOS. New file; enr1b,
v4_world and m1_ruler are imported, never modified. Cells and pass rule are frozen in FROZEN_ENR1b_LS.json, written
before this runs; this file refuses any other frozen file and refuses to overwrite its output.

  world   V4 DEV grid (v4_world.load(DEVZ) + world(feats=False)); main fwd = m1_ruler.prepare_shard(winsor_pct=0) +
          v4_world.install_layer, asserted == world()'s fwd_x; the winsorised fwd_wx is report only.
  events  ENR1b rules: T2_neg_retail_LS_24h (h1 formula via m2h_verdict_dense.handwritten), TS_lo over the coin's
          past 2160 h (>= 1080 values, current hour included), TOP100 at the trigger hour, not BTC, beta (720 h,
          >= 360) finite, next bar tradable, t + 1 < L, trigger hour in 2026-01..06, t..t+h+1 inside the trigger
          month, greedy per-coin dedup (next event at >= t + h).
  settle  each event through m1_ruler.settle_multi on its own two-column book [coin, BTC] = [d, -d beta] on rows
          t+1..t+h (both 0 from L), 0 on the exit row t+h+1; flat6 and cost0; held / fwd columns asserted equal.
"""
from __future__ import annotations

import hashlib
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
import enr1b as E

OUT_DIR = os.path.join(E.ROOT, "artifacts", "m2h_v1", "XS_SEL1")
FROZEN = os.path.join(OUT_DIR, "FROZEN_ENR1b_LS.json")
FROZEN_SHA16 = "232cf015aed57426"
OUT = os.path.join(OUT_DIR, "ENR1b_LS_DEV.json")
SIG = "T2_neg_retail_LS_24h"
HOLDS = (72, 168)
D_DIR = -1
DEV_FIRST = pd.Timestamp("2026-01-01")


def log(m):
    print(m, flush=True)


def main():
    t0 = time.time()
    if E.sha16(FROZEN) != FROZEN_SHA16:
        raise SystemExit("frozen file sha16 %s != %s" % (E.sha16(FROZEN), FROZEN_SHA16))
    if os.path.exists(OUT):
        raise SystemExit("%s exists; DEV is read once" % OUT)
    import m2h_verdict_dense as V
    import v4_world as X
    import m1_ruler as R
    w = X.load(X.DEVZ)
    X.world(w, feats=False)
    g, ts, univ, sh = w["g"], w["ts"], w["univ"], w["shard"]
    A, T = univ.shape
    bi = w["syms"].index("BTCUSDT")
    sh0 = dict(fwd=sh["fwd"], eligible=sh["eligible"], fi=sh["fi"], fr=sh["fr"])
    R.prepare_shard(sh0, 0)
    X.install_layer(sh0, w["dead"])
    if not np.array_equal(sh0["fwd_wx"], sh["fwd_x"], equal_nan=True):
        raise SystemExit("prepare_shard(winsor_pct=0) + layer != world fwd_x")
    fw_main, fw_win = sh0["fwd_wx"], sh["fwd_wx"]
    s = V.handwritten(g, ts)[SIG]
    s = np.where(np.isfinite(s), s, np.nan)
    ro = pd.DataFrame(s.T).rolling(E.TS_WIN, min_periods=E.TS_MIN)
    rr, cc = ro.rank().to_numpy().T, ro.count().to_numpy().T
    with np.errstate(all="ignore"):
        qt = np.where(np.isfinite(s) & (cc >= E.TS_MIN), (rr - 1.0) / (cc - 1.0), np.nan)
    logc = np.log(g["trade_close"])
    lr = np.diff(logc, axis=1, prepend=np.nan)
    x = lr[bi]
    msk = np.isfinite(lr) & np.isfinite(x)[None, :]
    X0, Y0 = np.where(msk, x[None, :], 0.0), np.where(msk, lr, 0.0)

    def rs(z):
        c = np.cumsum(z, axis=1)
        c[:, E.BETA_WIN:] -= c[:, :-E.BETA_WIN].copy()
        return c
    nn, sx, sy, sxy, sxx = rs(msk.astype(np.float64)), rs(X0), rs(Y0), rs(X0 * Y0), rs(X0 * X0)
    with np.errstate(all="ignore"):
        var = sxx - sx * sx / nn
        beta = np.where((nn >= E.BETA_MIN) & (var > 0), (sxy - sx * sy / nn) / var, np.nan)
    i0 = int(ts.searchsorted(DEV_FIRST))
    nxt_ok = np.zeros((A, T), bool)
    nxt_ok[:, :-1] = np.isfinite(fw_main[:, 1:])
    E0 = univ & nxt_ok & np.isfinite(beta)
    E0[:, :i0] = False
    E0[bi] = False
    Larr = np.full(A, T + 10, np.int64)
    for a, L in w["dead"].items():
        E0[a, max(L - 1, 0):] = False
        Larr[a] = L
    trig = (qt <= E.Q_LO) & E0
    fa, ft = np.nonzero(trig)
    mon = np.asarray(ts.year * 12 + ts.month)
    wk = np.asarray((ts - pd.Timestamp("2020-10-26")) // pd.Timedelta(hours=168), dtype=np.int64)
    D = dict(EH=ft.astype(np.int64), CO=fa.astype(np.int64))
    for h in HOLDS:
        v = np.zeros(T, bool)
        v[:T - h - 1] = mon[:T - h - 1] == mon[h + 1:]
        D["V%d" % h] = v[ft]
    specs = {c: R.cost_specs(R.NAV_USD)[c] for c in ("flat6", "cost0")}
    log("DEV world %.0fs: %d coins x %d h (%s .. %s), %d triggers" % (time.time() - t0, A, T, ts[i0], ts[-1], fa.size))
    fz = json.load(open(FROZEN, encoding="utf-8"))
    res = {}
    for h in HOLDS:
        ev = E.dedup(np.arange(fa.size), D, h)
        rec = dict(net=[], cost0=[], net_win=[], cost0_win=[], month=[], week=[], t=[])
        bad_exit = 0
        for p in ev:
            a, t = int(D["CO"][p]), int(D["EH"][p])
            L = int(Larr[a])
            b = float(beta[a, t])
            rows = np.arange(t + 1, t + h + 2)
            held = np.zeros((h + 1, 2))
            held[:h, 0], held[:h, 1] = D_DIR, -D_DIR * b
            held[rows >= L] = 0.0
            cols = [a, bi]
            if L > t + h and not np.isfinite(fw_main[a, t + h + 1]):
                bad_exit += 1
            for key, fw in (("", fw_main), ("_win", fw_win)):
                f = fw[cols][:, rows].T
                fi, fr = sh["fi"][cols][:, rows].T, sh["fr"][cols][:, rows].T
                assert held.shape[1] == f.shape[1] == fi.shape[1] == fr.shape[1], "held / fwd column count differs"
                r_, _ = R.settle_multi(held, f, fi, fr, specs)
                rec["net" + key].append(r_["flat6"]["log_nav"])
                rec["cost0" + key].append(r_["cost0"]["log_nav"])
            rec["month"].append(str(ts[t])[:7])
            rec["week"].append(int(wk[t]))
            rec["t"].append(str(ts[t]))
        net, c0 = np.array(rec["net"]), np.array(rec["cost0"])
        n = net.size
        st = None
        if n >= 2:
            ww = np.array(rec["week"])
            Dt = dict(ST=np.zeros(n, np.int8), HF=np.zeros(n, np.int8), WK=ww - ww.min(), NW=np.array([int(ww.max() - ww.min()) + 1]))
            _, _, tt = E.tstats(c0, np.arange(n), Dt)
            st = float(tt[0, 0])
        kk = max(1, int(np.ceil(E.TRIM * n)))
        srt = np.sort(net)
        months = sorted(set(rec["month"]))
        res["H%d" % h] = dict(
            n_events=int(n), net_bp=round(1e4 * float(net.mean()), 2) if n else None,
            cost0_bp=round(1e4 * float(c0.mean()), 2) if n else None,
            t_cost0_week_clustered=round(st, 2) if st is not None and np.isfinite(st) else None,
            trimmed_net_bp=round(1e4 * float(srt[:n - kk].mean()), 2) if n > kk else None, trimmed_dropped=kk,
            net_bp_winsor_report_only=round(1e4 * float(np.mean(rec["net_win"])), 2) if n else None,
            per_month={m_: dict(n=int(sum(1 for x in rec["month"] if x == m_)),
                                net_bp=round(1e4 * float(np.mean([v for v, x in zip(rec["net"], rec["month"]) if x == m_])), 2))
                       for m_ in months},
            nontradable_exit_events=bad_exit,
            passed=bool(n > 0 and float(net.mean()) > 0))
        log("H%d: n %d net %.2f cost0 %.2f t %s trimmed %s -> %s" % (h, n, res["H%d" % h]["net_bp"] or 0,
            res["H%d" % h]["cost0_bp"] or 0, res["H%d" % h]["t_cost0_week_clustered"], res["H%d" % h]["trimmed_net_bp"],
            res["H%d" % h]["passed"]))
    out = dict(schema="ENR1b_LS_DEV_V1", frozen_sha16=FROZEN_SHA16, code_sha16=E.sha16(os.path.abspath(__file__)),
               dev_grid_sha16=E.sha16(X.DEVZ), pass_rule=fz["dev"]["pass_rule"], d=D_DIR,
               span=[str(ts[i0]), str(ts[-1])], results=res, weak_evidence=fz["weak_evidence"],
               dev_read=True, oos_read=False, seconds=round(time.time() - t0, 1),
               written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    log("wrote %s" % OUT)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
