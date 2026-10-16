"""CR1 step 0 (order V4_COMBO_V1_CR1 s3): fixed combo on the A half, orthogonal plant s_perp, alpha for
p8 / p32 and the ruler check. TRAIN (V4 grid) only; no DEV, no OOS. Refuses if CR1_ALPHA.json exists.

  orth     per hour corr(s_perp, combo C) over coins (Pearson on the z values): mean |corr| < 0.05.
  alpha    s_perp alone, full book, cost0, mean over the 30 A tiles: lift over the real world = +8 / +32
           bp/day (within 2 %), secant from TV1's p8 alpha.
  ruler    the plant formula itself (D.plant_body) as the ninth cell: B-half marginal K10 net t >= 2 in
           the p8 world (else "PC not found" could not be put on the trainer; veto rule, hand to user).
           Reported for real / null / p8 / p32, K10 and full, A and B, with the seconds per call.

    python -I cr1_calib.py            rc 0 ruler ok, 7 ruler check failed, 8 orthogonality failed
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cr1_common as Q
import numpy as np
import torch
import m2h_common as M
import m2h_dense_common as D
import v4_world as X


def log(m):
    print(m, flush=True)


def pcorr(a, c, min_coins=20):
    """Per column Pearson over jointly finite coins; columns with < min_coins dropped."""
    out = []
    for j in range(a.shape[1]):
        m = np.isfinite(a[:, j]) & np.isfinite(c[:, j])
        if m.sum() >= min_coins:
            out.append(float(np.corrcoef(a[m, j], c[m, j])[0, 1]))
    return np.asarray(out)


def gross_alone(sh, sp, starts):
    v = []
    for s0 in starts:
        v.append(X.tile(X.tile_rows(sp, s0), sh, s0)["gross"])
    return float(np.mean(v))


def main():
    if os.path.exists(Q.ALPHA_JSON):
        raise SystemExit("%s exists; not overwritten" % Q.ALPHA_JSON)
    os.makedirs(os.path.dirname(Q.ALPHA_JSON), exist_ok=True)
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vm = Q.make_vm()
    if os.path.exists(Q.S_PERP_NPY):
        raise SystemExit("%s exists; not overwritten" % Q.S_PERP_NPY)
    ctx = Q.build("real", dev, vm, log=log)
    base, sp, A, B = ctx["base"], ctx["s_perp"], ctx["A"], ctx["B"]
    # orthogonality, every 24 h over the tile hours
    hrs = np.concatenate([np.arange(s0, s0 + X.E - 1, 24) for s0 in A + B])
    c = ctx["Cg"][:, hrs]
    perp, raw = pcorr(sp[:, hrs], c), pcorr(ctx["s_raw"][:, hrs], c)
    orth = dict(mean_abs_corr=round(float(np.mean(np.abs(perp))), 5), mean_corr=round(float(np.mean(perp)), 5),
                hours=len(perp), raw_vs_combo_mean_corr=round(float(np.mean(raw)), 4),
                raw_vs_perp_spearman=round(float(np.mean(D.cs_spearman_rows(ctx["s_raw"][:, hrs].T, sp[:, hrs].T))), 4))
    log("orth %s" % orth)
    g0 = gross_alone(base, sp, A)
    tv1 = json.load(open(Q.TV1_ALPHA, encoding="utf-8"))
    out = dict(schema="CR1_ALPHA_V1", s_perp_sha16=Q.sha16a(sp), s_perp_npy=Q.S_PERP_NPY, orth=orth,
               fit=ctx["fit"], A=[str(ctx["ts"][s]) for s in A], B=[str(ctx["ts"][s]) for s in B],
               base_gross_alone_bpd=round(g0, 4), plant_expr=vm.describe(D.plant_body(vm), list(M.TERMINALS)))
    for mode, target in Q.TARGETS.items():
        a_, hist = float(tv1["p8"]["alpha"]) * target / 8.0, []
        for _ in range(8):
            sh = X.install_layer(D.plant_shard(base, sp, a_), ctx["dead"])
            L = gross_alone(sh, sp, A) - g0
            hist.append((a_, round(L, 4)))
            if abs(L - target) <= 0.02 * target:
                break
            a_ = a_ * target / L if L > 0 else a_ * 4.0
        out[mode] = dict(alpha=hist[-1][0], target_bpd=target, lift_bpd=hist[-1][1], history=hist,
                         ok=abs(hist[-1][1] - target) <= 0.05 * target)
        log("alpha %s %s" % (mode, out[mode]))
    # ruler check: the plant formula as the ninth cell
    pb = D.plant_body(vm)
    slA, slB = M.tile_slices(ctx["feats_pad"], A, dev), M.tile_slices(ctx["feats_pad"], B, dev)
    books = (X.K10, ("full",))
    rul, secs = {}, None
    for mode in ("real", "null", "p8", "p32"):
        if mode == "real":
            sh = base
        elif mode == "null":
            sh = X.install_layer(D.null_shard(base, ctx["univ"], D.PLANT_SEED), ctx["dead"])
        else:
            sh = X.install_layer(D.plant_shard(base, sp, out[mode]["alpha"]), ctx["dead"])
        Q.set_world(ctx, sh)
        r = {}
        for half, st, sl in (("A", A, slA), ("B", B, slB)):
            t = time.time()
            res = Q.marginal(ctx, vm, pb, st, sl, books=books, corr_tiles=len(st))
            secs = round((time.time() - t) / len(st), 3)
            for b in ("conc_k10", "full"):
                v = [Q.bpd(x) for x in res["books"][b]["net"]]
                r["%s_%s_net" % (half, b)] = dict(mean=round(float(np.mean(v)), 3), t=round(Q.tstat(v), 2),
                                                  n_pos=int(np.sum(np.array(v) > 0)), n=len(v))
            r["%s_corr_s" % half] = round(res["corr"], 4)
            r["%s_combo_k10_net" % half] = round(float(np.mean([ctx["combo"][s]["conc_k10"]["net"] for s in st])), 3)
        rul[mode] = r
        log("ruler %s %s" % (mode, json.dumps(r)))
    k = rul["p8"]["B_conc_k10_net"]
    passes = {b: bool(rul["p8"]["B_%s_net" % b]["t"] >= Q.RULER_T_MIN and rul["p8"]["B_%s_net" % b]["mean"] > 0
                      and rul["p8"]["B_corr_s"] > Q.PLANT_CORR_MIN) for b in ("conc_k10", "full")}
    rank_book = "conc_k10" if passes["conc_k10"] else ("full" if passes["full"] else None)
    out.update(ruler=rul, ruler_rule="p8 world: plant formula as the ninth cell, B-half marginal net t >= %.1f and "
                                     "its corr_s with s_perp on B > %.1f. Ranking book = conc_k10 if it passes, else "
                                     "full if it passes (declared before the run); neither -> veto, user"
                                     % (Q.RULER_T_MIN, Q.PLANT_CORR_MIN),
               ruler_passes=passes, rank_book=rank_book, ruler_ok=rank_book is not None,
               seconds_per_tile_2books=secs,
               orth_ok=bool(orth["mean_abs_corr"] < Q.CORR_ORTH_MAX), dev_read=False, oos_read=False,
               seconds=round(time.time() - t0, 1), written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    out["ok"] = bool(out["ruler_ok"] and out["orth_ok"] and out["p8"]["ok"] and out["p32"]["ok"])
    np.save(Q.S_PERP_NPY, sp)                       # written with the JSON, so a crash leaves neither
    json.dump(out, open(Q.ALPHA_JSON, "w", encoding="utf-8"), indent=1, default=float)
    log("CALIB orth %s ruler %s rank_book %s (p8 B K10 %s full %s corr %s) alpha p8 %.5f p32 %.5f -> ok %s %.0fs"
        % (out["orth_ok"], out["ruler_ok"], rank_book, k, rul["p8"]["B_full_net"], rul["p8"]["B_corr_s"],
           out["p8"]["alpha"], out["p32"]["alpha"], out["ok"], out["seconds"]))
    sys.exit(0 if out["ok"] else (8 if not out["orth_ok"] else 7))


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
