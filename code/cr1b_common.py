"""CR1b shared world (QuantMind 10-14 approval of CR1b; order V4_COMBO_V1_CR1 s3; contract V4). New file;
cr1_common (CR1's run record), v4_world, m1_ruler, m2h_* are imported, never modified.

Same as CR1 (cr1_common docstring): the 8-cell combo fit on the A half with real labels, the A/B halves,
the worlds (real / null / p8 / p32), the ruler (rank_weights -> held_path H168 -> delisting layer ->
settle_multi, flat6 / cost0, lag 1), the marginal = log NAV(combo + f) - log NAV(combo) per tile.
Changed:
  residual  the formula's TOP100 rank z per tile row is regressed per hour on the 8 cells' z (OLS with
            intercept over the coins where z_f is finite and the combo is defined; a missing cell z
            enters as 0, its cross-sectional mean); the residual, re-standardised (>= 20 coins), is the
            ninth equal-weight cell: (N + r_f) / (K + 1). Only what the 8 cells do not hold is rewarded.
  plant     s_perp8 = the plant formula's rank z residualised the same way on every tile row, so the
            plant formula's own residual is s_perp8; fwd[t] += alpha s_perp8[t-1]; alpha from s_perp8
            alone (full book, cost0, A tiles) = +8 / +32 bp/day.
  corr_s    Spearman of the residual r_f with s_perp8 (corr_raw: the raw signal with s_perp8, reported).
  books     every decision (pre-run checks, frozen set, NULL bar, found) on the K10 book net only
            (QuantMind E3: K10/K20 are the books we trade; ADR-0009 conc_k10_net). K20 and full are
            reported, never gated. Training reward = A-half marginal full-book net (CR1's; a training
            signal, not a gate).
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cr1_common as Q1                      # first: sys.path for m1_stage1 / okx_alpha_discovery / legacy
import m2h_verdict_dense as V
import m2h_common as M
import m2h_dense_common as D
import v4_world as X
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid
from cr1_common import (CELLS, STEP, MIN_CELLS, COV_MIN, TARGETS, MODES, A_ROOT, TV1_ALPHA, PLANT_CORR_MIN,
                        RULER_T_MIN, CORR_ORTH_MAX, sha16a, sha12f, make_vm, combine, tile_sigs, pack,
                        unpack, bpd, tstat)

K20 = ("conc", 20)
BOOKS3 = (X.K10, K20, ("full",))
RANK_BOOK, RANK_MAX = "conc_k10", 10
OUT = os.path.join(A_ROOT, "CR1B")
ALPHA_JSON = os.path.join(OUT, "CR1B_ALPHA.json")
S_PERP_NPY = os.path.join(OUT, "CR1B_S_PERP8.npy")
PC_POOL = os.path.join(A_ROOT, "CR1_PC_P8", "ROUND_PC.json")
SD_MIN = 1e-4


def fit_combo8(S, fut, univ, A, tiles):
    """Q1.fit_combo, plus every active cell's z on every tile's rows (NaN -> 0) for the residual."""
    hrs = np.concatenate([np.arange(s0, s0 + X.E, STEP) for s0 in A])
    ics = {k: Q1.stat(Q1.ic_rows(S[k], fut, univ, hrs)) for k in CELLS}
    act = [k for k in CELLS if ics[k]["ic"] is not None and ics[k]["cov"] >= COV_MIN]
    sign = {k: (1 if ics[k]["ic"] >= 0 else -1) if k in act else 0 for k in CELLS}
    N = np.zeros(univ.shape)
    K = np.zeros(univ.shape, dtype=np.int16)
    Z = {s0: np.zeros((len(act), X.E, univ.shape[0]), dtype=np.float32) for s0 in tiles}
    for j, k in enumerate(act):
        z = X.zrank_rows(np.where(univ & np.isfinite(S[k]), S[k], np.nan).T).T
        f = np.isfinite(z)
        N += np.where(f, sign[k] * z, 0.0)
        K += f.astype(np.int16)
        for s0 in tiles:
            Z[s0][j] = np.nan_to_num(X.tile_rows(z, s0), nan=0.0)
        del z, f
    need = min(MIN_CELLS, len(act))
    return N, K, need, dict(fit="rank IC over the A half: signal every 24 h inside every A tile, 168 h "
                                "forward, TOP100, real labels", sample_hours=int(len(hrs)), ic=ics,
                            sign=sign, active=act, need=need), Z


def resid_rows(zf, Zt, Kt, need, min_coins=D.PLANT_MIN_COINS):
    """(E, A) formula z -> per row OLS residual on [1, cell z] over coins with z_f finite and K >= need,
    re-standardised; rows with < min_coins coins or a ~0 residual -> NaN."""
    k = np.nan_to_num(Kt, nan=0.0)
    m = np.isfinite(zf) & (k >= max(need, 1))
    out = np.full(zf.shape, np.nan)
    cols = np.flatnonzero(m.any(axis=0))
    if cols.size == 0:
        return out
    mm = m[:, cols]
    y = np.where(mm, zf[:, cols], 0.0)
    Xr = np.concatenate([np.ones((1,) + y.shape), np.asarray(Zt[:, :, cols], dtype=np.float64)], axis=0)
    Xm = Xr * mm[None]
    XtX = np.einsum("pta,qta->tpq", Xm, Xm)
    Xty = np.einsum("pta,ta->tp", Xm, y)
    b = np.einsum("tpq,tq->tp", np.linalg.pinv(XtX, rcond=1e-10, hermitian=True), Xty)
    r = y - np.einsum("pta,tp->ta", Xr, b)
    n = mm.sum(axis=1)
    nn = np.maximum(n, 1)
    mu = np.where(mm, r, 0.0).sum(axis=1) / nn
    d = np.where(mm, r - mu[:, None], 0.0)
    sd = np.sqrt((d * d).sum(axis=1) / nn)
    ok = (n >= min_coins) & (sd > SD_MIN)
    out[:, cols] = np.where(mm & ok[:, None], d / np.where(sd > 0, sd, 1.0)[:, None], np.nan)
    return out


def load_alpha(mode, s_perp):
    cal = json.load(open(ALPHA_JSON, encoding="utf-8"))
    if cal["s_perp_sha16"] != sha16a(s_perp):
        raise RuntimeError("s_perp8 differs from the calibrated one")
    return float(cal[mode]["alpha"])
# --------------------------------------------------------------------------- world
def build(mode, device, vm, log=print, alpha=None, need_cal=True):
    """V4 TRAIN world for one mode: grid, TOP100, A/B tiles, the fixed combo (fit on A, real labels),
    the cells' z on every tile row, s_perp8, the mode's shard (+ delisting layer), the combo's own
    tile results in K10 / K20 / full."""
    if mode not in MODES:
        raise KeyError(mode)
    t0 = time.time()
    w = X.load(X.GRID)
    X.world(w, feats=True)
    S = V.handwritten(w["g"], w["ts"])
    S = {k: S[k] for k in CELLS}
    del w["g"]
    ts, univ, base = w["ts"], w["univ"], w["shard"]
    halves = X.ab_tiles(ts)
    A, B = list(halves["A"]), list(halves["B"])
    fut = Q1.fwd_cum(base["fwd_wx"])
    N, K, need, fit, Z = fit_combo8(S, fut, univ, A, A + B)
    del S, fut
    Cg = np.where(K >= need, N / np.maximum(K, 1), np.nan)
    feats_pad = M.pad_feats(w.pop("feats").astype(np.float32))
    Ng = np.where(K >= need, N, np.nan)
    Kf = K.astype(np.float64)
    rows = {s0: (X.tile_rows(Ng, s0), X.tile_rows(Kf, s0)) for s0 in A + B}
    del Kf, Ng
    # plant: the plant formula's TOP100 rank z on every tile row, residualised on the 8 cells
    pb = D.plant_body(vm)
    s_raw = np.full(univ.shape, np.nan)
    s_perp = np.full(univ.shape, np.nan)
    for s0 in A + B:
        sl = M.tile_slices(feats_pad, [s0], device)[0]
        z = X.zrank_rows(M.window_signal(vm, pb, sl, s0, univ))
        r = resid_rows(z, Z[s0], rows[s0][1], need)
        s_raw[:, s0:s0 + X.E - M.LAG] = z[M.LAG:].T
        s_perp[:, s0:s0 + X.E - M.LAG] = r[M.LAG:].T
        del sl
    s_diff = None
    if os.path.exists(S_PERP_NPY):                 # the calibrated s_perp8 is the one every world uses
        s_cal = np.load(S_PERP_NPY)
        if s_cal.shape != s_perp.shape:
            raise RuntimeError("s_perp8 shape %s != %s" % (s_cal.shape, s_perp.shape))
        both = np.isfinite(s_cal) & np.isfinite(s_perp)
        d = np.abs(s_cal[both] - s_perp[both])
        s_diff = dict(max=float(d.max()) if d.size else 0.0, frac_gt_1e4=float((d > 1e-4).mean()) if d.size else 0.0,
                      finite_mismatch=int((np.isfinite(s_cal) != np.isfinite(s_perp)).sum()))
        if s_diff["frac_gt_1e4"] > 0.01 or s_diff["finite_mismatch"] > 0.01 * both.sum():
            raise RuntimeError("s_perp8 recomputed differs from %s: %s" % (S_PERP_NPY, s_diff))
        s_perp = s_cal
    elif need_cal and mode != "real":
        raise RuntimeError("%s missing: run cr1b_calib first" % S_PERP_NPY)
    if mode == "real":
        sh = base
    elif mode == "null":
        sh = X.install_layer(D.null_shard(base, univ, D.PLANT_SEED), w["dead"])
    else:
        a = load_alpha(mode, s_perp) if alpha is None else float(alpha)
        sh = X.install_layer(D.plant_shard(base, s_perp, a), w["dead"])
    ctx = dict(mode=mode, ts=ts, univ=univ, syms=w["syms"], dead=w["dead"], A=A, B=B, need=need, fit=fit,
               s_perp_diff=s_diff, s_perp_sha16=sha16a(s_perp), feats_pad=feats_pad, base=base, shard=sh,
               s_raw=s_raw, s_perp=s_perp, Cg=Cg, rows=rows, Z=Z, combo={})
    set_world(ctx, sh)
    log("CR1b world %s: %d coins x %d h, A %d / B %d tiles, %d delistings, active cells %d, %.0fs"
        % (mode, len(w["syms"]), len(ts), len(A), len(B), len(w["dead"]), len(fit["active"]), time.time() - t0))
    return ctx


def set_world(ctx, sh):
    """Swap the shard and recompute the combo's own result on every tile and every book."""
    ctx["shard"] = sh
    ctx["combo"] = {}
    for s0, (Nt, Kt) in ctx["rows"].items():
        Ct = combine(Nt, Kt, ctx["need"], None)
        ctx["combo"][s0] = {X.bk(b): X.tile(X.book_sig(Ct, sh, s0, b), sh, s0) for b in BOOKS3}
    return ctx


def tile_sigs_list(ctx, vm, body, starts, slices):
    """Q1.tile_sigs with the live fraction per tile (the trainer's (body, tile) cache needs it)."""
    sigs, lf = [], []
    for s0, sl in zip(starts, slices):
        sg, f = tile_sigs(ctx, vm, body, [s0], [sl])
        sigs.append(sg[0])
        lf.append(f)
    return sigs, lf


def marginal_from_sigs(rc, sigs, starts, books=(("full",),), corr_tiles=0, strict=True):
    """CPU part, per tile: log NAV(combo + r_f) - log NAV(combo), r_f = the formula's z residualised on
    the 8 cells. rc is a ctx or a worker's cache view. strict: CandidateLocalInvalid when r_f is empty on
    every tile given (the trainer passes strict=False and applies the same rule over all drawn tiles)."""
    sh = rc["shard"]
    out = {X.bk(b): dict(net=[], gross=[], turn=[]) for b in books}
    corr, craw, nzs = [], [], []
    for i, (s0, sigT) in enumerate(zip(starts, sigs)):
        sg = np.asarray(sigT, dtype=np.float64)
        Nt, Kt = rc["rows"][s0]
        rf = resid_rows(X.zrank_rows(sg), rc["Z"][s0], Kt, rc["need"])
        nzs.append(int(np.isfinite(rf).any(axis=1).sum()))
        comb = combine(Nt, Kt, rc["need"], rf)
        for b in books:
            k = X.bk(b)
            r = X.tile(X.book_sig(comb, sh, s0, b), sh, s0)
            c = rc["combo"][s0][k]
            out[k]["net"].append(float(r["ln_net"] - c["ln_net"]))
            out[k]["gross"].append(float(r["ln_gross"] - c["ln_gross"]))
            out[k]["turn"].append(float(r["turn"]))
        if i < corr_tiles:
            corr.append(D.corr_with_s(rf, rc["s_perp"], s0))
            craw.append(D.corr_with_s(sg, rc["s_perp"], s0))
    if strict and sum(nzs) == 0:
        raise CandidateLocalInvalid("RESID_NO_Z")
    return dict(books=out, corr=float(np.mean(corr)) if corr else float("nan"),
                corr_raw=float(np.mean(craw)) if craw else float("nan"),
                per_tile=dict(nz=nzs, corr=corr, corr_raw=craw))


def marginal(ctx, vm, body, starts, slices, books=(("full",),), corr_tiles=0):
    sigs, lf = tile_sigs(ctx, vm, body, starts, slices)
    res = marginal_from_sigs(ctx, sigs, starts, books, corr_tiles)
    res["live_frac"] = lf
    return res


# --------------------------------------------------------------------------- worker pool (CPU ruler)
CACHE_KEYS = ("eligible", "fwd_wx", "fi", "fr")


def write_cache(ctx, cdir):
    """What a worker needs, as .npy (mmap-read) + one pickle; float64 / float32 kept as built."""
    import pickle
    os.makedirs(cdir, exist_ok=True)
    for k in CACHE_KEYS:
        np.save(os.path.join(cdir, k + ".npy"), ctx["shard"][k])
    np.save(os.path.join(cdir, "s_perp.npy"), ctx["s_perp"])
    keys = sorted(ctx["rows"])
    np.save(os.path.join(cdir, "rows.npy"), np.stack([np.stack(ctx["rows"][s]) for s in keys]))
    np.save(os.path.join(cdir, "Z.npy"), np.stack([ctx["Z"][s] for s in keys]))
    pickle.dump(dict(keys=keys, combo=ctx["combo"], need=ctx["need"], dead=ctx["shard"]["dead"]),
                open(os.path.join(cdir, "meta.pkl"), "wb"))
    return cdir


_W = {}


def init_worker(cdir):
    import pickle
    meta = pickle.load(open(os.path.join(cdir, "meta.pkl"), "rb"))
    sh = {k: np.load(os.path.join(cdir, k + ".npy"), mmap_mode="c") for k in CACHE_KEYS}
    sh["dead"] = meta["dead"]
    rows = np.load(os.path.join(cdir, "rows.npy"), mmap_mode="c")
    Z = np.load(os.path.join(cdir, "Z.npy"), mmap_mode="r")
    _W.update(shard=sh, need=meta["need"], combo=meta["combo"],
              s_perp=np.load(os.path.join(cdir, "s_perp.npy"), mmap_mode="c"),
              rows={s: (np.asarray(rows[i, 0]), np.asarray(rows[i, 1])) for i, s in enumerate(meta["keys"])},
              Z={s: Z[i] for i, s in enumerate(meta["keys"])})
    warnings.simplefilter("ignore", RuntimeWarning)


def worker_task(task):
    key, packed, starts, books, corr_tiles = task[:5]
    strict = task[5] if len(task) > 5 else True
    try:
        return key, marginal_from_sigs(_W, unpack(packed), starts, books, corr_tiles, strict)
    except CandidateLocalInvalid as e:
        return key, "INVALID:" + str(e)
