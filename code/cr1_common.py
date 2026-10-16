"""CR1 shared world (order V4_COMBO_V1_CR1 s3; ADR-0010 decision 3; contract V4). New file; v4_world,
m1_ruler, m2h_common, m2h_dense_common and the B5 dbg_memo trainer are imported, never modified.

  combo    COMBO_V1's 8 h1 handwritten H168 cells. Signs = sign of the rank IC over the A half only
           (signal every 24 h inside every A tile, 168 h forward, TOP100, delisting layer); a cell is
           active if its IC coverage >= 50 %; equal weight. Per hour per cell: average rank -> z over
           TOP100 & finite (>= 20 coins). C = N / K with N = sum sign z, K = finite active cells; C is NaN
           where K < min(5, n_active). Fit once on the real labels; the same combo in every world.
  formula  z_f = average-rank z of the formula's TOP100 signal per row (>= 20 coins). combo + f =
           (N + z_f) / (K + 1) where z_f is finite, C where it is not, NaN where C is NaN: f enters as
           a ninth equal-weight cell.
  marginal per tile: log NAV(combo + f) - log NAV(combo) on the same ruler (v4_world.tile / books:
           rank_weights -> held_path H168 -> delisting layer -> settle_multi; flat6 / cost0; lag 1).
  halves   v4_world.ab_tiles: 30 A / 30 B tiles of 336 h, alternate calendar months from 2021-01,
           >= 168 h buffer to both month edges. Training and hall of fame on A, ranking on B.
  worlds   real; null = D.null_shard (per hour one coin permutation of fwd / eligible / funding inside
           the TOP100 of the signal row; terminals untouched); p8 / p32: fwd[t] += alpha s_perp[t-1],
           s_perp = D.plant_signal (DELTA(TS_MEAN(log_quote_volume,168),168), TOP100 rank z) with C
           regressed out per hour (OLS with intercept over coins where both are finite), re-standardised;
           alpha such that s_perp alone (full book, cost0, A tiles) gains +8 / +32 bp/day. The delisting
           layer is installed after every world swap.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m2h_b5_trainer_dbg_memo as TB      # first: sys.path for m1_stage1 / okx_alpha_discovery / legacy
import m2h_verdict_dense as V
import m2h_common as M
import m2h_dense_common as D
import v4_world as X
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid

CELLS = ("T1_mom_168h", "CTREND_lite_17") + V.T2_CELLS + V.T3_CELLS
H_IC, STEP, MIN_CELLS, COV_MIN = 168, 24, 5, 0.5
TARGETS = dict(p8=8.0, p32=32.0)
MODES = ("real", "null", "p8", "p32")
BOOKS2 = (X.K10, ("full",))
A_ROOT = os.path.join(X.ROOT, "artifacts", "m2h_v1")
ALPHA_JSON = os.path.join(A_ROOT, "CR1", "CR1_ALPHA.json")
TV1_ALPHA = os.path.join(A_ROOT, "TV1_20261010", "TV1_ALPHA.json")
CORR_ORTH_MAX, PLANT_CORR_MIN, RULER_T_MIN = 0.05, 0.7, 2.0


def sha16a(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def sha12f(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]


# --------------------------------------------------------------------------- combo fit (combo_v1 copies)
def fwd_cum(fw, h=H_IC):
    lr = np.log1p(np.nan_to_num(fw, nan=0.0))
    cs = np.concatenate([np.zeros((lr.shape[0], 1)), np.cumsum(lr, axis=1)], axis=1)
    out = np.full(lr.shape, np.nan)
    T = lr.shape[1]
    out[:, :T - h + 1] = cs[:, h:] - cs[:, :T - h + 1]
    return out


def ic_rows(sig, fut, mask, hours):
    a = np.where(mask[:, hours - 1], sig[:, hours - 1], np.nan).T
    b = np.where(mask[:, hours - 1], fut[:, hours], np.nan).T
    v = D.cs_spearman_rows(a, b)
    ok = (np.isfinite(a) & np.isfinite(b)).sum(axis=1) >= D.IC_MIN_COINS
    return np.where(ok, v, np.nan)


def stat(v):
    n0 = len(v)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return dict(ic=None, ir=None, t=None, n=int(len(v)), cov=round(len(v) / max(n0, 1), 4))
    m, sd = float(v.mean()), float(v.std(ddof=1))
    return dict(ic=round(m, 5), ir=round(m / sd, 4) if sd > 0 else None,
                t=round(m / (sd / np.sqrt(len(v))), 2) if sd > 0 else None, n=int(len(v)),
                cov=round(len(v) / max(n0, 1), 4))


def fit_combo(S, fut, univ, A):
    hrs = np.concatenate([np.arange(s0, s0 + X.E, STEP) for s0 in A])
    ics = {k: stat(ic_rows(S[k], fut, univ, hrs)) for k in CELLS}
    act = [k for k in CELLS if ics[k]["ic"] is not None and ics[k]["cov"] >= COV_MIN]
    sign = {k: (1 if ics[k]["ic"] >= 0 else -1) if k in act else 0 for k in CELLS}
    N = np.zeros(univ.shape)
    K = np.zeros(univ.shape, dtype=np.int16)
    for k in act:
        z = X.zrank_rows(np.where(univ & np.isfinite(S[k]), S[k], np.nan).T).T
        f = np.isfinite(z)
        N += np.where(f, sign[k] * z, 0.0)
        K += f.astype(np.int16)
    need = min(MIN_CELLS, len(act))
    return N, K, need, dict(fit="rank IC over the A half: signal every 24 h inside every A tile, 168 h "
                                "forward, TOP100, real labels", sample_hours=int(len(hrs)), ic=ics,
                            sign=sign, active=act, need=need)


def orthogonalise(s, Cg, min_coins=D.PLANT_MIN_COINS):
    """Per hour (column): residual of s on Cg (OLS with intercept over jointly finite coins), unit std."""
    m = np.isfinite(s) & np.isfinite(Cg)
    n = m.sum(axis=0)
    nn = np.maximum(n, 1)
    mx = np.where(m, Cg, 0.0).sum(axis=0) / nn
    my = np.where(m, s, 0.0).sum(axis=0) / nn
    dx = np.where(m, Cg - mx, 0.0)
    dy = np.where(m, s - my, 0.0)
    vx = (dx * dx).sum(axis=0)
    b = np.where(vx > 0, (dx * dy).sum(axis=0) / np.where(vx > 0, vx, 1.0), 0.0)
    r = dy - b * dx
    sd = np.sqrt((r * r).sum(axis=0) / nn)
    ok = (n >= min_coins) & (sd > 0)
    return np.where(m & ok[None, :], r / np.where(sd > 0, sd, 1.0)[None, :], np.nan)
def make_vm(block_elems=1 << 25, ema_cache_max=1):
    import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vm_mod
    vm_mod.BLOCK_ELEMS = int(block_elems)
    vm_mod.EMA_KERNEL_CACHE_MAX = int(ema_cache_max)
    return vm_mod.XsTsCsStackVM(len(M.TERMINALS), windows=M.WINDOWS)


S_PERP_NPY = os.path.join(A_ROOT, "CR1", "CR1_S_PERP.npy")


def load_alpha(mode, s_perp):
    cal = json.load(open(ALPHA_JSON, encoding="utf-8"))
    if cal["s_perp_sha16"] != sha16a(s_perp):
        raise RuntimeError("s_perp differs from the calibrated one")
    return float(cal[mode]["alpha"])


# --------------------------------------------------------------------------- world
def build(mode, device, vm, log=print, alpha=None):
    """V4 TRAIN world for one mode: grid, TOP100, A/B tiles, the fixed combo (fit on A, real labels),
    the orthogonal plant s_perp, the mode's shard (+ delisting layer), per-tile combo rows and the
    combo's own tile results in both books."""
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
    fut = fwd_cum(base["fwd_wx"])
    N, K, need, fit = fit_combo(S, fut, univ, A)
    del S, fut
    Cg = np.where(K >= need, N / np.maximum(K, 1), np.nan)
    feats_pad = M.pad_feats(w.pop("feats").astype(np.float32))
    # plant: the plant formula's TOP100 rank z on every tile row (signal hours s0 .. s0+E-2)
    pb = D.plant_body(vm)
    s_raw = np.full(univ.shape, np.nan)
    for s0 in A + B:
        sl = M.tile_slices(feats_pad, [s0], device)[0]
        z = X.zrank_rows(M.window_signal(vm, pb, sl, s0, univ))
        s_raw[:, s0:s0 + X.E - M.LAG] = z[M.LAG:].T
        del sl
    s_perp = orthogonalise(s_raw, Cg)
    s_diff = None
    if os.path.exists(S_PERP_NPY):                 # the calibrated s_perp is the one every world uses
        s_cal = np.load(S_PERP_NPY)
        if s_cal.shape != s_perp.shape:
            raise RuntimeError("s_perp shape %s != %s" % (s_cal.shape, s_perp.shape))
        both = np.isfinite(s_cal) & np.isfinite(s_perp)
        d = np.abs(s_cal[both] - s_perp[both])
        s_diff = dict(max=float(d.max()) if d.size else 0.0, frac_gt_1e4=float((d > 1e-4).mean()) if d.size else 0.0,
                      finite_mismatch=int((np.isfinite(s_cal) != np.isfinite(s_perp)).sum()))
        if s_diff["frac_gt_1e4"] > 0.01 or s_diff["finite_mismatch"] > 0.01 * both.sum():
            raise RuntimeError("s_perp recomputed differs from %s: %s" % (S_PERP_NPY, s_diff))
        s_perp = s_cal
    if mode == "real":
        sh = base
    elif mode == "null":
        sh = X.install_layer(D.null_shard(base, univ, D.PLANT_SEED), w["dead"])
    else:
        a = load_alpha(mode, s_perp) if alpha is None else float(alpha)
        sh = X.install_layer(D.plant_shard(base, s_perp, a), w["dead"])
    ctx = dict(mode=mode, ts=ts, univ=univ, syms=w["syms"], dead=w["dead"], A=A, B=B, need=need, fit=fit,
               s_perp_diff=s_diff, s_perp_sha16=sha16a(s_perp),
               feats_pad=feats_pad, base=base, shard=sh, s_raw=s_raw, s_perp=s_perp, Cg=Cg, rows={}, combo={})
    Ng = np.where(K >= need, N, np.nan)
    Kf = K.astype(np.float64)
    for s0 in A + B:
        ctx["rows"][s0] = (X.tile_rows(Ng, s0), X.tile_rows(Kf, s0))
    del Kf
    set_world(ctx, sh)
    log("CR1 world %s: %d coins x %d h, A %d / B %d tiles, %d delistings, active cells %d, %.0fs"
        % (mode, len(w["syms"]), len(ts), len(A), len(B), len(w["dead"]), len(fit["active"]), time.time() - t0))
    return ctx


def set_world(ctx, sh):
    """Swap the shard and recompute the combo's own result on every tile and both books."""
    ctx["shard"] = sh
    ctx["combo"] = {}
    for s0, (Nt, Kt) in ctx["rows"].items():
        Ct = combine(Nt, Kt, ctx["need"], None)
        ctx["combo"][s0] = {X.bk(b): X.tile(X.book_sig(Ct, sh, s0, b), sh, s0) for b in BOOKS2}
    return ctx


def combine(Nt, Kt, need, zf):
    """Combo rows C = N / K (NaN where K < need); with zf: (N + zf) / (K + 1) where zf is finite."""
    k = np.nan_to_num(Kt, nan=0.0)
    ok = k >= max(need, 1)
    out = np.where(ok, Nt / np.where(ok, k, 1.0), np.nan)
    if zf is None:
        return out
    f = ok & np.isfinite(zf)
    return np.where(f, (Nt + np.where(f, zf, 0.0)) / (k + 1.0), out)


def tile_sigs(ctx, vm, body, starts, slices):
    """GPU part: the formula's TOP100 tile signals (E, A) float32 and the live fraction. Raises
    M.ZeroCoverage on the probe rows exactly as M.score_formula / D.score_dense do."""
    sh, univ = ctx["shard"], ctx["univ"]
    sigs, lf = [], []
    for s0, sl in zip(starts, slices):
        sigT = M.window_signal(vm, body, sl, s0, univ)
        el = sh["eligible"][:, s0:s0 + X.E].T
        live = np.isfinite(sigT) & el
        probe = [t for t in [M.LAG] + list(range(X.E // 4, X.E, X.E // 4)) if t < X.E]
        pf = [float(live[t].sum() / max(el[t].sum(), 1)) for t in probe]
        if min(pf, default=0.0) < M.MIN_COVERAGE:
            raise M.ZeroCoverage(f"ZERO_COVERAGE_MIN_{min(pf, default=0.0):.2f}")
        lf.append(float(np.mean(pf)))
        sigs.append(sigT)
    return sigs, float(np.mean(lf))


def marginal_from_sigs(rc, sigs, starts, books=(("full",),), corr_tiles=0):
    """CPU part, per tile: log NAV(combo + f) - log NAV(combo), same ruler, same book. rc is a ctx or a
    worker's cache view (shard, rows, combo, need, s_perp). CandidateLocalInvalid when f has no z."""
    sh = rc["shard"]
    out = {X.bk(b): dict(net=[], gross=[], turn=[]) for b in books}
    corr, nz = [], 0
    for i, (s0, sigT) in enumerate(zip(starts, sigs)):
        zf = X.zrank_rows(np.asarray(sigT, dtype=np.float64))
        nz += int(np.isfinite(zf).any(axis=1).sum())
        Nt, Kt = rc["rows"][s0]
        comb = combine(Nt, Kt, rc["need"], zf)
        for b in books:
            k = X.bk(b)
            r = X.tile(X.book_sig(comb, sh, s0, b), sh, s0)
            c = rc["combo"][s0][k]
            out[k]["net"].append(float(r["ln_net"] - c["ln_net"]))
            out[k]["gross"].append(float(r["ln_gross"] - c["ln_gross"]))
            out[k]["turn"].append(float(r["turn"]))
        if i < corr_tiles:
            corr.append(D.corr_with_s(np.asarray(sigT, dtype=np.float64), rc["s_perp"], s0))
    if nz == 0:
        raise CandidateLocalInvalid("CONSTCS_NO_Z")
    return dict(books=out, corr=float(np.mean(corr)) if corr else float("nan"))


def marginal(ctx, vm, body, starts, slices, books=(("full",),), corr_tiles=0):
    sigs, lf = tile_sigs(ctx, vm, body, starts, slices)
    res = marginal_from_sigs(ctx, sigs, starts, books, corr_tiles)
    res["live_frac"] = lf
    return res


# --------------------------------------------------------------------------- worker pool (CPU ruler)
CACHE_KEYS = ("eligible", "fwd_wx", "fi", "fr")


def write_cache(ctx, cdir):
    """What a worker needs, as .npy (mmap-read) + one pickle; float64 kept (bit-identical results)."""
    import pickle
    os.makedirs(cdir, exist_ok=True)
    for k in CACHE_KEYS:
        np.save(os.path.join(cdir, k + ".npy"), ctx["shard"][k])
    np.save(os.path.join(cdir, "s_perp.npy"), ctx["s_perp"])
    keys = sorted(ctx["rows"])
    np.save(os.path.join(cdir, "rows.npy"), np.stack([np.stack(ctx["rows"][s]) for s in keys]))
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
    _W.update(shard=sh, need=meta["need"], combo=meta["combo"],
              s_perp=np.load(os.path.join(cdir, "s_perp.npy"), mmap_mode="c"),
              rows={s: (np.asarray(rows[i, 0]), np.asarray(rows[i, 1])) for i, s in enumerate(meta["keys"])})
    warnings.simplefilter("ignore", RuntimeWarning)


def pack(sigs):
    """(E, A) tile signals -> only the columns with a finite entry (TOP100 ~ 1/5 of the coins)."""
    out = []
    for s in sigs:
        cols = np.flatnonzero(np.isfinite(s).any(axis=0))
        out.append((s.shape, cols, np.ascontiguousarray(s[:, cols])))
    return out


def unpack(packed):
    out = []
    for shp, cols, v in packed:
        s = np.full(shp, np.nan, dtype=v.dtype)
        s[:, cols] = v
        out.append(s)
    return out


def worker_task(task):
    key, packed, starts, books, corr_tiles = task
    try:
        return key, marginal_from_sigs(_W, unpack(packed), starts, books, corr_tiles)
    except CandidateLocalInvalid as e:
        return key, "INVALID:" + str(e)


def bpd(ln):
    return X.bpd(ln)


def tstat(v):
    v = np.asarray(v, dtype=np.float64)
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return float(v.mean() / (sd / np.sqrt(len(v)))) if sd > 0 else float("nan")
