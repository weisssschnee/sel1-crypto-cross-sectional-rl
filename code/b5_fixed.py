"""BASIN5 fixed-tile scorer (merged plan, user-approved 2026-10-13): step 1 ruler check (G0/G1/G2)
and the hall-of-fame freeze both score formulas on the same 50 fixed TRAIN tiles.

The parent writes one TRAIN world cache (B5_FIXED_CACHE: padded float32 terminals, TOP100
universe, the Binance shard arrays, the TV1 plant s, the exposures of the neutralised column);
spawn workers read it back and score one formula per task, CPU only:

  per tile and world   full-pool book (the ruler's rank book) gross / net log NAV (cost0 / flat6),
                       conc K10 book gross / net (m2h_conc_weights, CONC_POOL.json), conc turnover
  per tile             corr with s (D.corr_with_s; world-free, meaningful in the plant world only)
  neutral (optional)   the signal residualised per hour on rank(size), rank(log price) and beta,
                       then the full-pool gross; size = log 720 h quote volume (the universe's
                       own rolling sum), beta = 720 h trailing beta to the equal-weight TOP100
                       return; all from data up to the signal hour

Path = b5_round.score_both (window_signal, coverage probe, R.score_window), not modified.
Worlds: real; p8 / p32 from TV1_ALPHA.json; p16 from B5_ALPHA_P16.json (calibrate_p16, the
tv1_selftest check-5 bisection at +16 bp/day); null (D.null_shard). TRAIN only, no DEV, no OOS.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem"
A_ROOT = os.path.join(ROOT, "artifacts", "m2h_v1")
CACHE_DIR = os.path.join(A_ROOT, "B5_FIXED_CACHE")
TV1_ALPHA = os.path.join(A_ROOT, "TV1_20261010", "TV1_ALPHA.json")
P16_ALPHA = os.path.join(A_ROOT, "B5_GATE1", "B5_ALPHA_P16.json")
HOLD = 168
K = 10
EXPO_WIN, EXPO_MIN = 720, 360


def sha16(a):
    return hashlib.sha256(np.ascontiguousarray(np.nan_to_num(a, nan=-9e9)).tobytes()).hexdigest()[:16]


def _imports():
    import m2h_b5_trainer as TR          # sys.path for m1_stage1 / okx_alpha_discovery / legacy roots
    import m2h_common as M
    import m2h_dense_common as D
    import m2h_conc_weights as W
    import m1_ruler as R
    return TR, M, D, W, R


# --------------------------------------------------------------------------- cache
def build_cache(cache_dir=CACHE_DIR):
    """Parent only. Writes the arrays every worker reads; returns the manifest."""
    import pandas as pd
    TR, M, D, W, R = _imports()
    os.makedirs(cache_dir, exist_ok=True)
    t0 = time.time()
    syms, ts, grids = M.load_train_grids(TR.PANEL)
    w = M.build_world(grids)
    feats, univ, sh = w["feats"], w["univ"], w["shard"]
    starts = M.tile_starts(feats.shape[2])
    assert len(starts) == 50, len(starts)
    s = np.load(TR.S_FILE)
    if s.shape != sh["fwd"].shape:
        raise RuntimeError("s shape %s != fwd %s" % (s.shape, sh["fwd"].shape))
    # exposures for the neutralised column, all trailing (data up to the hour itself)
    qv = grids["trade_quote_volume"]
    size = np.log(pd.DataFrame(qv.T).rolling(EXPO_WIN, min_periods=EXPO_MIN).sum().to_numpy().T)
    close = grids["trade_close"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.log(close[:, 1:] / close[:, :-1])
    r = np.concatenate([np.full((r.shape[0], 1), np.nan), r], axis=1)
    rm = np.where(univ, r, np.nan)
    mkt = np.nanmean(np.where(np.isfinite(rm), rm, np.nan), axis=0)
    df = pd.DataFrame(r.T)
    ms = pd.Series(mkt)
    cov = df.rolling(EXPO_WIN, min_periods=EXPO_MIN).cov(ms)
    var = ms.rolling(EXPO_WIN, min_periods=EXPO_MIN).var()
    beta = (cov.div(var, axis=0)).to_numpy().T
    arrs = dict(feats_pad=M.pad_feats(feats).astype(np.float32), univ=univ, fwd=sh["fwd"],
                eligible=sh["eligible"], fi=sh["fi"], fr=sh["fr"], s=s,
                size=size, lprice=feats[:, 0, :], beta=beta)
    for k, v in arrs.items():
        np.save(os.path.join(cache_dir, k + ".npy"), v)
    man = dict(schema="B5_FIXED_CACHE_V1", panel=TR.PANEL, symbols=list(syms), hours=int(feats.shape[2]),
               first_hour=str(ts[0]), last_hour=str(ts[-1]), starts=[int(x) for x in starts],
               s_file=TR.S_FILE, s_sha16=sha16(s), fwd_sha16=sha16(sh["fwd"]),
               feats_sha16=sha16(feats), shapes={k: list(v.shape) for k, v in arrs.items()},
               expo=dict(size="log %d h rolling quote volume (min %d)" % (EXPO_WIN, EXPO_MIN),
                         lprice="log close", beta="%d h trailing beta to the equal-weight TOP100 "
                                                  "hourly log return (min %d)" % (EXPO_WIN, EXPO_MIN)),
               seconds=round(time.time() - t0, 1), written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               dev_read=False, oos_read=False)
    if str(ts[-1]) >= "2026-01-01":
        raise RuntimeError("TRAIN cache reaches DEV")
    json.dump(man, open(os.path.join(cache_dir, "CACHE_MANIFEST.json"), "w", encoding="utf-8"), indent=1)
    return man


def load_cache(cache_dir=CACHE_DIR, mmap=True):
    man = json.load(open(os.path.join(cache_dir, "CACHE_MANIFEST.json"), encoding="utf-8"))
    mode = "r" if mmap else None
    arr = {k: np.load(os.path.join(cache_dir, k + ".npy"), mmap_mode=mode)
           for k in ("feats_pad", "univ", "fwd", "eligible", "fi", "fr", "s", "size", "lprice", "beta")}
    return man, arr


def alpha_of(world):
    if world in ("p8", "p32"):
        cal = json.load(open(TV1_ALPHA, encoding="utf-8"))[world]
    elif world == "p16":
        cal = json.load(open(P16_ALPHA, encoding="utf-8"))["p16"]
    else:
        raise KeyError(world)
    return cal


def make_shard(world, arr):
    """The world's ruler shard, as m2h_plant.install builds it (plant on fwd, prepare_shard)."""
    TR, M, D, W, R = _imports()
    base = dict(fwd=np.array(arr["fwd"]), eligible=np.array(arr["eligible"]), fi=np.array(arr["fi"]),
                fr=np.array(arr["fr"]))
    if world == "real":
        R.prepare_shard(base, R.WINSOR_PCT)
        return base
    if world == "null":
        return D.null_shard(base, np.array(arr["univ"]), D.PLANT_SEED)
    cal = alpha_of(world)
    s = np.array(arr["s"])
    if sha16(s) != cal["s_sha16"]:
        raise RuntimeError("s differs from the calibrated one (%s)" % world)
    return D.plant_shard(base, s, cal["alpha"])


def make_vm():
    import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vm_mod
    TR, M, D, W, R = _imports()
    vm_mod.BLOCK_ELEMS = 1 << 16          # CPU: 5x faster than the GPU's 1 << 25; bit-identical (b5_block_eq)
    vm_mod.EMA_KERNEL_CACHE_MAX = 1
    return vm_mod.XsTsCsStackVM(len(M.TERMINALS), windows=M.WINDOWS)


def tile_slices(arr, starts):
    import torch
    TR, M, D, W, R = _imports()
    fp = arr["feats_pad"]
    out = []
    for s0 in starts:
        lo, hi = s0 - M.LOOKBACK, s0 + M.LAG + M.EPISODE
        out.append(torch.from_numpy(np.ascontiguousarray(fp[:, :, lo:hi])))
    return out


# --------------------------------------------------------------------------- p16
def calibrate_p16(cache_dir=CACHE_DIR, target=16.0):
    """tv1_selftest check 5 at +16 bp/day: bisection on alpha so that the plant formula's H168
    full-pool gross on the 50 TRAIN tiles rises by target (within 2 %); p8 / p32 re-measured
    with their TV1 alpha as a check of this path."""
    TR, M, D, W, R = _imports()
    from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod
    man, arr = load_cache(cache_dir)
    starts = man["starts"]
    vm = make_vm()
    sl = tile_slices(arr, starts)
    univ = np.array(arr["univ"])
    pb = D.plant_body(vm)
    psig = [M.window_signal(vm, pb, x, s0, univ) for s0, x in zip(starts, sl)]

    def gross(shard):
        v = []
        for s0, sigT in zip(starts, psig):
            lg, _ = R.score_window(sigT, shard, s0, M.EPISODE, [HOLD], policy_mod, costs=("cost0",),
                                   variants=("win",), band=R.BAND, nav_usd=R.NAV_USD)[HOLD]["win"]
            v.append(D.bpd(lg["cost0"]))
        return float(np.mean(v))
    base = make_shard("real", arr)
    g0 = gross(base)
    s = np.array(arr["s"])
    check = {}
    for k in ("p8", "p32"):
        cal = alpha_of(k)
        check[k] = dict(alpha=cal["alpha"], tv1_lift=cal["lift_bpd"],
                        lift_here=round(gross(D.plant_shard(base, s, cal["alpha"])) - g0, 4))
    a, hist = 2.0 * alpha_of("p8")["alpha"], []
    for _ in range(10):
        L = gross(D.plant_shard(base, s, a)) - g0
        hist.append((a, round(L, 4)))
        if abs(L - target) <= 0.02 * target:
            break
        a = a * target / L if L > 0 else a * 4.0
    a, L = hist[-1]
    out = dict(p16=dict(alpha=a, target_bpd=target, lift_bpd=L, iterations=len(hist), history=hist,
                        s_sha16=sha16(s), expr=vm.describe(pb, list(M.TERMINALS))),
               base_gross_bpd=round(g0, 4), check_tv1=check, ok=abs(L - target) <= 0.10 * target,
               written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    os.makedirs(os.path.dirname(P16_ALPHA), exist_ok=True)
    json.dump(out, open(P16_ALPHA, "w", encoding="utf-8"), indent=1)
    return out


# --------------------------------------------------------------------------- worker
_G = {}


def init_worker(cache_dir, threads=1):
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    import torch
    torch.set_num_threads(int(threads))
    TR, M, D, W, R = _imports()
    from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid
    man, arr = load_cache(cache_dir)
    _G.update(man=man, arr=arr, starts=man["starts"], vm=make_vm(), univ=np.array(arr["univ"]),
              s=np.array(arr["s"]), slices=tile_slices(arr, man["starts"]), shards={},
              pool=W.pool_cols(man["symbols"], os.path.join(HERE, "CONC_POOL.json")),
              policy=policy_mod, invalid=CandidateLocalInvalid)


def _shard(world):
    if world not in _G["shards"]:
        _G["shards"][world] = make_shard(world, _G["arr"])
    return _G["shards"][world]


def _neutralise(sigT, s0):
    """Per hour: rank(signal) residual on [1, rank(size), rank(log price), beta]; NaN elsewhere."""
    import pandas as pd
    TR, M, D, W, R = _imports()
    E = sigT.shape[0]
    cols = np.arange(E) + s0 - M.LAG
    X = [np.asarray(_G["arr"][k][:, cols], dtype=np.float64).T for k in ("size", "lprice", "beta")]
    out = np.full_like(sigT, np.nan, dtype=np.float64)
    for t in range(E):
        y = sigT[t]
        m = np.isfinite(y) & np.isfinite(X[0][t]) & np.isfinite(X[1][t]) & np.isfinite(X[2][t])
        if m.sum() < 20:
            continue
        rk = lambda v: pd.Series(v).rank().to_numpy() - (m.sum() + 1) / 2.0
        b = np.clip(X[2][t][m], -5, 5)
        A_ = np.column_stack([np.ones(m.sum()), rk(X[0][t][m]), rk(X[1][t][m]), b - b.mean()])
        yy = rk(y[m])
        coef, *_ = np.linalg.lstsq(A_, yy, rcond=None)
        out[t, m] = yy - A_ @ coef
    return out


def score_task(task):
    """task: dict(id, body, worlds, conc=True, neutral=False). Per-tile lists per world."""
    TR, M, D, W, R = _imports()
    E = M.EPISODE
    vm, univ, s, pool, pol = _G["vm"], _G["univ"], _G["s"], _G["pool"], _G["policy"]
    body = [int(x) for x in task["body"]]
    worlds = list(task["worlds"])
    out = dict(id=task["id"], ok=False, reason=None, corr=[],
               w={wn: dict(fg=[], fn=[], cg=[], cn=[], cturn=[]) for wn in worlds})
    if task.get("neutral"):
        out["neutral"] = {wn: [] for wn in worlds}
    t0 = time.time()
    if task.get("corr_only"):                  # tv1_round.curve's measurement: no probe, no ruler
        try:
            for k in task["tiles"]:
                s0 = _G["starts"][k]
                sigT = M.window_signal(vm, body, _G["slices"][k], s0, univ)
                out["corr"].append(float(D.corr_with_s(sigT, s, s0)))
        except (M.ZeroCoverage, _G["invalid"]) as exc:
            out["reason"] = str(exc)[:60]
            return out
        out["ok"] = True
        return out
    try:
        for s0, sl in zip(_G["starts"], _G["slices"]):
            sigT = M.window_signal(vm, body, sl, s0, univ)
            el0 = _shard(worlds[0])["eligible"][:, s0:s0 + E].T
            live = np.isfinite(sigT) & el0
            probe_t = [t for t in [M.LAG] + list(range(E // 4, E, E // 4)) if t < E]
            if min(float(live[t].sum() / max(el0[t].sum(), 1)) for t in probe_t) < M.MIN_COVERAGE:
                out["reason"] = "ZERO_COVERAGE"
                return out
            out["corr"].append(float(D.corr_with_s(sigT, s, s0)))
            nsig = _neutralise(sigT, s0) if task.get("neutral") else None
            for wn in worlds:
                sh = _shard(wn)
                el = sh["eligible"][:, s0:s0 + E].T
                full = R.score_window(sigT, sh, s0, E, [HOLD], pol, costs=M.COSTS, variants=("win",),
                                      band=R.BAND, nav_usd=R.NAV_USD)[HOLD]["win"][0]
                rec = out["w"][wn]
                rec["fg"].append(float(full["cost0"]))
                rec["fn"].append(float(full["flat6"]))
                if task.get("conc", True):
                    cl, cp = R.score_window(W.conc_signal(sigT, el, pool, K), sh, s0, E, [HOLD], pol,
                                            costs=M.COSTS, variants=("win",), band=R.BAND,
                                            nav_usd=R.NAV_USD)[HOLD]["win"]
                    rec["cg"].append(float(cl["cost0"]))
                    rec["cn"].append(float(cl["flat6"]))
                    rec["cturn"].append(float(cp["turnover"] / (E / 24.0)))
                if nsig is not None:
                    nf = R.score_window(nsig, sh, s0, E, [HOLD], pol, costs=("cost0",), variants=("win",),
                                        band=R.BAND, nav_usd=R.NAV_USD)[HOLD]["win"][0]
                    out["neutral"][wn].append(float(nf["cost0"]))
    except (M.ZeroCoverage, _G["invalid"]) as exc:
        out["reason"] = str(exc)[:60]
        return out
    vals = [v for wn in worlds for k in ("fg", "fn", "cg", "cn") for v in out["w"][wn][k]]
    if not all(np.isfinite(vals)):
        out["reason"] = "NONFINITE"
        return out
    out["ok"] = True
    out["seconds"] = round(time.time() - t0, 2)
    return out


def run_pool(tasks, cache_dir=CACHE_DIR, workers=12, threads=1, log=print):
    import multiprocessing as mp
    res = {}
    if not tasks:
        return res
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(threads)          # spawn children read these when numpy / torch load
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ctx.Pool(min(workers, len(tasks)), initializer=init_worker, initargs=(cache_dir, threads)) as p:
        for i, r in enumerate(p.imap_unordered(score_task, tasks, chunksize=1)):
            res[r["id"]] = r
            if (i + 1) % 25 == 0 or i + 1 == len(tasks):
                log("  scored %d/%d %.0fs" % (i + 1, len(tasks), time.time() - t0))
    return res


# --------------------------------------------------------------------------- stats
def tstat(v):
    v = np.asarray(v, dtype=np.float64)
    sd = v.std(ddof=1)
    return float(v.mean() / (sd / np.sqrt(len(v)))) if sd > 0 else float("nan")


def gross_t(v):
    """The training reward's statistic on the given tiles: mean / sd (ddof 1)."""
    v = np.asarray(v, dtype=np.float64)
    sd = v.std(ddof=1)
    return float(v.mean() / sd) if sd > 0 else float("nan")


def bpd(log_nav):
    return 1e4 * float(log_nav) / (336 / 24.0)
