"""XS_SEL1 OKX availability, the two order items (XS_SEL1_ORDER 五). Replaces sel1_okx_pool.py, which settled on the
OKX grid's own months (2021-.. table on a 2024+ grid) and had no features to run the VM on. CPU only; no OOS (both grids
stop before 2026-07-01, checked on load). New file; sel1_common / v4_world are imported, never modified.

  signal     always the V4 world (the trainer's features, chunks, beta, BTC state, delisting layer); real mode.
  variants   v4       the trainer's book unchanged (TOP100, not BTC, finite beta), V4 fwd / funding.
             okx_sub  item 1: the same, selection restricted to coins present in the OKX 205 pool (coin level).
             okx_nat  item 2: okx_sub's mask & OKX bar present at the signal hour, settled on OKX native 1h close
                      (fwd) and funding (OKX_SEL1_1H_V1.npz: partitions <= 20260701, rows < 07-01 UTC), winsor 0 +
                      the V4 delisting layer, the BTC hedge leg from OKX too.
  worlds     train    V4_TRAIN_GRID; v4 on 2021-01..2025-12 and, beside every variant, on 2024-01..2025-12 (OKX's span).
             dev      V4_DEV2026_HOURLY_PANEL, months 2026-01..06; needs --frozen, written once, never overwritten.
  entries    --frozen SEL1_FROZEN_V1 (entries: body, h, st, dr, expr) or --plant (the PC plant at h 168, all, both:
             a plumbing check, train only).
  selfcheck  (train) own builder == sel1_common.build("real") months / chunks / mask / shard; the OKX shard path fed
             the V4 bars == the trainer's shard bit for bit; OKX vs V4 hourly fwd correlation on shared 2024-25 cells.
    python -I sel1_okx_pool2.py --world train|dev (--frozen F | --plant) [--out OUT] [--device cpu]
"""
from __future__ import annotations

import argparse
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
import torch
import sel1_common as S
import v4_world as X
import m2h_common as M
import m2h_dense_common as D
import cr1_common as Q
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid

NPZ = os.path.join(HERE, "..", "b5_scratch", "OKX_SEL1_1H_V1.npz")
RECEIPT = os.path.join(HERE, "..", "b5_scratch", "OKX_SEL1_1H_V1.json")
OKX_FIRST = "2024-01"
DEV_FIRST, DEV_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-07-01")


def log(m):
    print(m, flush=True)


def months_of(ts, first, end):
    """sel1_common.months on an arbitrary span (A = even index, as there)."""
    out = []
    for k, m in enumerate(pd.date_range(first, end - pd.Timedelta(hours=1), freq="MS")):
        m0 = int(ts.get_loc(m))
        nx = m + pd.offsets.MonthBegin(1)
        m1 = int(ts.get_loc(nx)) if nx <= ts[-1] else len(ts)
        out.append(dict(m0=m0, m1=m1, half="A" if k % 2 == 0 else "B", year=int(m.year), month=str(m)[:7]))
    return out


def world(path, first, end, device):
    """sel1_common.build("real") on any V4 grid file and month span."""
    t0 = time.time()
    w = X.load(path)
    X.world(w, feats=True)
    ts, univ, syms = w["ts"], w["univ"], w["syms"]
    A, T = univ.shape
    bi = syms.index("BTCUSDT")
    close, fr_raw = w["g"]["trade_close"], w["g"]["funding_rate"]
    beta = S.rolling_beta(close, bi)
    state = S.btc_state(close[bi])
    del w["g"]
    feats = w.pop("feats")
    base = S.no_winsor_shard(w["shard"], w["dead"])
    if not np.array_equal(base["fwd_wx"], w["shard"]["fwd_x"], equal_nan=True):
        raise RuntimeError("winsor 0 path != fwd_x")
    mask = univ & np.isfinite(beta)
    mask[bi] = False
    Larr = np.full(A, T + 10, np.int64)
    for c, L in w["dead"].items():
        Larr[c] = L
    ctx = dict(mode="real", ts=ts, syms=syms, univ=univ, mask=mask, bi=bi, beta=beta, state=state, Larr=Larr,
               dead=w["dead"], months=months_of(ts, first, end), shard=base, alpha=None, zp=None,
               feats_gpu=torch.from_numpy(np.ascontiguousarray(feats.astype(np.float32))).to(device), offsets=None)
    del feats
    ctx["cover_rows"] = np.concatenate([np.arange(m["m0"], m["m1"]) for m in ctx["months"]])
    ctx["chunks"] = S.year_chunks(ts, first, pd.Timestamp("%d-01-01" % ((end - pd.Timedelta(hours=1)).year + 1)))
    ctx["_close"], ctx["_fr"] = close, fr_raw
    log("world %s: %d coins x %d h, %d months %s..%s, %d delistings, %.0fs"
        % (os.path.basename(path), A, T, len(ctx["months"]), ctx["months"][0]["month"], ctx["months"][-1]["month"],
           len(w["dead"]), time.time() - t0))
    return ctx


def okx_grid(ctx):
    """OKX close / funding on the V4 world's (syms, ts); NaN where OKX has no bar."""
    z = np.load(NPZ, allow_pickle=False)
    oh = pd.DatetimeIndex(z["hours"].astype("datetime64[ns]"))
    if oh.max() >= M.OOS_FIRST:
        raise SystemExit("OKX grid reaches OOS")
    rows = pd.Index(ctx["syms"]).get_indexer([str(s) for s in z["symbols"]])
    col = pd.DatetimeIndex(ctx["ts"]).get_indexer(oh)
    ok = col >= 0
    A, T = ctx["mask"].shape
    close, fr = np.full((A, T), np.nan), np.full((A, T), np.nan)
    for j, r in enumerate(rows):
        if r >= 0:
            close[r, col[ok]] = z["close"][j][ok]
            fr[r, col[ok]] = z["funding_rate"][j][ok]
    in_okx = np.zeros(A, bool)
    in_okx[rows[rows >= 0]] = True
    return close, fr, in_okx, dict(okx_symbols=int(len(rows)), mapped=int((rows >= 0).sum()), okx_hours=[str(oh.min()), str(oh.max())])


def okx_shard(ctx, close, fr_raw):
    fwd = X.C.forward_returns(close, None)
    sh = dict(fwd=fwd, eligible=np.isfinite(fwd), fi=np.isfinite(fr_raw), fr=np.where(np.isfinite(fr_raw), fr_raw, 0.0))
    return S.no_winsor_shard(sh, ctx["dead"])


def variant(ctx, mask=None, shard=None, since=None):
    v = dict(ctx)
    if mask is not None:
        v["mask"] = mask
    if shard is not None:
        v["shard"] = shard
    if since is not None:
        v["months"] = [m for m in ctx["months"] if m["month"] >= since]
        v["cover_rows"] = np.concatenate([np.arange(m["m0"], m["m1"]) for m in v["months"]])
    return v


def summ(rows):
    if not rows:
        return None
    return dict(months=len(rows), bpd_cost0=S.bpd(rows), bpd_net=S.bpd(rows, "net"), t_cost0=S.reward(rows, "t"),
                turn_day=sum(r["turn"] for r in rows) / sum(r["days"] for r in rows),
                n_pos=float(np.mean([r["n_pos"] for r in rows])))


def settle(e, raw, v, halves=("A", "B")):
    """One entry on one variant: mask, select, score_sel; -> A / B / all summaries + per-month cost0, net (bp/day)."""
    try:
        sg = S.mask_signal(v, raw)
    except S.Invalid as x:
        return dict(invalid=str(x))
    rows = S.score_sel(S.select_months(sg, v, halves), v, e["h"], e["st"], e["dr"], halves)
    per = [dict(month=r["month"], cost0=1e4 * r["cost0"] / r["days"], net=1e4 * r["net"] / r["days"], n_pos=r["n_pos"]) for r in rows]
    return dict(A=summ([r for r in rows if r["half"] == "A"]), B=summ([r for r in rows if r["half"] == "B"]),
                all=summ(rows), per_month=per)


def selfcheck(ctx, vm, close):
    """Builder == sel1_common.build("real"); okx_nat plumbing; OKX vs V4 fwd agreement (2024-25 cells, no stats beyond)."""
    out = {}
    ref = S.build("real", torch.device("cpu"), vm, log=lambda m: None)
    out["builder_eq_sel1_build"] = bool(ref["months"] == ctx["months"] and ref["chunks"] == ctx["chunks"]
                                        and np.array_equal(ref["mask"], ctx["mask"])
                                        and np.array_equal(ref["shard"]["fwd_wx"], ctx["shard"]["fwd_wx"], equal_nan=True))
    del ref
    own = okx_shard(ctx, ctx["_close"], ctx["_fr"])                # the OKX shard path fed the V4 bars == the trainer's
    out["okx_shard_path_eq_v4_on_v4_bars"] = bool(all(np.array_equal(own[k], ctx["shard"][k], equal_nan=True)
                                                      for k in ("fwd_wx", "fi", "fr", "eligible")))
    i0 =int(ctx["ts"].get_loc(pd.Timestamp(OKX_FIRST + "-01")))
    fo = X.C.forward_returns(close, None)[:, i0:]
    fv = ctx["shard"]["fwd"][:, i0:]
    both = np.isfinite(fo) & np.isfinite(fv)
    rho = [float(np.corrcoef(fo[a, both[a]], fv[a, both[a]])[0, 1]) for a in range(fo.shape[0]) if both[a].sum() >= 720]
    out["fwd_corr_okx_v4"] = dict(coins=len(rho), median=float(np.median(rho)) if rho else None,
                                  p10=float(np.percentile(rho, 10)) if rho else None)
    out["okx_cells_vs_v4_univ_2024_25"] = float((np.isfinite(close[:, i0:]) & ctx["univ"][:, i0:]).sum()
                                               / max(ctx["univ"][:, i0:].sum(), 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=("train", "dev"), required=True)
    ap.add_argument("--frozen")
    ap.add_argument("--plant", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if bool(a.frozen) == bool(a.plant):
        raise SystemExit("exactly one of --frozen / --plant")
    if a.world == "dev" and not a.frozen:
        raise SystemExit("dev needs --frozen")
    out_path = a.out or os.path.join(S.OUT_ROOT, "SEL1_OKX_%s_%s.json" % (a.world.upper(), "PLANT" if a.plant else "FROZEN"))
    if os.path.exists(out_path):
        raise SystemExit("output exists: %s" % out_path)
    t0 = time.time()
    torch.set_num_threads(4)
    vm = Q.make_vm()
    if a.plant:
        fz = dict(k=S.K, entries=[dict(expr="PLANT DELTA(TS_MEAN(log_quote_volume,168),168)", body=D.plant_body(vm), h=168, st=0, dr=0)])
    else:
        fz = json.load(open(a.frozen, encoding="utf-8"))
        if fz.get("schema") != "SEL1_FROZEN_V1":
            raise SystemExit("frozen schema not SEL1_FROZEN_V1: %s" % fz.get("schema"))
    S.K = int(fz.get("k", 5))
    if a.world == "train":
        ctx = world(X.GRID, S.TRAIN_FIRST, S.TRAIN_END, torch.device(a.device))
    else:
        ctx = world(X.DEVZ, DEV_FIRST, DEV_END, torch.device(a.device))
    close, fr, in_okx, info = okx_grid(ctx)
    m_sub = ctx["mask"] & in_okx[:, None]
    m_nat = m_sub & np.isfinite(close)
    osh = okx_shard(ctx, close, fr)
    since = OKX_FIRST if a.world == "train" else None
    V = dict(v4=variant(ctx, since=since), okx_sub=variant(ctx, mask=m_sub, since=since),
             okx_nat=variant(ctx, mask=m_nat, shard=osh, since=since))
    if a.world == "train":
        V = dict(v4_2021_25=variant(ctx), **V)
    info.update(okx_sub_coins=int(in_okx.sum()), okx_sub_mask_cells=int(m_sub[:, ctx["cover_rows"]].sum()),
                v4_mask_cells=int(ctx["mask"][:, ctx["cover_rows"]].sum()), okx_nat_mask_cells=int(m_nat[:, ctx["cover_rows"]].sum()))
    rep = dict(schema="SEL1_OKX_V2", world=a.world, frozen=a.frozen, frozen_sha16=S.sha16f(a.frozen) if a.frozen else None,
               okx_npz_sha16=S.sha16f(NPZ), okx_receipt=json.load(open(RECEIPT, encoding="utf-8"))["red_line"], okx=info,
               k=S.K, entries=[], oos_read=False, dev_read=a.world == "dev")
    if a.world == "train":
        rep["selfcheck"] = selfcheck(ctx, vm, close)
    for k, e in enumerate(fz["entries"]):
        try:
            raw = S.raw_signal(ctx, vm, tuple(e["body"]))
        except (S.Invalid, CandidateLocalInvalid, M.ZeroCoverage) as x:
            rep["entries"].append(dict(expr=e.get("expr"), invalid=str(x)))
            continue
        r = dict(expr=e.get("expr"), h=e["h"], state=S.STATES[e["st"]], dir=S.DIRS[e["dr"]],
                 **{n: settle(e, raw, v) for n, v in V.items()})
        rep["entries"].append(r)
        if a.world == "train":
            log("[%d/%d] %s | B t cost0: v4_21_25 %s v4 %s okx_sub %s okx_nat %s" % (
                k + 1, len(fz["entries"]), e.get("expr"),
                *[("%.2f" % r[n]["B"]["t_cost0"]) if r[n].get("B") else r[n].get("invalid") for n in V]))
        else:
            log("[%d/%d] %s done" % (k + 1, len(fz["entries"]), e.get("expr")))
        del raw
    rep["seconds"] = round(time.time() - t0, 1)
    json.dump(rep, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    log("-> %s | selfcheck %s | %.0fs" % (out_path, json.dumps(rep.get("selfcheck"), default=float), rep["seconds"]))


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
