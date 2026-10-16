"""XS_SEL1 OKX pool settlement (order task 3a: re-settle FROZEN on OKX-tradable 205 coin pool). CPU only, read
only; no DEV vs OOS distinction; OKX_SEL1_1H_V1.npz already enforces the OOS red line. New file; sel1_common
and v4_world are imported, never modified.

  grid       OKX_SEL1_1H_V1.npz (205 OKX swaps, 2023-12-31T16:00..2026-06-30T23:00, rows at/after 2026-07-01T00:00Z
             dropped). Universe = all 205 coins at all hours (no TOP100 filter); beta = the pool's BTC, refusal
             if missing; months = 2021-01..2026-06 (the grid's own months(ts)), A half = odd months.
  candidate  SEL1_FROZEN_V1.json entries (formula + h/st/dr). Each is run through sel1_common.score_sel (the same
             compact_desired -> held_compact -> settle_window as TRAIN), per month (A, B, all 60 months).
  result     per entry: A/B rows, bpd, t, net/gross/turn/n_pos/days; bins = bins over the 60 months, A, B and all.
    py -I sel1_okx_pool.py FROZEN_JSON OUT_JSON [--top N] [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import torch
import sel1_common as S
import v4_world as X
import m1_common as C
import m1_ruler as R
import cr1_common as Q
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid


NPZ = os.path.join(HERE, "..", "b5_scratch", "OKX_SEL1_1H_V1.npz")
RECEIPT = os.path.join(HERE, "..", "b5_scratch", "OKX_SEL1_1H_V1.json")


def log(m):
    print(m, flush=True)


def okx_world(device):
    """Build a sel1_common world but with OKX pool grid (no plant, no null shift, all 205 coins in univ)."""
    t0 = time.time()
    z = np.load(NPZ, allow_pickle=False)
    ts = pd.DatetimeIndex(z["hours"].astype("datetime64[ns]"))
    if ts.max() >= pd.Timestamp("2026-07-01", tz="UTC"):
        raise SystemExit("OKX grid reaches OOS")
    syms = [str(s) for s in z["symbols"]]
    A, T = len(syms), len(ts)
    if "BTCUSDT" not in syms:
        raise SystemExit("BTC not in OKX pool")
    bi = syms.index("BTCUSDT")
    close = z["close"]
    fr_raw = z["funding_rate"]
    fwd = C.forward_returns(close, None)
    sh = dict(fwd=fwd, eligible=np.isfinite(fwd), fi=np.isfinite(fr_raw),
              fr=np.where(np.isfinite(fr_raw), fr_raw, 0.0))
    R.prepare_shard(sh, R.WINSOR_PCT)
    univ = np.ones((A, T), dtype=bool)                     # all 205 coins always in univ
    beta = S.rolling_beta(close, bi)
    state = S.btc_state(close[bi])
    Larr = np.full(A, T + 10, np.int64)                     # no delistings (OKX pool persists)
    mask = univ & np.isfinite(beta)
    mask[bi] = False
    ctx = dict(mode="okx_pool", ts=ts, syms=syms, univ=univ, mask=mask, bi=bi, beta=beta, state=state, Larr=Larr,
               dead={}, months=S.months(ts), shard=sh, alpha=None, zp=None, offsets=None)
    ctx["cover_rows"] = np.concatenate([np.arange(m["m0"], m["m1"]) for m in ctx["months"]])
    ctx["feats_gpu"] = None                                # no VM signal needed here
    log("OKX pool world: %d coins x %d h, %d months (2021-01..%s), %.1fs"
        % (A, T, len(ctx["months"]), ctx["months"][-1]["month"], time.time() - t0))
    return ctx


def bins(vals):
    """Bins over monthly cost0 log NAV (6 bins: ≤-40, (-40,-20], (-20,0], (0,20], (20,40], >40 bp)."""
    c = np.asarray(vals, np.float64) * 1e4
    edges = [-np.inf, -40, -20, 0, 20, 40, np.inf]
    h, _ = np.histogram(c, bins=edges)
    return list(h.astype(int))


def settle(e, ctx, vm):
    """One entry through sel1_common.score_sel on all 60 months (A + B), bin the monthly results."""
    try:
        sg = S.signal(ctx, vm, tuple(e["body"]))
    except (S.Invalid, CandidateLocalInvalid):
        return dict(invalid=True)
    sels = S.select_months(sg, ctx, ("A", "B"))
    rows = S.score_sel(sels, ctx, e["h"], e["st"], e["dr"], ("A", "B"))
    ra = [r for r in rows if r["half"] == "A"]
    rb = [r for r in rows if r["half"] == "B"]
    return dict(expr=e["expr"], hold=e["h"], state=S.STATES[e["st"]], dir=S.DIRS[e["dr"]],
                months_all=len(rows), months_A=len(ra), months_B=len(rb),
                bpd_A=S.bpd(ra), t_A=S.reward(ra, "t"), bins_A=bins([x["cost0"] for x in ra]),
                bpd_B=S.bpd(rb), t_B=S.reward(rb, "t"), bins_B=bins([x["cost0"] for x in rb]),
                net_bpd_B=S.bpd(rb, "net"), turn_day_B=sum(x["turn"] for x in rb) / sum(x["days"] for x in rb),
                bpd_all=S.bpd(rows), t_all=S.reward(rows, "t"), bins_all=bins([x["cost0"] for x in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frozen")
    ap.add_argument("out")
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    if not os.path.exists(a.frozen):
        raise SystemExit("frozen file not found: %s" % a.frozen)
    if os.path.exists(a.out):
        raise SystemExit("output exists: %s" % a.out)
    t0 = time.time()
    torch.set_num_threads(4)
    rec = json.load(open(RECEIPT, encoding="utf-8"))
    fz = json.load(open(a.frozen, encoding="utf-8"))
    if fz.get("schema") not in ("SEL1_FROZEN_V1",):
        raise SystemExit("frozen schema not SEL1_FROZEN_V1: %s" % fz.get("schema"))
    ctx = okx_world(torch.device(a.device))
    vm = Q.make_vm()
    S.K = int(fz.get("k", 5))
    entries = fz["entries"][:a.top] if a.top else fz["entries"]
    log("%d entries from %s (K %d)" % (len(entries), a.frozen, S.K))
    out = []
    for k, e in enumerate(entries):
        r = settle(e, ctx, vm)
        log("[%d/%d] %s | A %.2f B %.2f all %.2f" % (k + 1, len(entries), e["expr"],
                                                      r.get("bpd_A", np.nan), r.get("bpd_B", np.nan), r.get("bpd_all", np.nan)))
        out.append(r)
    rep = dict(schema="SEL1_OKX_POOL_V1", frozen=a.frozen, frozen_sha256=Q.sha256(a.frozen),
               okx_receipt_sha256=Q.sha256(RECEIPT), okx_npz_sha256=rec["npz_sha256"], okx_pool=rec["pool"],
               n=len(out), invalid=len([r for r in out if r.get("invalid")]),
               entries=out, seconds=round(time.time() - t0, 1), written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log("-> %s | %d entries, %d invalid, %.1fs" % (a.out, len(out), rep["invalid"], rep["seconds"]))
    sys.exit(0)


if __name__ == "__main__":
    main()
