"""BASIN5 verdict (BASIN5_ORDER s4/s5/s6): freeze on TRAIN, then DEV paired verdict on the
concentrated-book net, plus the OKX settlement column.

    python b5_verdict.py <RUN_NAME>          e.g. B5_REAL_H1

Reused from m2h_verdict_dense (imported, not modified): the h1 handwritten baselines and their
TRAIN port check, the momentum overlap, tier(), world_of / make_vm / summary. Changed:
  TR       m2h_b5_trainer (same sampler code, H168 mask), final batch 256/family seed 20261108
  books    ("conc", K): m2h_conc_weights ternary book on the CONC pool (TOP100 at the signal hour
           ∩ CONC_POOL.json), equal weight +-1/(2K), through the unchanged ruler; ("full",) = the
           ruler's rank book. Primary = conc K=10; K=5 / K=20 sensitivity only (order s4).
  freeze   grammar-illegal (m2h_b5_grammar) and level-flag (D.level_flag) formulas dropped, ranked
           by conc K=10 TRAIN net on the 50 tiles, dedup on that net vector, top 5 per family ->
           FROZEN_B5_TOP5.json (n_tokens, level_flag, explanation, conc K5/10/20 + full TRAIN
           net). DEV is opened only after it is on disk.
  paired   initial = update-0 rows, final = final batch, both grammar-legal only; per DEV tile
           mean over the batch of conc K=10 net / gross; one-sided paired t; tier on conc beta /
           turnover. The full-pool paired numbers are reported next to it.
  gates    T1: conc K=10 DEV net beats 1-week momentum and CTREND-lite on the same conc book.
           T2/T3: conc K=10 DEV net > 0 and > the best handwritten cell of the family (conc).
  OKX      top-5 rows: the same conc K=10 held path settled on OKX native 1h close + OKX funding
           (OKX_DEV_1H_V1.npz, 2026-01..06 only, built by b5_okx_dev_grid.py; winsorised within
           the OKX shard like the ruler's 'win'), plus the raw (unwinsorised) value. Binance
           settlement of the same path through settle_multi is checked equal to score_window.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else None
import m2h_verdict_dense as V          # helpers only; its own main() is not called
import tv1_round as TV
import m2h_b5_trainer as TR
import m2h_b5_grammar as G
import m2h_conc_weights as W
import b5_explain as X
import m2h_dense_common as D
import m1_common as C
import m1_ruler as R
import m2h_common as M
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid
from okx_alpha_discovery.adapters.xs_tscs_sampler_v2 import build_legality_table
from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod

ROOT = TR.ROOT
RUN_DIR = os.path.join(ROOT, "artifacts", "m2h_v1", RUN_NAME or "_")
# B5V_TEST=1: pipeline test on a smoke checkpoint (8 per family, 4 TRAIN / 3 DEV tiles, *_TEST
# outputs; nothing in them is used for selection)
TEST = os.environ.get("B5V_TEST") == "1"
SFX = "_TEST" if TEST else ""
FROZEN = os.path.join(RUN_DIR, "FROZEN_B5_TOP5%s.json" % SFX)
OUT = os.path.join(RUN_DIR, "VERDICT_B5%s.json" % SFX)
N_FINAL = 8 if TEST else 256
OKX_NPZ = os.path.join(HERE, "OKX_DEV_1H_V1.npz")
POOL_JSON = os.path.join(HERE, "CONC_POOL.json")
FINAL_SEED = 20261108
E = M.EPISODE
DAYS = E / 24.0
HOLD = 168
PRIMARY = ("conc", W.K_DEFAULT)
BOOKS = (PRIMARY,) + tuple(("conc", k) for k in W.K_SENS) + (("full",),)
FIELDS = ("net", "gross", "turn", "beta", "long", "short", "gexp", "nexp")


def bk(book):
    return "conc_k%d" % book[1] if book[0] == "conc" else "full"


def bpd(x):
    return R.bp_per_day(x, E * 60)


def book_signal(sigT, el, book, pool):
    return W.conc_signal(sigT, el, pool, book[1]) if book[0] == "conc" else sigT


def tile_metrics(sigT, shard, s0, book, pool):
    el = shard["eligible"][:, s0:s0 + E].T
    bs = book_signal(sigT, el, book, pool)
    sc = R.score_window(bs, shard, s0, E, [HOLD], policy_mod, costs=M.COSTS, variants=("win",),
                        band=R.BAND, nav_usd=R.NAV_USD)
    lg, p = sc[HOLD]["win"]
    fw = np.nan_to_num(shard["fwd_w"][:, s0:s0 + E].T)
    held = R.held_path(C.rank_weights(bs, el), HOLD, R.BAND, policy_mod)
    return dict(net=bpd(lg["flat6"]), gross=bpd(lg["cost0"]), turn=p["turnover"] / DAYS,
                beta=p["beta_mkt"], long=1e4 * float((np.maximum(held, 0) * fw).sum()) / DAYS,
                short=1e4 * float((np.minimum(held, 0) * fw).sum()) / DAYS,
                gexp=p["gross_exposure_mean"], nexp=p["net_exposure_mean"]), held


def signals(vm, body, slices, starts, univ, shard):
    """The trainer's formula path (window_signal + coverage probe); None if invalid."""
    out = []
    try:
        for s0, sl in zip(starts, slices):
            sigT = M.window_signal(vm, list(body), sl, s0, univ)
            el = shard["eligible"][:, s0:s0 + E].T
            live = np.isfinite(sigT) & el
            probe = [t for t in [M.LAG] + list(range(E // 4, E, E // 4)) if t < E]
            if min(float(live[t].sum() / max(el[t].sum(), 1)) for t in probe) < M.MIN_COVERAGE:
                return None
            out.append(sigT)
    except CandidateLocalInvalid:
        return None
    return out


def score_sigs(sigs, starts, shard, books, pool, keep_held=False):
    """{book key: {field: [per tile]}} (+ '_held' for the primary book if asked). None if any
    tile's net is not finite."""
    rec = {}
    for book in books:
        per, helds = [], []
        for sigT, s0 in zip(sigs, starts):
            m, held = tile_metrics(sigT, shard, s0, book, pool)
            if not np.isfinite(m["net"]):
                return None
            per.append(m)
            if keep_held and book == PRIMARY:
                helds.append(held)
        rec[bk(book)] = {k: [float(d[k]) for d in per] for k in FIELDS}
        if keep_held and book == PRIMARY:
            rec["_held"] = helds
    return rec


def score_body(vm, body, slices, starts, univ, shard, books, pool, keep=False):
    sigs = signals(vm, body, slices, starts, univ, shard)
    if sigs is None:
        return None
    rec = score_sigs(sigs, starts, shard, books, pool, keep_held=keep)
    if rec is not None and keep:
        rec["_sig"] = sigs
    return rec


def score_signal(sig, starts, univ, shard, books, pool):
    """Handwritten path = h1's: pool mask on the absolute grid, sigL[:, LAG:] = sig[s0:...]."""
    sig = np.where(univ, sig, np.nan)
    sigs = []
    for s0 in starts:
        sigL = np.full((sig.shape[0], E), np.nan)
        sigL[:, M.LAG:] = sig[:, s0:s0 + E - M.LAG]
        sigs.append(sigL.T)
    return score_sigs(sigs, starts, shard, books, pool)


def final_batch(vm, model, fam, device):
    tab = build_legality_table(vm, max_len=M.MAX_PROGRAM_LEN, available_terminals=M.basin_terms(fam),
                               available_windows=M.WINDOWS).to(device)
    torch.manual_seed(FINAL_SEED)
    gen = torch.Generator(device=device)
    gen.manual_seed(FINAL_SEED)
    with torch.no_grad():
        b = TR.sample_batch_with_hold_basins(model, vm, [tab], [N_FINAL], device=device, generator=gen,
                                             basins=(fam,))
    return [(list(TR.strip_hold(vm, list(t))), TR.hold_of(vm, list(t))) for t in b.formulas]


def legal(vm, body):
    return G.check(vm, list(body)) is None


# --------------------------------------------------------------------------- OKX shard
def okx_shard(dev_syms, dev_hours):
    """OKX close / funding on the DEV panel's (symbol, hour) grid; NaN where OKX has no bar or the
    coin is not in the OKX grid (non-pool coins). fwd / eligible / fi / fr as build_world makes
    them, then prepare_shard (fwd_w)."""
    z = np.load(OKX_NPZ, allow_pickle=False)
    oh = pd.DatetimeIndex(z["hours"].astype("datetime64[ns]"))
    if oh.max() >= M.OOS_FIRST:
        raise SystemExit("OKX grid reaches OOS")
    di = pd.Index(list(dev_syms))
    close = np.full((len(dev_syms), len(dev_hours)), np.nan)
    fr_raw = np.full_like(close, np.nan)
    col = pd.DatetimeIndex(dev_hours).get_indexer(oh)
    ok = col >= 0
    rows = di.get_indexer(list(z["symbols"]))
    for j, r in enumerate(rows):
        if r < 0:
            continue
        close[r, col[ok]] = z["close"][j][ok]
        fr_raw[r, col[ok]] = z["funding_rate"][j][ok]
    fwd = C.forward_returns(close, None)
    sh = dict(fwd=fwd, eligible=np.isfinite(fwd), fi=np.isfinite(fr_raw),
              fr=np.where(np.isfinite(fr_raw), fr_raw, 0.0))
    R.prepare_shard(sh, R.WINSOR_PCT)
    return sh, dict(okx_symbols=int(len(z["symbols"])), mapped=int((rows >= 0).sum()),
                    okx_hours=[str(oh.min()), str(oh.max())])


def okx_settle(helds, starts, osh):
    """Same held paths, settled on the OKX shard: {variant: per-tile net / gross bp/day}."""
    specs = {c: R.cost_specs(R.NAV_USD)[c] for c in M.COSTS}
    out = {v: dict(net=[], gross=[]) for v in ("win", "raw")}
    for held, s0 in zip(helds, starts):
        fim, frt = osh["fi"][:, s0:s0 + E].T, osh["fr"][:, s0:s0 + E].T
        for v, key in (("win", "fwd_w"), ("raw", "fwd")):
            res, _ = R.settle_multi(held, osh[key][:, s0:s0 + E].T, fim, frt, specs)
            out[v]["net"].append(bpd(res["flat6"]["log_nav"]))
            out[v]["gross"].append(bpd(res["cost0"]["log_nav"]))
    return {v: {k: V.mean_or_none(x) for k, x in d.items()} for v, d in out.items()}


def binance_resettle_check(helds, starts, shard, rec):
    """settle_multi on the Binance shard with the same held path must equal score_window's net."""
    specs = {c: R.cost_specs(R.NAV_USD)[c] for c in M.COSTS}
    d = 0.0
    for held, s0, net in zip(helds, starts, rec[bk(PRIMARY)]["net"]):
        res, _ = R.settle_multi(held, shard["fwd_w"][:, s0:s0 + E].T, shard["fi"][:, s0:s0 + E].T,
                                shard["fr"][:, s0:s0 + E].T, specs)
        d = max(d, abs(bpd(res["flat6"]["log_nav"]) - net))
    return d


# --------------------------------------------------------------------------- freeze
def freeze(vm, device, pool_t):
    print("== freeze on TRAIN", flush=True)
    syms, ts, grids = M.load_train_grids(V.TRAIN_PANEL)
    w = V.world_of(grids, M.LOOKBACK)
    assert len(w["starts"]) == 50, len(w["starts"])
    if TEST:
        w["starts"] = w["starts"][:4]
    slices = M.tile_slices(w["feats_pad"], w["starts"], device)
    years = [int(ts[s].year) for s in w["starts"]]
    model, ck_name, ck_upd = TV.load_final(TR, RUN_DIR, device)
    fz = dict(schema="B5_FROZEN_TOP5_V1", run=RUN_NAME,
              pool="final-checkpoint batch, 256 per family, seed %d, train mode, distinct (body, hold)"
                   % FINAL_SEED, checkpoint=ck_name, checkpoint_update=ck_upd, n_tiles=50,
              rank="conc book K=%d flat6 TRAIN net, H%d" % (PRIMARY[1], HOLD),
              conc_pool_train=int(pool_t.sum()), tile_years=years, families={}, dropped={},
              dev_read=False, oos_read=False)
    train_rows = {}
    for fam in M.BASIN_NAMES:
        seen, pool = set(), []
        for body, hold in final_batch(vm, model, fam, device):
            k = (tuple(int(x) for x in body), int(hold))
            if k not in seen:
                seen.add(k)
                pool.append(k)
        t0 = time.time()
        scored, n_illegal, n_flag = [], 0, 0
        for body, hold in pool:
            if not legal(vm, body):
                n_illegal += 1
                continue
            if D.level_flag(vm, list(body)):
                n_flag += 1
                continue
            rec = score_body(vm, list(body), slices, w["starts"], w["univ"], w["shard"], BOOKS, pool_t)
            if rec is not None:
                scored.append((float(np.mean(rec[bk(PRIMARY)]["net"])), list(body), hold, rec))
        scored.sort(key=lambda x: -x[0])
        kept = []
        for s in scored:
            v = np.asarray(s[3][bk(PRIMARY)]["net"])
            if any(np.array_equal(v, np.asarray(k[3][bk(PRIMARY)]["net"])) for k in kept):
                continue
            kept.append(s)
            if len(kept) == 5:
                break
        print("   %s: pool %d illegal %d level-flagged %d valid %d kept %d %.0fs"
              % (fam, len(pool), n_illegal, n_flag, len(scored), len(kept), time.time() - t0), flush=True)
        fz["dropped"][fam] = dict(distinct=len(pool), grammar_illegal=n_illegal, level_flag=n_flag,
                                  valid=len(scored))
        rows = []
        for i, (m, b, h, rec) in enumerate(kept):
            pr = V.summary(rec[bk(PRIMARY)])
            rows.append(dict(rank=i + 1, body=b, hold=h, expr=vm.describe(b, list(M.TERMINALS)),
                             n_tokens=len(b) - 1, level_flag=bool(D.level_flag(vm, b)),
                             explain=X.explain(vm, b),
                             train50_conc_k10_net=round(m, 4), train50_conc_k10_gross=pr["gross"],
                             train50_conc_k5_net=V.summary(rec["conc_k5"])["net"],
                             train50_conc_k20_net=V.summary(rec["conc_k20"])["net"],
                             train50_full_net=V.summary(rec["full"])["net"],
                             train50_full_gross=V.summary(rec["full"])["gross"],
                             conc_turnover_per_day=pr["turn"],
                             flag_A=bool(pr["gexp"] is not None and pr["gexp"] < 0.5),
                             flag_B=bool((pr["beta"] is not None and abs(pr["beta"]) > 0.3)
                                         or (pr["nexp"] is not None and abs(pr["nexp"]) > 0.1))))
        fz["families"][fam] = rows
        train_rows[fam] = [rec for _, _, _, rec in kept]
    del model
    blob = json.dumps(fz["families"], sort_keys=True).encode()
    fz["sha256"] = hashlib.sha256(blob).hexdigest()
    fz["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(fz, open(FROZEN, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("FROZEN sha256", fz["sha256"], flush=True)
    del slices
    torch.cuda.empty_cache()
    return fz, train_rows, years, (syms, grids, w, ts)


# --------------------------------------------------------------------------- main
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if RUN_NAME is None:
        raise SystemExit("usage: b5_verdict.py <RUN_NAME>")
    if not any(os.path.exists(os.path.join(RUN_DIR, f)) for f in ("TRAIN_RECEIPT_V1.json", "STOPPED.json")):
        raise SystemExit("training not finished (no receipt, no STOPPED.json)")
    if os.path.exists(OUT):
        raise SystemExit("VERDICT_B5.json exists; not overwritten")
    vm = V.make_vm()
    t_all = time.time()
    import pyarrow.parquet as pq
    tsyms = sorted(pq.read_table(V.TRAIN_PANEL, columns=["symbol"]).column(0).unique().to_pylist())
    pool_t = W.pool_cols(tsyms, POOL_JSON)
    fz, train_rows, years, (tsyms2, tgrids, tw, ts_t) = freeze(vm, device, pool_t)
    assert list(tsyms2) == list(tsyms)

    # port check (full book, all 50 tiles): the handwritten code reproduces h1's TRAIN net at H168
    h1 = json.load(open(V.H1, encoding="utf-8"))["results"]
    tS = V.handwritten(tgrids, ts_t)
    port_starts = M.tile_starts(tw["feats"].shape[2])
    port = {}
    for name in ("T1_mom_168h", "CTREND_lite_17", "T2_neg_funding_7d", "T3_neg_premium_24h"):
        mine = float(np.mean(score_signal(tS[name], port_starts, tw["univ"], tw["shard"],
                                          (("full",),), pool_t)["full"]["net"]))
        ref = float(np.mean(h1[f"{name}|TOP100|lag1"]["168"]["net"]))
        port[name] = dict(mine=round(mine, 8), h1=round(ref, 8), d=abs(mine - ref))
        print("PORT %s %s mine=%.6f h1=%.6f" % ("PASS" if abs(mine - ref) <= 1e-6 else "FAIL", name, mine, ref),
              flush=True)
    if any(v["d"] > 1e-6 for v in port.values()):
        raise SystemExit("handwritten port does not reproduce h1 on TRAIN; stop before DEV")
    mom_tr = tS["T1_mom_168h"]
    del tS

    # ---- DEV is opened only now, after FROZEN_B5_TOP5.json is on disk
    assert os.path.exists(FROZEN)
    z = np.load(V.DEV_NPZ, allow_pickle=False)
    hours = pd.DatetimeIndex(z["hours"].astype("datetime64[ns]"))
    if hours.max() >= M.OOS_FIRST:
        raise SystemExit("DEV panel reaches OOS")
    dsyms = [str(x) for x in z["symbols"]]
    dgrids = {c: z[c] for c in M.PANEL_COLS}
    jan1 = int((pd.Timestamp("2026-01-01") - hours[0]) / pd.Timedelta(hours=1))
    dw = V.world_of(dgrids, jan1)
    dw["starts"] = dw["starts"][:12]
    if len(dw["starts"]) != 12 or hours[dw["starts"][-1] + E - 1] >= M.OOS_FIRST:
        raise SystemExit(f"DEV tiling wrong: {len(dw['starts'])} tiles")
    if TEST:
        dw["starts"] = dw["starts"][:3]
    pool_d = W.pool_cols(dsyms, POOL_JSON)
    dslices = M.tile_slices(dw["feats_pad"], dw["starts"], device)
    dev_span = (str(hours[dw["starts"][0]]), str(hours[dw["starts"][-1] + E - 1]))
    live_pool = [float((dw["univ"][:, s0 + t] & pool_d).sum()) for s0 in dw["starts"] for t in range(0, E, 24)]
    print("DEV tiles 12, %s -> %s; TOP100 ∩ pool per hour mean %.1f min %d"
          % (dev_span + (np.mean(live_pool), min(live_pool))), flush=True)
    osh, okx_info = okx_shard(dsyms, hours)

    # handwritten cells on DEV (TOP100, lag 1), conc K10 and full
    dS = V.handwritten(dgrids, hours)
    base = {}
    for name in ("T1_mom_168h", "CTREND_lite_17") + V.T2_CELLS + V.T3_CELLS:
        rec = score_signal(dS[name], dw["starts"], dw["univ"], dw["shard"], (PRIMARY, ("full",)), pool_d)
        base[name] = {bk(b): V.summary(rec[bk(b)]) for b in (PRIMARY, ("full",))}
    mom_dev = dS["T1_mom_168h"]
    del dS

    model, ck_name, ck_upd = TV.load_final(TR, RUN_DIR, device)
    rows = []
    for l in open(os.path.join(RUN_DIR, "FORMULAS.jsonl"), encoding="utf-8"):
        if l.startswith('{"update": 0,'):
            rows.append(json.loads(l))
    if TEST:
        rows = [r for k, r in enumerate(rows) if k % 4 == 0]
    yr = np.asarray(years)
    out = dict(schema="B5_VERDICT_V1", run=RUN_NAME, test=TEST, frozen_sha256=fz["sha256"],
               checkpoint=ck_name, checkpoint_update=ck_upd, final_seed=FINAL_SEED, dev_tiles=12,
               dev_span=dev_span, primary_book=bk(PRIMARY), sensitivity_books=["conc_k5", "conc_k20"],
               conc_pool_dev=int(pool_d.sum()), top100_pool_live_mean=round(float(np.mean(live_pool)), 1),
               top100_pool_live_min=int(min(live_pool)), okx=okx_info, port_check=port,
               baselines_dev=base, families={}, oos_read=False)
    tslices = None
    for fam in M.BASIN_NAMES:
        t0 = time.time()
        init = [(list(TR.strip_hold(vm, list(r["tokens"]))), int(r["hold_h"])) for r in rows if r["basin"] == fam]
        fin = final_batch(vm, model, fam, device)
        paired = {}
        res = {}
        for tag, pairs in (("init", init), ("final", fin)):       # one VM pass, both books
            ok = [(b, h) for b, h in pairs if legal(vm, b)]
            recs = [score_body(vm, b, dslices, dw["starts"], dw["univ"], dw["shard"],
                               (PRIMARY, ("full",)), pool_d) for b, h in ok]
            res[tag] = ([r for r in recs if r is not None], sum(r is None for r in recs), len(pairs) - len(ok))
        for book in (PRIMARY, ("full",)):
            per = lambda rs, k: np.nanmean(np.array([r[bk(book)][k] for r in rs]), axis=0)
            ri, rf = res["init"][0], res["final"][0]
            a_n, b_n, a_g, b_g = per(ri, "net"), per(rf, "net"), per(ri, "gross"), per(rf, "gross")
            tn, tg = ttest_rel(b_n, a_n, alternative="greater"), ttest_rel(b_g, a_g, alternative="greater")
            bi, bf = float(np.nanmean(per(ri, "beta"))), float(np.nanmean(per(rf, "beta")))
            ui, uf = float(np.mean(per(ri, "turn"))), float(np.mean(per(rf, "turn")))
            dN, dG = float(np.mean(b_n - a_n)), float(np.mean(b_g - a_g))
            paired[bk(book)] = dict(
                n_initial=len(ri), n_initial_invalid=res["init"][1], n_initial_grammar_illegal=res["init"][2],
                n_final=len(rf), n_final_invalid=res["final"][1], n_final_grammar_illegal=res["final"][2],
                init_net=round(float(a_n.mean()), 3), final_net=round(float(b_n.mean()), 3),
                d_net=round(dN, 3), p_net=round(float(tn.pvalue), 4),
                init_gross=round(float(a_g.mean()), 3), final_gross=round(float(b_g.mean()), 3),
                d_gross=round(dG, 3), p_gross=round(float(tg.pvalue), 4),
                beta_init=round(bi, 3), beta_final=round(bf, 3), turn_init=round(ui, 3), turn_final=round(uf, 3),
                tier=V.tier(dN, dG, float(tg.pvalue), bi, bf, ui, uf))
        paired["final_distinct"] = len({(tuple(b), h) for b, h in fin})
        if tslices is None:
            tslices = M.tile_slices(tw["feats_pad"], tw["starts"], device)
        top = []
        for f, trec in zip(fz["families"][fam], train_rows[fam]):
            drec = score_body(vm, f["body"], dslices, dw["starts"], dw["univ"], dw["shard"],
                              BOOKS, pool_d, keep=True)
            tsig = signals(vm, f["body"], tslices, tw["starts"], tw["univ"], tw["shard"])
            row = dict(rank=f["rank"], expr=f["expr"], explain=f["explain"], n_tokens=f["n_tokens"],
                       level_flag=f["level_flag"], flag_A=f["flag_A"], flag_B=f["flag_B"],
                       train50_conc_k10_net=f["train50_conc_k10_net"],
                       train2024=V.summary(trec[bk(PRIMARY)], list(np.where(yr == 2024)[0])),
                       train2025=V.summary(trec[bk(PRIMARY)], list(np.where(yr == 2025)[0])),
                       dev={bk(b): V.summary(drec[bk(b)]) for b in BOOKS} if drec else None,
                       rho_mom1w_train=V.mom_overlap(tsig, mom_tr, tw["univ"], tw["shard"], tw["starts"])
                       if tsig else None,
                       rho_mom1w_dev=(V.mom_overlap(drec["_sig"], mom_dev, dw["univ"], dw["shard"], dw["starts"])
                                      if drec else None))
            if drec:
                row["dev_okx_conc_k10"] = okx_settle(drec["_held"], dw["starts"], osh)
                row["binance_resettle_maxdiff_bpd"] = binance_resettle_check(drec["_held"], dw["starts"],
                                                                              dw["shard"], drec)
                b_net = row["dev"][bk(PRIMARY)]["net"]
                o_net = row["dev_okx_conc_k10"]["win"]["net"]
                row["okx_minus_binance_net_bpd"] = (round(o_net - b_net, 3)
                                                    if o_net is not None and b_net is not None else None)
            row["momentum_variant"] = bool(fam != "T1" and row["rho_mom1w_train"] is not None
                                           and row["rho_mom1w_train"] > 0.7)
            top.append(row)
        dnet = lambda r: r["dev"][bk(PRIMARY)]["net"] if r["dev"] else None
        if fam == "T1":
            bar = {k: base[k][bk(PRIMARY)]["net"] for k in ("T1_mom_168h", "CTREND_lite_17")}
            gate = dict(rule="conc K10 DEV net beats 1-week momentum and CTREND-lite (same conc book)", bars=bar,
                        pass_rank=[r["rank"] for r in top if dnet(r) is not None
                                   and all(dnet(r) > v for v in bar.values() if v is not None)])
        else:
            cells = V.T2_CELLS if fam == "T2" else V.T3_CELLS
            best_k = max(cells, key=lambda k: base[k][bk(PRIMARY)]["net"])
            bar = base[best_k][bk(PRIMARY)]["net"]
            gate = dict(rule="conc K10 DEV net > 0 and > best handwritten cell of the family (conc, H168)",
                        best_cell=best_k, bar=bar,
                        pass_rank=[r["rank"] for r in top if dnet(r) is not None
                                   and dnet(r) > max(0.0, bar) and not r["momentum_variant"]])
        out["families"][fam] = dict(paired=paired, top5=top, gate=gate)
        p = paired[bk(PRIMARY)]
        print("%s: tier %s dNet %+.3f (p %.3f) dGross %+.3f (p %.3f) gate %s  %.0fs"
              % (fam, p["tier"], p["d_net"], p["p_net"], p["d_gross"], p["p_gross"], gate["pass_rank"],
                 time.time() - t0), flush=True)
    out["seconds"] = round(time.time() - t_all, 1)
    out["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print("wrote", OUT, "%.0fs" % (time.time() - t_all))


if __name__ == "__main__":
    main()
