"""m2h_verdict.py copied for dense runs (TV1_A2_BASIN4_ORDER s3: "only the loading changes").

    python m2h_verdict_dense.py <RUN_NAME>          e.g. BASIN4_H1_DENSE_20261011

Changed against m2h_verdict.py (tier(), the paired t, baselines, port check, overlap untouched):
  - RUN_DIR from argv; ends on TRAIN_RECEIPT_V1.json OR STOPPED.json; writes VERDICT_DENSE.json
    and FROZEN_B4_TOP5.json
  - TR = m2h_dense_trainer (its sampler carries the H168 hold mask; same seed / train mode)
  - freeze pool: dense FORMULAS rows have no "reward"; the pool is the distinct (body, hold) of
    the final-checkpoint batch (256 per family, seed 20261108, train mode) -- the TV1 round
    freeze (s0: "only from the u256 / STOPPED checkpoint"), so it compares 1:1 with R4's
  - SIGN/ABS level formulas (D.level_flag) dropped before the top 5 (AGENTS.md, order s3)
  - columns: flag A = gross exposure < 0.5, flag B = |beta| > 0.3 or |net exposure| > 0.1
    (ADR-0004; listed only); R4 luck level from TV1_R4_DENSE_NULL\\ROUND_REPORT.json, noise =
    family top-5 mean TRAIN net not above R4's top-5 max of that family

Original header:
BASIN3_H1_20261009 verdict, handoff section VI (rewritten; the parallel session's draft is
archived as m2h_verdict_PREV.py). Run after TRAIN_RECEIPT_V1.json exists.

  1  freeze: per family, the top-100 distinct (body, hold) by training reward are re-scored on
     all 50 TRAIN tiles; behaviour dedup (identical 50-tile net vector), top-5 by mean net ->
     FROZEN_TOP5.json, sha256 printed. The DEV panel is opened only after this file exists.
  2  paired batches on the 12 DEV tiles: initial = that family's update-0 rows, final = 256 from
     the last checkpoint, basin token forced, seed 20261108. Per tile, mean over the batch;
     one-sided paired t on dNet and dGross. A formula invalid on DEV is dropped (same filter).
  3  baselines on the same 12 tiles, h1 code paths: T1 vs 1-week momentum and CTREND-lite
     (TOP100, hold 168); T2/T3 gate = DEV net > 0 and > best handwritten cell of the family.
  4  overlap: CS Spearman of each top-5 signal with 1-week momentum (every 24 h); T2/T3 > 0.7
     flagged as momentum variants.
  5  every row: TRAIN 2024 / TRAIN 2025 / DEV net, gross, turnover/day, beta, long and short legs.

Tier reading (stage-1 four tiers, operationalised here before DEV is read):
  可以推进    dGross > 0 at p < 0.05 and dNet > 0
  学会吃 β   dNet > 0, dGross not significant, final mean |beta| exceeds initial by > 0.1
  学会少交易  dNet > 0, dGross not significant, final turnover/day below initial
  没学到      otherwise
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, ttest_rel

ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem"
RUN_NAME = sys.argv[1] if len(sys.argv) > 1 else None
RUN_DIR = os.path.join(ROOT, "artifacts", "m2h_v1", RUN_NAME or "_")
TRAIN_PANEL = os.path.join(ROOT, r"runtime\train2024_2025_panel_v1\TRAIN2024JAN_2025DEC_HOURLY54_TARGET_LONG_V1.parquet")
DEV_NPZ = os.path.join(ROOT, "artifacts", "m2h_v1", "DEV2026_HOURLY_PANEL_V1.npz")
H1 = r"D:\ChengboRemote\runtime\strategy_survey_v1\h1_handwritten_baselines_TRAIN.json"
FROZEN = os.path.join(RUN_DIR, "FROZEN_B4_TOP5.json")
R4_REPORT = os.path.join(ROOT, "artifacts", "m2h_v1", "TV1_R4_DENSE_NULL", "ROUND_REPORT.json")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # -I drops cwd and PYTHONPATH
import m2h_dense_trainer as TR    # first: it puts the v4 / model roots on sys.path
import m2h_dense_common as D
import m1_common as C
import m1_ruler as R
import m2h_common as M
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import XsTsCsStackVM, OPERATOR_NAMES, CandidateLocalInvalid
from okx_alpha_discovery.adapters.xs_tscs_sampler_v2 import build_legality_table
from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod

FINAL_SEED = 20261108
POOL = 100
E = M.EPISODE
DAYS = E / 24.0
FIELDS = ("net", "gross", "turn", "beta", "long", "short", "gexp", "nexp")
T2_CELLS = ("T2_neg_funding_7d", "T2_neg_retail_LS_24h", "T2_neg_dlogOI_7d")
T3_CELLS = ("T3_neg_premium_24h", "T3_neg_mark_index_24h", "T3_abs_premium_high_1h")
BASE_HOLDS = (4, 24, 168)


def bpd(x):
    return R.bp_per_day(x, E * 60)


def tile_metrics(sigT, shard, s0, hold):
    """One tile through the ruler + h1's leg split. sigT is (E, A), lagged and pool-masked."""
    sc = R.score_window(sigT, shard, s0, E, [hold], policy_mod, costs=M.COSTS, variants=("win",),
                        band=R.BAND, nav_usd=R.NAV_USD)
    lg, p = sc[hold]["win"]
    el = shard["eligible"][:, s0:s0 + E].T
    fw = np.nan_to_num(shard["fwd_w"][:, s0:s0 + E].T)
    held = R.held_path(C.rank_weights(sigT, el), hold, R.BAND, policy_mod)
    return dict(net=bpd(lg["flat6"]), gross=bpd(lg["cost0"]), turn=p["turnover"] / DAYS,
                beta=p["beta_mkt"], long=1e4 * float((np.maximum(held, 0) * fw).sum()) / DAYS,
                short=1e4 * float((np.minimum(held, 0) * fw).sum()) / DAYS,
                gexp=p["gross_exposure_mean"], nexp=p["net_exposure_mean"])


def collect(per_tile):
    return {k: [float(d[k]) for d in per_tile] for k in FIELDS}


def score_body(vm, body, hold, slices, starts, univ, shard, want_sig=False):
    """Formula path = the trainer's (window_signal + coverage probe). None if invalid."""
    out, sigs = [], []
    try:
        for s0, sl in zip(starts, slices):
            sigT = M.window_signal(vm, body, sl, s0, univ)
            el = shard["eligible"][:, s0:s0 + E].T
            live = np.isfinite(sigT) & el
            probe = [t for t in [M.LAG] + list(range(E // 4, E, E // 4)) if t < E]
            if min(float(live[t].sum() / max(el[t].sum(), 1)) for t in probe) < M.MIN_COVERAGE:
                return None
            m = tile_metrics(sigT, shard, s0, hold)
            if not np.isfinite(m["net"]):
                return None
            out.append(m)
            if want_sig:
                sigs.append(sigT)
    except CandidateLocalInvalid:
        return None
    rec = collect(out)
    if want_sig:
        rec["_sig"] = sigs
    return rec


def score_signal(sig, hold, starts, univ, shard):
    """Handwritten path = h1's: pool mask on the absolute grid, then sigL[:, LAG:] = sig[s0:...]."""
    sig = np.where(univ, sig, np.nan)
    out = []
    for s0 in starts:
        sigL = np.full((sig.shape[0], E), np.nan)
        sigL[:, M.LAG:] = sig[:, s0:s0 + E - M.LAG]
        out.append(tile_metrics(sigL.T, shard, s0, hold))
    return collect(out)
def make_vm():
    import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vm_mod
    vm_mod.BLOCK_ELEMS = 1 << 25
    vm_mod.EMA_KERNEL_CACHE_MAX = 1
    return XsTsCsStackVM(len(M.TERMINALS), windows=M.WINDOWS)


def world_of(grids, first):
    w = M.build_world(grids)
    w["feats_pad"] = M.pad_feats(w["feats"])
    w["starts"] = M.tile_starts(w["feats"].shape[2], first=first)
    return w


def mean_or_none(v):
    v = [x for x in v if np.isfinite(x)]
    return round(float(np.mean(v)), 4) if v else None


def summary(rec, sel=None):
    idx = range(len(rec["net"])) if sel is None else sel
    return {k: mean_or_none([rec[k][i] for i in idx]) for k in FIELDS}


# --------------------------------------------------------------------------- 1. freeze
def freeze(vm, device):
    print("== freeze on TRAIN", flush=True)
    syms, ts, grids = M.load_train_grids(TRAIN_PANEL)
    w = world_of(grids, M.LOOKBACK)
    assert len(w["starts"]) == 50, len(w["starts"])
    slices = M.tile_slices(w["feats_pad"], w["starts"], device)
    years = [int(ts[s].year) for s in w["starts"]]
    model, ck_name, ck_upd = load_model(vm, device)
    r4 = json.load(open(R4_REPORT, encoding="utf-8")) if os.path.exists(R4_REPORT) else None
    fz = dict(schema="BASIN4_FROZEN_TOP5_V1", run=RUN_NAME, pool="final-checkpoint batch, 256 per family, "
              "seed %d, train mode, distinct (body, hold)" % FINAL_SEED, checkpoint=ck_name,
              checkpoint_update=ck_upd, n_tiles=50, tile_years=years, families={}, dropped_level_flag={},
              r4_luck=None, dev_read=False, oos_read=False)
    train_rows = {}
    for fam in M.BASIN_NAMES:
        seen, pool = set(), []
        for body, hold in final_batch(vm, model, fam, device):
            k = (tuple(int(x) for x in body), int(hold))
            if k not in seen:
                seen.add(k)
                pool.append(k)
        t0 = time.time()
        scored, flagged = [], 0
        for body, hold in pool:
            if D.level_flag(vm, list(body)):
                flagged += 1
                continue
            rec = score_body(vm, list(body), hold, slices, w["starts"], w["univ"], w["shard"])
            if rec is not None:
                scored.append((float(np.mean(rec["net"])), list(body), hold, rec))
        scored.sort(key=lambda x: -x[0])
        kept = []
        for s in scored:
            v = np.asarray(s[3]["net"])
            if any(np.array_equal(v, np.asarray(k[3]["net"])) for k in kept):
                continue
            kept.append(s)
            if len(kept) == 5:
                break
        print("   %s: pool %d level-flagged %d valid %d kept %d %.0fs"
              % (fam, len(pool), flagged, len(scored), len(kept), time.time() - t0), flush=True)
        fz["dropped_level_flag"][fam] = flagged
        fz["families"][fam] = [dict(rank=i + 1, body=b, hold=h, expr=vm.describe(b, list(M.TERMINALS)),
                                    train50_net=round(m, 4), train50_gross=mean_or_none(rec["gross"]),
                                    flag_A=bool(mean_or_none(rec["gexp"]) is not None and mean_or_none(rec["gexp"]) < 0.5),
                                    flag_B=bool((mean_or_none(rec["beta"]) is not None and abs(mean_or_none(rec["beta"])) > 0.3)
                                                or (mean_or_none(rec["nexp"]) is not None and abs(mean_or_none(rec["nexp"])) > 0.1)))
                               for i, (m, b, h, rec) in enumerate(kept)]
        train_rows[fam] = [rec for _, _, _, rec in kept]
        if r4 is not None:
            lv = [f["train50_net_bpd"] for f in r4["frozen"][fam]]
            mine = [f["train50_net"] for f in fz["families"][fam]]
            fz.setdefault("r4_luck", {})
            fz["r4_luck"][fam] = dict(r4_top5_net_max=max(lv) if lv else None,
                                      r4_top5_net_mean=round(float(np.mean(lv)), 3) if lv else None,
                                      b4_top5_net_mean=round(float(np.mean(mine)), 3) if mine else None,
                                      noise=bool(lv and mine and float(np.mean(mine)) <= max(lv)))
    del model
    torch.cuda.empty_cache()
    blob = json.dumps(fz["families"], sort_keys=True).encode()
    fz["sha256"] = hashlib.sha256(blob).hexdigest()
    json.dump(fz, open(FROZEN, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("FROZEN sha256", fz["sha256"], flush=True)
    del slices
    torch.cuda.empty_cache()
    return fz, train_rows, years, (grids, w, ts)


# --------------------------------------------------------------------------- 2. batches
def load_model(vm, device):
    from train_basin_riskpg_v1 import make_basin_model, apply_output_bias
    import model_core.alphagpt as alphagpt_mod
    from model_core.config import ModelConfig
    from model_core.vocab import FormulaVocab
    ModelConfig.DEVICE = device
    op_names = (list(OPERATOR_NAMES) + [f"W{w}" for w in M.WINDOWS] + ["END"] + M.HOLD_NAMES
                + list(M.BASIN_TOKEN_NAMES))
    model, _ = make_basin_model(alphagpt_mod=alphagpt_mod, ModelConfig=ModelConfig,
                                FormulaVocab=FormulaVocab, feature_names=list(M.TERMINALS),
                                operator_token_names=op_names, max_len=M.MAX_PROGRAM_LEN, device=device)
    apply_output_bias(model, len(M.TERMINALS), len(OPERATOR_NAMES), len(M.WINDOWS), TR.OP_BIAS, TR.END_BIAS)
    cks = sorted(glob.glob(os.path.join(RUN_DIR, "checkpoint_*.pt")))
    ck = torch.load(cks[-1], map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    return model, os.path.basename(cks[-1]), int(ck["update"])


def final_batch(vm, model, fam, device):
    """256 for one family, basin token forced, seed 20261108; model left in train mode, as the
    trainer samples (stage-1 s1_final_batch did the same)."""
    tab = build_legality_table(vm, max_len=M.MAX_PROGRAM_LEN, available_terminals=M.basin_terms(fam),
                               available_windows=M.WINDOWS).to(device)
    torch.manual_seed(FINAL_SEED)
    gen = torch.Generator(device=device)
    gen.manual_seed(FINAL_SEED)
    with torch.no_grad():
        b = TR.sample_batch_with_hold_basins(model, vm, [tab], [256], device=device, generator=gen,
                                             basins=(fam,))
    return [(list(TR.strip_hold(vm, list(t))), TR.hold_of(vm, list(t))) for t in b.formulas]
# --------------------------------------------------------------------------- 3. handwritten (h1 code)
def _lagk(x, k):
    y = np.full_like(x, np.nan); y[:, k:] = x[:, :-k]; return y


def _r(x, k, mp=None):
    return pd.DataFrame(x.T).rolling(k, min_periods=mp or k)


def _roll(x, k):
    return _r(x, k).mean().to_numpy().T


def _ema(x, k):
    return pd.DataFrame(x.T).ewm(span=k, adjust=False, min_periods=k).mean().to_numpy().T


CT_SIGN = dict(rsi=1, stochK=1, stochD=1, cci=1, sma5=-1, sma10=-1, sma20=-1, sma50=-1, macd=1,
               macd_ds=1, volsma50=-1, volsma100=-1, volsma200=-1, volmacd=1,
               boll_low=-1, boll_mid=-1, boll_high=-1)


def handwritten(g, ts):
    """h1_handwritten_baselines.py lines 76-154, verbatim formulas, on any hourly grid."""
    A, T = g["trade_close"].shape
    close = g["trade_close"]
    logc = np.log(close)
    oi = g["open_interest_value_last"]
    loi = np.where(oi > 0, np.log(np.where(oi > 0, oi, 1.0)), np.nan)
    fund_lvl = pd.DataFrame(g["funding_rate"].T).ffill().to_numpy().T
    S = {"T1_mom_168h": logc - _lagk(logc, 168),
         "T2_neg_funding_7d": -_roll(fund_lvl, 168),
         "T2_neg_retail_LS_24h": -_roll(g["global_long_short_account_ratio_last"], 24),
         "T2_neg_dlogOI_7d": -(loi - _lagk(loi, 168)),
         "T3_neg_premium_24h": -_roll(g["premium_close_bps"], 24),
         "T3_neg_mark_index_24h": -_roll(g["mark_index_basis_bps"], 24),
         "T3_abs_premium_high_1h": np.abs(g["premium_high"])}
    rs_ = lambda x: pd.DataFrame(x.T, index=ts).resample("D")
    Cd, Hd = rs_(close).last().to_numpy().T, rs_(g["trade_high"]).max().to_numpy().T
    Ld, Vd = rs_(g["trade_low"]).min().to_numpy().T, rs_(g["trade_quote_volume"]).sum(min_count=1).to_numpy().T
    days = rs_(close).last().index
    pos = np.clip(((days + pd.Timedelta(hours=23) - ts[0]) / pd.Timedelta(hours=1)).astype(int), 0, T - 1)
    d = {}
    ch = np.diff(Cd, axis=1, prepend=np.nan)
    rs = _roll(np.maximum(ch, 0), 14) / _roll(np.maximum(-ch, 0), 14)
    d["rsi"] = rs / (1 + rs)
    lo, hi = _r(Ld, 14).min().to_numpy().T, _r(Hd, 14).max().to_numpy().T
    k = (Cd - lo) / (hi - lo)
    d["stochK"], d["stochD"] = k, _roll(k, 3)
    tp = (Hd + Ld + Cd) / 3
    mtp = _roll(tp, 20)
    mad = np.mean([np.abs((_lagk(tp, j) if j else tp) - mtp) for j in range(20)], axis=0)
    d["cci"] = (tp - mtp) / (0.015 * mad)
    for n in (5, 10, 20, 50):
        d[f"sma{n}"] = _roll(Cd, n) / Cd
    macd = (_ema(Cd, 12) - _ema(Cd, 26)) / _ema(Cd, 12)
    d["macd"], d["macd_ds"] = macd, macd - _ema(macd, 9)
    for n in (50, 100, 200):
        d[f"volsma{n}"] = _roll(Vd, n) / Vd
    d["volmacd"] = (_ema(Vd, 12) - _ema(Vd, 26)) / _ema(Vd, 12)
    m20, sd = _roll(Cd, 20), _r(Cd, 20).std(ddof=0).to_numpy().T
    d["boll_low"], d["boll_mid"], d["boll_high"] = (m20 - 2 * sd) / Cd, m20 / Cd, (m20 + 2 * sd) / Cd
    cs_rank = lambda x: pd.DataFrame(np.where(np.isfinite(x), x, np.nan).T).rank(axis=1, pct=True).to_numpy().T
    rk = np.stack([CT_SIGN[k] * (cs_rank(v) - 0.5) for k, v in d.items()])
    n_ok = np.isfinite(rk).sum(0)
    ct = np.where(n_ok >= 9, np.nansum(rk, 0) / np.maximum(n_ok, 1), np.nan)
    h = np.full((A, T), np.nan); h[:, pos] = ct
    S["CTREND_lite_17"] = pd.DataFrame(h.T).ffill(limit=24).to_numpy().T
    return S


def mom_overlap(sigs, mom, univ, shard, starts):
    """CS Spearman vs 1-week momentum, every 24 h, at the same absolute signal hour."""
    mom = np.where(univ, mom, np.nan)
    vals = []
    for sigT, s0 in zip(sigs, starts):
        for t in range(M.LAG, E, 24):
            h = s0 + t - M.LAG
            a, b = sigT[t], mom[:, h]
            m = np.isfinite(a) & np.isfinite(b) & shard["eligible"][:, h]
            if m.sum() >= 20:
                vals.append(float(np.corrcoef(rankdata(a[m]), rankdata(b[m]))[0, 1]))
    return round(float(np.mean(vals)), 3) if vals else None
def tier(d_net, d_gross, p_gross, beta_i, beta_f, turn_i, turn_f):
    if d_gross > 0 and p_gross < 0.05 and d_net > 0:
        return "可以推进"
    if d_net > 0 and abs(beta_f) - abs(beta_i) > 0.1:
        return "学会吃 β"
    if d_net > 0 and turn_f < turn_i:
        return "学会少交易"
    return "没学到"


def batch_eval(vm, pairs, slices, starts, univ, shard):
    recs = [score_body(vm, b, h, slices, starts, univ, shard) for b, h in pairs]
    return [r for r in recs if r is not None], sum(r is None for r in recs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if RUN_NAME is None:
        raise SystemExit("usage: m2h_verdict_dense.py <RUN_NAME>")
    if not any(os.path.exists(os.path.join(RUN_DIR, f)) for f in ("TRAIN_RECEIPT_V1.json", "STOPPED.json")):
        raise SystemExit("training not finished (no receipt, no STOPPED.json)")
    vm = make_vm()
    t_all = time.time()
    fz, train_rows, years, (tgrids, tw, ts_t) = freeze(vm, device)

    # port check: the handwritten code used on DEV must reproduce h1's TRAIN net at hold 168
    h1 = json.load(open(H1, encoding="utf-8"))["results"]
    tS = handwritten(tgrids, ts_t)
    port = {}
    for name in ("T1_mom_168h", "CTREND_lite_17", "T2_neg_funding_7d", "T3_neg_premium_24h"):
        mine = float(np.mean(score_signal(tS[name], 168, tw["starts"], tw["univ"], tw["shard"])["net"]))
        ref = float(np.mean(h1[f"{name}|TOP100|lag1"]["168"]["net"]))
        port[name] = dict(mine=round(mine, 8), h1=round(ref, 8), d=abs(mine - ref))
        print("PORT %s %s mine=%.6f h1=%.6f" % ("PASS" if abs(mine - ref) <= 1e-6 else "FAIL", name, mine, ref),
              flush=True)
    if any(v["d"] > 1e-6 for v in port.values()):
        raise SystemExit("handwritten port does not reproduce h1 on TRAIN; stop before DEV")
    del tS

    # ---- DEV is opened only now, after FROZEN_TOP5.json is on disk
    assert os.path.exists(FROZEN)
    z = np.load(DEV_NPZ, allow_pickle=False)
    hours = pd.DatetimeIndex(z["hours"].astype("datetime64[ns]"))
    if hours.max() >= M.OOS_FIRST:
        raise SystemExit("DEV panel reaches OOS")
    dgrids = {c: z[c] for c in M.PANEL_COLS}
    jan1 = int((pd.Timestamp("2026-01-01") - hours[0]) / pd.Timedelta(hours=1))
    dw = world_of(dgrids, jan1)
    dw["starts"] = dw["starts"][:12]
    if len(dw["starts"]) != 12 or hours[dw["starts"][-1] + E - 1] >= M.OOS_FIRST:
        raise SystemExit(f"DEV tiling wrong: {len(dw['starts'])} tiles")
    dslices = M.tile_slices(dw["feats_pad"], dw["starts"], device)
    dev_span = (str(hours[dw["starts"][0]]), str(hours[dw["starts"][-1] + E - 1]))
    print("DEV tiles 12, %s -> %s" % dev_span, flush=True)

    # handwritten cells on DEV (TOP100, lag 1); T2/T3 at every h1 hold so "best cell" is the max
    dS = handwritten(dgrids, hours)
    base = {}
    for name in ("T1_mom_168h", "CTREND_lite_17") + T2_CELLS + T3_CELLS:
        hs = (168,) if name.startswith(("T1", "CTREND")) else BASE_HOLDS
        for h in hs:
            base[f"{name}|H{h}"] = summary(score_signal(dS[name], h, dw["starts"], dw["univ"], dw["shard"]))
    mom_dev = dS["T1_mom_168h"]
    mom_tr = handwritten(tgrids, ts_t)["T1_mom_168h"]

    model, ck_name, ck_upd = load_model(vm, device)
    rows = [json.loads(l) for l in open(os.path.join(RUN_DIR, "FORMULAS.jsonl"), encoding="utf-8")]
    yr = np.asarray(years)
    out = dict(schema="BASIN4_VERDICT_DENSE_V1", run=RUN_NAME, frozen_sha256=fz["sha256"],
               r4_luck=fz.get("r4_luck"),
               checkpoint=ck_name, checkpoint_update=ck_upd, final_seed=FINAL_SEED, dev_tiles=12,
               dev_span=dev_span, port_check=port, baselines_dev=base, families={}, oos_read=False)
    tslices = None
    for fam in M.BASIN_NAMES:
        t0 = time.time()
        init = [(list(TR.strip_hold(vm, list(r["tokens"]))), int(r["hold_h"]))
                for r in rows if r["update"] == 0 and r["basin"] == fam]
        fin = final_batch(vm, model, fam, device)
        ri, ni = batch_eval(vm, init, dslices, dw["starts"], dw["univ"], dw["shard"])
        rf, nf = batch_eval(vm, fin, dslices, dw["starts"], dw["univ"], dw["shard"])
        per = lambda rs, k: np.nanmean(np.array([r[k] for r in rs]), axis=0)
        a_n, b_n, a_g, b_g = per(ri, "net"), per(rf, "net"), per(ri, "gross"), per(rf, "gross")
        tn, tg = ttest_rel(b_n, a_n, alternative="greater"), ttest_rel(b_g, a_g, alternative="greater")
        bi, bf = float(np.nanmean(per(ri, "beta"))), float(np.nanmean(per(rf, "beta")))
        ui, uf = float(np.mean(per(ri, "turn"))), float(np.mean(per(rf, "turn")))
        dN, dG = float(np.mean(b_n - a_n)), float(np.mean(b_g - a_g))
        paired = dict(n_initial=len(ri), n_initial_invalid=ni, n_final=len(rf), n_final_invalid=nf,
                      final_distinct=len({(tuple(b), h) for b, h in fin}),
                      final_hold_hist={h: sum(x[1] == h for x in fin) for h in M.HOLDS},
                      init_net=round(float(a_n.mean()), 3), final_net=round(float(b_n.mean()), 3),
                      d_net=round(dN, 3), p_net=round(float(tn.pvalue), 4),
                      init_gross=round(float(a_g.mean()), 3), final_gross=round(float(b_g.mean()), 3),
                      d_gross=round(dG, 3), p_gross=round(float(tg.pvalue), 4),
                      beta_init=round(bi, 3), beta_final=round(bf, 3),
                      turn_init=round(ui, 3), turn_final=round(uf, 3),
                      tier=tier(dN, dG, float(tg.pvalue), bi, bf, ui, uf))
        # top-5 rows: TRAIN by year (records from freeze), DEV, overlap
        if tslices is None:
            tslices = M.tile_slices(tw["feats_pad"], tw["starts"], device)
        top = []
        for f, trec in zip(fz["families"][fam], train_rows[fam]):
            drec = score_body(vm, f["body"], f["hold"], dslices, dw["starts"], dw["univ"], dw["shard"], want_sig=True)
            tsig = score_body(vm, f["body"], f["hold"], tslices, tw["starts"], tw["univ"], tw["shard"], want_sig=True)
            row = dict(rank=f["rank"], expr=f["expr"], hold=f["hold"], flag_A=f["flag_A"], flag_B=f["flag_B"],
                       train50_net=f["train50_net"], train50_gross=f["train50_gross"],
                       train2024=summary(trec, list(np.where(yr == 2024)[0])),
                       train2025=summary(trec, list(np.where(yr == 2025)[0])),
                       dev=summary(drec) if drec else None,
                       rho_mom1w_train=mom_overlap(tsig["_sig"], mom_tr, tw["univ"], tw["shard"], tw["starts"]),
                       rho_mom1w_dev=(mom_overlap(drec["_sig"], mom_dev, dw["univ"], dw["shard"], dw["starts"])
                                      if drec else None))
            row["momentum_variant"] = bool(fam != "T1" and row["rho_mom1w_train"] is not None
                                           and row["rho_mom1w_train"] > 0.7)
            top.append(row)
        if fam == "T1":
            bar = {k: base[f"{k}|H168"]["net"] for k in ("T1_mom_168h", "CTREND_lite_17")}
            gate = dict(rule="beat 1-week momentum and CTREND-lite on DEV net", bars=bar,
                        pass_rank=[r["rank"] for r in top if r["dev"] and r["dev"]["net"] is not None
                                   and all(r["dev"]["net"] > v for v in bar.values() if v is not None)])
        else:
            cells = T2_CELLS if fam == "T2" else T3_CELLS
            best_k = max((k for k in base if k.split("|")[0] in cells), key=lambda k: base[k]["net"])
            bar = base[best_k]["net"]
            gate = dict(rule="DEV net > 0 and > best handwritten cell of the family", best_cell=best_k,
                        bar=bar, pass_rank=[r["rank"] for r in top if r["dev"] and r["dev"]["net"] is not None
                                            and r["dev"]["net"] > max(0.0, bar) and not r["momentum_variant"]])
        out["families"][fam] = dict(paired=paired, top5=top, gate=gate)
        print("%s: tier %s dNet %+.3f (p %.3f) dGross %+.3f (p %.3f) gate %s  %.0fs"
              % (fam, paired["tier"], dN, tn.pvalue, dG, tg.pvalue, gate["pass_rank"], time.time() - t0),
              flush=True)
    path = os.path.join(RUN_DIR, "VERDICT_DENSE.json")
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print("wrote", path, "%.0fs" % (time.time() - t_all))


if __name__ == "__main__":
    main()
