"""TV1 pre-launch self-checks (TV1_ORDER s5 + TV1_A2_BASIN4_ORDER s2). One line per check.

 1  BASIN3 checks 1-5 through the dense scoring path (D.score_dense) and the dense sampler
    (5 also: every sampled hold token is H168); BASIN3 check 6 (smoke) = check 8
 2  1-week momentum, 16 tiles: D.score_dense cost0 AND flat6 bit-equal to M.score_formula
    (the m1_ruler path m2h_trainer uses)
 3  IC: constant signal = 0; the true next-168 h cumulative return as the signal = 1.
    C1's IC term is the same D.ic_tile, so this also covers A2's check.
 5  alpha bisection (+32 / +8 bp/day H168 gross lift of the plant formula on the 50 TRAIN tiles
    of the trainer's shard, within 10 %); writes TV1_S.npy + TV1_ALPHA.json; reports gross at
    the 4 holds and corr(s, 1-week momentum)
 6  shuffled null: 1-week momentum TRAIN gross |t| < 2
 7  plant / null only touch TRAIN: panel ends before 2026-01-01, feats untouched, no DEV / OOS
    path in the TV1 sources
 4,8 smoke through m2h_plant: dense 2 updates x 32 (gradient rows = batch, z rows = valid,
    checkpoint written, seconds per update); old trainer the same
 F  funding_level == 0 in the TRAIN TOP100 pool (TV1 s8), two lines
Writes TV1_SELFTEST.json. Any FAIL: the chain does not start.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m2h_selftest as ST            # sys.path for m1_* / okx_alpha_discovery
import m2h_dense_trainer as DT       # legacy v4 roots
import m2h_common as M
import m2h_dense_common as D
import m1_ruler as R
import m1_common as C
from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod

TV1 = os.path.join(DT.ROOT, "artifacts", "m2h_v1", "TV1_20261010")
TARGETS = dict(p32=32.0, p8=8.0)
RES = {}


def say(key, ok, msg):
    RES[key] = dict(ok=bool(ok), msg=msg)
    print("CHK%s %s %s" % (key, "PASS" if ok else "FAIL", msg), flush=True)


def gross_bpd(vm, body, slices, starts, univ, shard, holds=(168,), cost="cost0"):
    res = D.score_dense(vm, body, slices, starts, univ, shard, policy_mod, holds=holds,
                        costs=(cost,), want_ic=False)
    return {h: np.asarray([D.bpd(x) for x in res["per_hold"][h][cost]]) for h in holds}


def arr_sha(a):
    return hashlib.sha256(np.ascontiguousarray(np.nan_to_num(a, nan=-9e9)).tobytes()).hexdigest()[:16]


def gross_of(sigs, starts, shard, h, cost="cost0"):
    """bp/day per tile of fixed tile signals (E, A) scored by the ruler on shard at hold h."""
    out = []
    for s0, sigT in zip(starts, sigs):
        lg, _ = R.score_window(sigT, shard, s0, M.EPISODE, [h], policy_mod, costs=(cost,),
                               variants=("win",), band=R.BAND, nav_usd=R.NAV_USD)[h]["win"]
        out.append(D.bpd(lg[cost]))
    return np.asarray(out)


def mean_cs(a, b, step=24):
    """mean over hours (every step) of the TOP100 CS Spearman of two (A, T) arrays."""
    t = np.arange(M.LOOKBACK, a.shape[1], step)
    return float(D.cs_spearman_rows(a[:, t].T, b[:, t].T).mean())


def main():
    t_all = time.time()
    os.makedirs(TV1, exist_ok=True)
    dev = torch.device("cuda")
    vm = ST.make_vm()
    syms, ts, grids = M.load_train_grids(DT.PANEL)
    w = M.build_world(grids)
    feats, univ, base = w["feats"], w["univ"], w["shard"]
    starts = M.tile_starts(feats.shape[2])
    fp = M.pad_feats(feats)
    sl64 = M.tile_slices(fp, starts, dev, dtype=torch.float64)
    mom = ST.program(vm, 0, "DELTA", 168)

    # ---- 1: BASIN3 checks 1-5 on the dense path
    fwd = C.forward_returns(grids["trade_close"], None)
    pf = grids["forward_trade_return_1h"]
    both = np.isfinite(fwd) & np.isfinite(pf)
    c1 = int((fwd[both] == pf[both]).sum()) == ST.H1_LABEL["bit_equal"] and \
        int((np.isfinite(fwd) != np.isfinite(pf)).sum()) == ST.H1_LABEL["nan_mismatch"]
    worst = 0.0
    for const in (1.0, -3.5, 0.0):
        sc = R.score_window(np.full((M.EPISODE, feats.shape[0]), const), base, starts[0], M.EPISODE,
                            M.HOLDS, policy_mod, costs=M.COSTS, variants=("win",), band=R.BAND,
                            nav_usd=R.NAV_USD)
        for h in M.HOLDS:
            lg, p = sc[h]["win"]
            worst = max(worst, abs(lg["flat6"]), abs(lg["cost0"]), p["turnover"])
    d34 = []
    for body, tgt in ((mom, ST.T1_TARGET), (ST.program(vm, 4, "TS_MEAN", 168, negate=True), ST.T2_TARGET)):
        net = gross_bpd(vm, body, sl64, starts, univ, base, cost="flat6")[168]
        d34.append(abs(float(net.mean()) - tgt))
    bad, holds_bad = 0, 0
    from train_basin_riskpg_v1 import make_basin_model, apply_output_bias
    import model_core.alphagpt as alphagpt_mod
    from model_core.config import ModelConfig
    from model_core.vocab import FormulaVocab
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import OPERATOR_NAMES
    from okx_alpha_discovery.adapters.xs_tscs_sampler_v2 import build_legality_table
    ModelConfig.DEVICE = dev
    torch.manual_seed(20261009)
    names = (list(OPERATOR_NAMES) + [f"W{x}" for x in M.WINDOWS] + ["END"] + M.HOLD_NAMES
             + list(M.BASIN_TOKEN_NAMES))
    model, _ = make_basin_model(alphagpt_mod=alphagpt_mod, ModelConfig=ModelConfig,
                                FormulaVocab=FormulaVocab, feature_names=list(M.TERMINALS),
                                operator_token_names=names, max_len=M.MAX_PROGRAM_LEN, device=dev)
    apply_output_bias(model, len(M.TERMINALS), len(OPERATOR_NAMES), len(M.WINDOWS), DT.OP_BIAS, DT.END_BIAS)
    for b in M.BASIN_NAMES:
        terms = set(M.basin_terms(b))
        tab = build_legality_table(vm, max_len=M.MAX_PROGRAM_LEN, available_terminals=M.basin_terms(b),
                                   available_windows=M.WINDOWS).to(dev)
        with torch.no_grad():
            sb = DT.sample_batch_with_hold_basins(model, vm, [tab], [1000], device=dev, basins=(b,))
        for toks in sb.formulas:
            holds_bad += DT.hold_of(vm, list(toks)) != 168
            body = list(DT.strip_hold(vm, list(toks)))
            bad += sum(1 for t in body if t < vm.feature_count and t not in terms)
            bad += not (body and body[0] < vm.feature_count and body[0] in terms)
    del model
    ok1 = c1 and worst == 0.0 and max(d34) <= ST.TOL and bad == 0 and holds_bad == 0
    say("1", ok1, "fwd=%s const=%.1e dT1=%.1e dT2=%.1e grammar=%d hold!=168:%d/3000"
        % (c1, worst, d34[0], d34[1], bad, holds_bad))
    del sl64
    torch.cuda.empty_cache()

    # ---- 2: 1-week momentum, 16 tiles, float32 (training dtype): dense path vs ruler path
    sl32 = M.tile_slices(fp, starts, dev)
    r_old = M.score_formula(vm, mom, sl32[:16], starts[:16], univ, base, [168], policy_mod,
                            costs=M.COSTS, band=R.BAND, nav_usd=R.NAV_USD)
    r_new = D.score_dense(vm, mom, sl32[:16], starts[:16], univ, base, policy_mod, holds=(168,),
                          costs=M.COSTS, want_ic=False)
    neq = sum(int(a != b) for c in M.COSTS
              for a, b in zip(r_old["per_hold"][168][c], r_new["per_hold"][168][c]))
    say("2", neq == 0 and len(r_new["per_hold"][168]["cost0"]) == 16,
        "16 tiles x {cost0,flat6} bit-differences=%d" % neq)

    # ---- 3: IC
    E, A = M.EPISODE, feats.shape[0]
    c_ic, f_ic = [], []
    for s0 in starts:
        c_ic.append(D.ic_tile(np.ones((E, A)), base, s0))
        rows, fut = D.ic_future_rows(base, s0)
        sig = np.full((E, A), np.nan)
        sig[rows] = np.where(univ[:, s0 - M.LAG + rows].T, fut, np.nan)
        f_ic.append(D.ic_tile(sig, base, s0))
    ok3 = max(abs(x) for x in c_ic) == 0.0 and min(f_ic) >= 1.0 - 1e-9
    say("3", ok3, "const max|IC|=%.1e  true-future IC min=%.6f max=%.6f (50 tiles; C1 uses this IC)"
        % (max(abs(x) for x in c_ic), min(f_ic), max(f_ic)))

    # ---- 5: alpha calibration on the trainer's shard, float32 path
    fe_sha, fwd_sha = arr_sha(feats), arr_sha(base["fwd"])
    s = D.plant_signal(vm, feats, univ, dev)
    pb = D.plant_body(vm)
    psig = [M.window_signal(vm, pb, sl, s0, univ) for s0, sl in zip(starts, sl32)]
    g0 = {h: gross_of(psig, starts, base, h) for h in M.HOLDS}
    cal, ok5 = {}, True
    import m2h_plant as PL
    for key, tgt in TARGETS.items():
        a, hist = 1e-4, []
        for _ in range(10):
            sh = D.plant_shard(base, s, a)
            L = float(gross_of(psig, starts, sh, 168).mean() - g0[168].mean())
            hist.append((a, L))
            if abs(L - tgt) <= 0.02 * tgt:
                break
            a = a * tgt / L if L > 0 else a * 4.0
        a, L = hist[-1]
        ok5 &= abs(L - tgt) <= 0.10 * tgt
        lifts = {h: round(float(gross_of(psig, starts, sh, h).mean() - g0[h].mean()), 3) for h in M.HOLDS}
        cal[key] = dict(alpha=a, target_bpd=tgt, lift_bpd=round(L, 4), lift_by_hold=lifts,
                        iterations=len(hist), s_sha16=PL.s_sha(s), expr=vm.describe(pb, list(M.TERMINALS)))
        del sh
    x = torch.from_numpy(np.ascontiguousarray(feats)).to(device=dev, dtype=torch.float64)
    mraw = np.where(univ, vm.execute(mom, x).detach().cpu().numpy(), np.nan)
    del x
    rho = mean_cs(s, mraw)
    np.save(os.path.join(TV1, "TV1_S.npy"), s)
    json.dump(dict(cal, base_gross_bpd={h: round(float(g0[h].mean()), 3) for h in M.HOLDS},
                   corr_s_mom168=round(rho, 4), written=time.strftime("%Y-%m-%dT%H:%M:%S%z")),
              open(os.path.join(TV1, "TV1_ALPHA.json"), "w"), indent=1)
    say("5", ok5, "p32 a=%.3e lift=%.2f  p8 a=%.3e lift=%.2f bp/d; lift by hold p8 %s; rho(s,mom168)=%.3f"
        % (cal["p32"]["alpha"], cal["p32"]["lift_bpd"], cal["p8"]["alpha"], cal["p8"]["lift_bpd"],
           cal["p8"]["lift_by_hold"], rho))

    # ---- 6: shuffled null
    msig = [M.window_signal(vm, mom, sl, s0, univ) for s0, sl in zip(starts, sl32)]
    tstat = lambda v: float(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))))
    g_real = gross_of(msig, starts, base, 168)
    g_null = gross_of(msig, starts, D.null_shard(base, univ, D.PLANT_SEED), 168)
    say("6", abs(tstat(g_null)) < 2.0, "mom168 TRAIN gross: null %.2f bp/d t=%.2f (real %.2f t=%.2f)"
        % (g_null.mean(), tstat(g_null), g_real.mean(), tstat(g_real)))

    # ---- 7: TRAIN only
    src = "".join(open(os.path.join(HERE, f), encoding="utf-8").read()
                  for f in ("m2h_dense_trainer.py", "m2h_dense_common.py", "m2h_plant.py"))
    leaks = [k for k in ("DEV2026", ".npz", "202607", "202609", "2026-07") if k in src]
    import pandas as pd
    ok7 = pd.Timestamp(np.max(ts)) < pd.Timestamp("2026-01-01") and arr_sha(feats) == fe_sha and \
        arr_sha(base["fwd"]) == fwd_sha and not leaks
    say("7", ok7, "panel max %s; feats / base fwd unchanged=%s; DEV/OOS refs in TV1 code=%s"
        % (str(ts.max())[:10], arr_sha(feats) == fe_sha and arr_sha(base["fwd"]) == fwd_sha, leaks or "none"))
    del sl32, psig, msig
    torch.cuda.empty_cache()

    # ---- F: funding_level == 0 (TV1 s8)
    lvl, fr = feats[:, 4, :], grids["funding_rate"]
    T = lvl.shape[1]
    pool = univ & np.isfinite(lvl)
    z = pool & (lvl == 0)
    idx = np.where(np.isfinite(fr), np.arange(T)[None, :], -10 ** 9)
    last = np.maximum.accumulate(idx, axis=1)
    stale = (np.arange(T)[None, :] - last) >= 8
    lastval = np.take_along_axis(np.nan_to_num(fr, nan=1.0), np.clip(last, 0, T - 1), axis=1)
    nz = max(int(z.sum()), 1)
    RES["F"] = dict(zero_frac=float(z.sum() / pool.sum()), no_settle_8h=float((z & stale).sum() / nz),
                    zero_settlement=float((z & ~stale & (lastval == 0)).sum() / nz),
                    raw_finite=float(np.isfinite(fr).mean()),
                    pool_nan=float((univ & ~np.isfinite(lvl)).sum() / max(int(univ.sum()), 1)))
    print("F1 TRAIN TOP100 funding_level==0: %.2f%% of finite pool cells; NaN (VM may read as 0): %.2f%% "
          "of pool cells; raw funding finite on %.1f%% of all cells"
          % (100 * RES["F"]["zero_frac"], 100 * RES["F"]["pool_nan"], 100 * RES["F"]["raw_finite"]), flush=True)
    print("F2 of those zeros: %.1f%% have no settlement in the last 8 h; %.1f%% sit on a settlement of exactly 0"
          % (100 * RES["F"]["no_settle_8h"], 100 * RES["F"]["zero_settlement"]), flush=True)
    del base, w, feats, fp
    torch.cuda.empty_cache()

    # ---- 4 / 8: smoke through the wrapper, 2 updates x 32, a fresh dir each time
    stamp = time.strftime("%H%M%S")
    smoke = {}
    for key, trainer, mode, extra in (
            ("dense", "dense", "p8", ["--reward", "ic", "--select", "full", "--tiles", "16"]),
            ("old", "old", "p32", [])):
        name = f"TV1_SMOKE_{key.upper()}_{stamp}"
        cmd = [sys.executable, "-I", os.path.join(HERE, "m2h_plant.py"), "--trainer", trainer,
               "--mode", mode, "--", name, "--updates", "2", "--batch", "32", "--seed", "20261009",
               "--checkpoint-every", "1"] + extra
        t0 = time.time()
        p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
        out_dir = os.path.join(DT.ROOT, "artifacts", "m2h_v1", name)
        open(os.path.join(TV1, f"smoke_{key}.log"), "w", encoding="utf-8").write(p.stdout + p.stderr)
        rows = []
        lp = os.path.join(out_dir, "BATCH_LOG.jsonl")
        if os.path.exists(lp):
            rows = [json.loads(l) for l in open(lp, encoding="utf-8") if l.strip()]
        cks = glob.glob(os.path.join(out_dir, "checkpoint_*.pt"))
        smoke[key] = dict(rc=p.returncode, rows=len(rows), cks=len(cks),
                          sec=[r["seconds"] for r in rows], wall=round(time.time() - t0, 1),
                          plant=os.path.exists(os.path.join(out_dir, "PLANT_MANIFEST.json")),
                          grad=[r.get("elite_count") for r in rows], z=[r.get("z_count") for r in rows],
                          valid=[r["n_valid"] for r in rows],
                          holds=[r["hold_hist"] for r in rows])
    d = smoke["dense"]
    ok4 = d["rows"] == 2 and all(g == 32 for g in d["grad"]) and d["z"] == d["valid"]
    say("4", ok4, "dense full-sample: gradient rows %s of batch 32, z rows %s = valid %s"
        % (d["grad"], d["z"], d["valid"]))
    ok8 = all(s["rc"] == 0 and s["rows"] == 2 and s["cks"] >= 2 and s["plant"] for s in smoke.values())
    h168 = all(h[M.HOLDS.index(168)] == 32 for h in d["holds"])
    # per-update seconds scale with formulas x tiles: dense 128 x 16, old 256 x 4
    est = dict(dense=float(np.mean(d["sec"][1:] or d["sec"])) * 128 / 32 if d["sec"] else None,
               old=float(np.mean(smoke["old"]["sec"][1:] or smoke["old"]["sec"])) * 256 / 32
               if smoke["old"]["sec"] else None)
    say("8", ok8 and h168, "rc dense/old=%d/%d ckpts=%d/%d sec/upd dense %s old %s; dense all H168=%s; "
        "est. full-size sec/upd dense(128x16)=%s old(256x4)=%s"
        % (d["rc"], smoke["old"]["rc"], d["cks"], smoke["old"]["cks"], d["sec"], smoke["old"]["sec"],
           h168, None if est["dense"] is None else round(est["dense"], 1),
           None if est["old"] is None else round(est["old"], 1)))
    RES["smoke"], RES["est_sec_per_update"] = smoke, est

    all_ok = all(v["ok"] for k, v in RES.items() if isinstance(v, dict) and "ok" in v)
    RES.update(all_pass=bool(all_ok), seconds=round(time.time() - t_all, 1),
               written=time.strftime("%Y-%m-%dT%H:%M:%S%z"), dev_read=False, oos_read=False)
    json.dump(RES, open(os.path.join(TV1, "TV1_SELFTEST.json"), "w"), indent=1, default=str)
    print("SELFTEST %s %.0fs" % ("ALL_PASS" if all_ok else "FAIL", time.time() - t_all), flush=True)


if __name__ == "__main__":
    main()
