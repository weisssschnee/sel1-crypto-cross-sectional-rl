"""BASIN5 step 1 ruler check (merged plan, user-approved 2026-10-13 "都批"). CPU only, TRAIN only,
the 50 fixed tiles (b5_fixed). Gates, in order, written before any result was seen:

 G0  the plant formula s = DELTA(TS_MEAN(log_quote_volume,168),168) is legal under m2h_b5_grammar,
     fits MAX_PROGRAM_LEN, every token is allowed by the T1 legality table on its path, and its
     own signal reproduces s (corr > 0.99). Fail -> stop (the plan's swap-s branch needs a new
     plant; this script does not pick one).
 G1  training ruler (reward b = full-pool gross per tile): s against each T1 attractor, paired
     difference of per-tile gross over the 50 tiles, one-sample t > 2 against every attractor.
     Attractors = T1 frozen top-5 of B5_PC_P8 and B5_PC_P8_ENT + each run's T1 mode (last
     PROGRESS_16 row and the whole-run most frequent T1 body). Strengths p8 -> p16 -> p32
     (p16 alpha calibrated by b5_fixed.calibrate_p16); the first passing one is used. None ->
     stop. Also reported, not gating: 50-tile gross t of each, the 16-tile bootstrap win rate
     (what one update sees), s's rank in a random legal T1 pool.
 G2  freeze ruler (conc K10 TRAIN net, flat6, dedup on the 50-tile vector), in the G1 world, on
     the pool = both runs' frozen T1 pools (256 at seed 20261108, distinct) + s + the T1 formulas
     with logged corr_s > 0.7 + the logged-gross_t top 100 (both runs, distinct) + attractors.
     Pass = s or a formula with 50-tile corr > 0.7 in the top 5 -> HoF rank = conc_k10_net.
     Fail -> the same test under the G1 ruler (50-tile full gross t); pass -> HoF rank =
     full_gross_t (plan's G2 branch); fail too -> stop (AGENTS.md 10-13 veto: a "not found"
     could not be told apart from the ruler).
 neutral column (not gating): s, attractors and corr>0.7 formulas re-scored after the signal is
     residualised on size / log price / beta.
Writes ROOT\\artifacts\\m2h_v1\\B5_GATE1\\GATE1_REPORT.json and DECISION.json. Exit 0 = proceed,
20 = stop. No DEV, no OOS.

    python b5_gate1.py [--workers 14] [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b5_fixed as F

OUT_DIR = os.path.join(F.A_ROOT, "B5_GATE1")
RUNS = ("B5_PC_P8", "B5_PC_P8_ENT")
STRENGTHS = ("p8", "p16", "p32")
N_RANDOM = 200
RANDOM_SEED = 20261113
T_PASS = 2.0


def log(msg):
    print(msg, flush=True)


def read_formulas(run):
    """Per T1 body: logged gross_t / corr_s lists and counts; plus the whole-run mode."""
    agg = collections.defaultdict(lambda: dict(gt=[], cs=[], n=0, expr=None))
    cnt = collections.Counter()
    for line in open(os.path.join(F.A_ROOT, run, "FORMULAS.jsonl"), encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["basin"] != "T1":
            continue
        toks = r["tokens"]
        end = toks.index(41) if 41 in toks else len(toks) - 1
        body = tuple(int(x) for x in toks[:end + 1])
        cnt[body] += 1
        if r.get("invalid_reason") is not None or r.get("gross_t") is None:
            continue
        a = agg[body]
        a["gt"].append(r["gross_t"])
        a["n"] += 1
        a["expr"] = r["expr"]
        if r.get("corr_s") is not None:
            a["cs"].append(r["corr_s"])
    return agg, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="test: cap the G2 / random pools")
    a = ap.parse_args()
    t_all = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    import torch
    torch.set_num_threads(8)
    TR, M, D, W, R = F._imports()
    import m2h_b5_grammar as G
    import tv1_round as TV
    import m2h_b5_trainer_ent as TE
    from okx_alpha_discovery.adapters.xs_tscs_sampler_v2 import build_legality_table

    if not os.path.exists(os.path.join(F.CACHE_DIR, "CACHE_MANIFEST.json")):
        log("building cache ...")
        F.build_cache()
    man, arr = F.load_cache()
    log("cache ok: %d symbols, %d hours, s sha %s" % (len(man["symbols"]), man["hours"], man["s_sha16"]))
    vm = F.make_vm()
    E = M.EPISODE
    rep = dict(schema="B5_GATE1_REPORT_V1", plan="merged plan 10-13 (user: 都批)", tiles=50, limit=a.limit,
               starts=man["starts"], dev_read=False, oos_read=False)

    # ---------------------------------------------------------------- G0
    pb = tuple(int(x) for x in D.plant_body(vm))
    tab = build_legality_table(vm, max_len=M.MAX_PROGRAM_LEN, available_terminals=M.basin_terms("T1"),
                               available_windows=M.WINDOWS)
    depth, win, path_ok = 0, 0, True
    import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vmm
    arity = dict(vmm.OPERATOR_ARITIES)
    for step, t in enumerate(pb):
        rem = M.MAX_PROGRAM_LEN - step
        if not bool(torch.isfinite(tab.mask[depth, win, rem, t])):
            path_ok = False
            break
        if vm.is_terminal(t):
            depth, win = depth + 1, 0
        elif vm.is_window(t):
            win = 1
        elif not vm.is_end(t):
            depth, win = depth - arity[vm.operator_name(t)] + 1, 0
    sl4 = F.tile_slices(arr, man["starts"][:4])
    univ = np.array(arr["univ"])
    s = np.array(arr["s"])
    self_corr = float(np.mean([D.corr_with_s(M.window_signal(vm, list(pb), x, s0, univ), s, s0)
                               for s0, x in zip(man["starts"][:4], sl4)]))
    g0 = dict(expr=vm.describe(list(pb), list(M.TERMINALS)), body=list(pb), n_tokens=len(pb) - 1,
              grammar=G.check(vm, list(pb)), table_path_ok=path_ok, self_corr=round(self_corr, 4))
    g0["passed"] = bool(g0["grammar"] is None and path_ok and len(pb) <= M.MAX_PROGRAM_LEN
                        and self_corr > 0.99)
    rep["G0"] = g0
    log("G0 %s %s corr %.4f" % ("PASS" if g0["passed"] else "FAIL", g0["expr"], self_corr))
    if not g0["passed"]:
        return finish(rep, dict(proceed=False, reason="G0 fail: s not legal / samplable"), t_all)

    # ---------------------------------------------------------------- candidates
    info = collections.OrderedDict()           # body -> dict(expr, src set)

    def add(body, src, expr=None):
        body = tuple(int(x) for x in body)
        if G.check(vm, list(body)) is not None:   # the freeze drops grammar-illegal formulas
            return
        if body not in info:
            info[body] = dict(expr=expr or vm.describe(list(body), list(M.TERMINALS)), src=set())
        info[body]["src"].add(src)
    add(pb, "s")
    attractors = []
    logged = {}
    for run in RUNS:
        rr = json.load(open(os.path.join(F.A_ROOT, run, "ROUND_REPORT.json"), encoding="utf-8"))
        for f in rr["frozen"]["T1"]:
            add(f["body"], "att_frozen:" + run)
            attractors.append(tuple(f["body"]))
        agg, cnt = read_formulas(run)
        logged[run] = agg
        mode_all = cnt.most_common(1)[0][0]
        if G.check(vm, list(mode_all)) is None:
            add(mode_all, "att_mode_run:" + run)
            attractors.append(mode_all)
        last = rr["progress_16"][-1]["families"]["T1"]["mode"]["expr"]
        for b, v in agg.items():
            if v["expr"] == last:
                add(b, "att_mode_last:" + run)
                attractors.append(b)
                break
    attractors = list(dict.fromkeys(attractors))
    merged = collections.defaultdict(lambda: dict(gt=[], cs=[]))
    for run in RUNS:
        for b, v in logged[run].items():
            merged[b]["gt"] += v["gt"]
            merged[b]["cs"] += v["cs"]
            merged[b]["expr"] = v["expr"]
    hi_corr = [b for b, v in merged.items() if v["cs"] and np.mean(v["cs"]) > 0.7]
    for b in hi_corr:
        add(b, "logged_corr_gt07")
    top_gt = sorted(merged, key=lambda b: -np.mean(merged[b]["gt"]))[:100]
    for b in top_gt[: (a.limit or 100)]:
        add(b, "logged_gt_top100")
    for run, TRm in (("B5_PC_P8", TR), ("B5_PC_P8_ENT", TE)):
        model, ck, ck_u = TV.load_final(TRm, os.path.join(F.A_ROOT, run), torch.device("cpu"))
        pool = TV.final_pool(TRm, model, vm, torch.device("cpu"), TV.PER_FAMILY)["T1"]
        for body, hold in pool[: (a.limit or 10 ** 6)]:
            add(body, "frozen_pool:" + run)
        rep.setdefault("frozen_pool_distinct", {})[run] = len(pool)
        del model
    # random legal T1 pool (untrained policy, the trainer's init: seed 20261009, output bias)
    from train_basin_riskpg_v1 import make_basin_model, apply_output_bias
    import model_core.alphagpt as alphagpt_mod
    from model_core.config import ModelConfig
    from model_core.vocab import FormulaVocab
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import OPERATOR_NAMES
    ModelConfig.DEVICE = torch.device("cpu")
    torch.manual_seed(20261009)
    names = (list(OPERATOR_NAMES) + [f"W{x}" for x in M.WINDOWS] + ["END"] + M.HOLD_NAMES
             + list(M.BASIN_TOKEN_NAMES))
    m0, _ = make_basin_model(alphagpt_mod=alphagpt_mod, ModelConfig=ModelConfig, FormulaVocab=FormulaVocab,
                             feature_names=list(M.TERMINALS), operator_token_names=names,
                             max_len=M.MAX_PROGRAM_LEN, device=torch.device("cpu"))
    apply_output_bias(m0, len(M.TERMINALS), len(OPERATOR_NAMES), len(M.WINDOWS), TR.OP_BIAS, TR.END_BIAS)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(RANDOM_SEED)
    rand, tries = [], 0
    n_rand = min(N_RANDOM, a.limit) if a.limit else N_RANDOM
    while len(rand) < n_rand and tries < 12:
        with torch.no_grad():
            b = TR.sample_batch_with_hold_basins(m0, vm, [tab], [512], device=torch.device("cpu"),
                                                 generator=gen, basins=("T1",))
        for t in b.formulas:
            body = tuple(TR.strip_hold(vm, list(t)))
            if body not in info and body not in rand and G.check(vm, list(body)) is None:
                rand.append(body)
        tries += 1
    rand = rand[:n_rand]
    del m0
    log("candidates: G2 pool %d (attractors %d, corr>0.7 %d), random %d"
        % (len(info), len(attractors), len(hi_corr), len(rand)))

    # ---------------------------------------------------------------- p16 alpha
    if not os.path.exists(F.P16_ALPHA):
        cal = F.calibrate_p16()
    else:
        cal = json.load(open(F.P16_ALPHA, encoding="utf-8"))
    rep["p16_calibration"] = cal
    log("p16 alpha %.4e lift %.3f (target 16); TV1 check %s" % (cal["p16"]["alpha"], cal["p16"]["lift_bpd"],
                                                                 cal["check_tv1"]))
    if not cal["ok"]:
        return finish(rep, dict(proceed=False, reason="p16 calibration off by >10%"), t_all)

    # ---------------------------------------------------------------- G1 (phase A)
    ta = [dict(id="A%d" % i, body=list(b), worlds=list(STRENGTHS), conc=True)
          for i, b in enumerate([pb] + attractors)]
    ra = F.run_pool(ta, workers=a.workers, log=log)
    idx = {tuple(t["body"]): t["id"] for t in ta}
    rs = ra[idx[pb]]
    if not rs["ok"]:
        return finish(rep, dict(proceed=False, reason="s not scorable: %s" % rs["reason"]), t_all)
    rng = np.random.default_rng(RANDOM_SEED)
    draws = [np.sort(rng.choice(50, size=16, replace=False)) for _ in range(4000)]
    g1 = {}
    for wn in STRENGTHS:
        fs = np.asarray(rs["w"][wn]["fg"])
        rows, ok = [], True
        gts_s = np.array([F.gross_t(fs[d]) for d in draws])
        best_other = np.full(len(draws), -np.inf)
        for b in attractors:
            r_ = ra[idx[b]]
            if not r_["ok"]:
                rows.append(dict(expr=info[b]["expr"], ok=False, reason=r_["reason"]))
                continue
            fa = np.asarray(r_["w"][wn]["fg"])
            d = fs - fa
            gts_a = np.array([F.gross_t(fa[x]) for x in draws])
            best_other = np.maximum(best_other, gts_a)
            t_ = F.tstat(d)
            ok &= bool(t_ > T_PASS)
            rows.append(dict(expr=info[b]["expr"], src=sorted(info[b]["src"]), paired_t=round(t_, 3),
                             d_gross_bpd=round(F.bpd(d.mean()), 3), gross_t50=round(F.gross_t(fa), 4),
                             gross_bpd=round(F.bpd(fa.mean()), 3), corr50=round(float(np.mean(r_["corr"])), 4),
                             win16_s_over_a=round(float((gts_s > gts_a).mean()), 4)))
        g1[wn] = dict(passed=ok, s_gross_t50=round(F.gross_t(fs), 4), s_gross_bpd=round(F.bpd(fs.mean()), 3),
                      s_win16_over_all=round(float((gts_s > best_other).mean()), 4), attractors=rows,
                      alpha=F.alpha_of(wn)["alpha"])
        log("G1 %s %s s t50 %.3f %.2fbpd | paired t %s" % (wn, "PASS" if ok else "fail", g1[wn]["s_gross_t50"],
                                                         g1[wn]["s_gross_bpd"],
                                                         [x.get("paired_t") for x in rows]))
    rep["G1"] = dict(rule="paired per-tile full-pool gross difference s - attractor over the 50 tiles, "
                          "t > %.0f against every T1 attractor" % T_PASS, by_strength=g1)
    chosen = next((wn for wn in STRENGTHS if g1[wn]["passed"]), None)
    if chosen is None:
        return finish(rep, dict(proceed=False, reason="G1 fails at p8, p16 and p32 (uncovered branch)"), t_all)
    log("G1 strength = %s" % chosen)

    # ---------------------------------------------------------------- phase B (chosen world)
    neutral_set = set([pb] + attractors + list(hi_corr))
    tb = [dict(id="B%d" % i, body=list(b), worlds=[chosen], conc=True, neutral=b in neutral_set)
          for i, b in enumerate(list(info) + rand)]
    rb = F.run_pool(tb, workers=a.workers, log=log)
    byb = {tuple(t["body"]): rb[t["id"]] for t in tb}

    def table(bodies, key):
        rows = []
        for b in bodies:
            r_ = byb[b]
            if not r_["ok"]:
                continue
            x = r_["w"][chosen]
            val = float(np.mean(x["cn"])) if key == "cn" else F.gross_t(x["fg"])
            rows.append((val, b, x["cn"] if key == "cn" else x["fg"], r_))
        rows.sort(key=lambda z: -z[0])
        kept = []
        for z in rows:
            if any(np.array_equal(z[2], k[2]) for k in kept):
                continue
            kept.append(z)
        return kept

    def g2_on(key):
        kept = table(list(info), key)
        top = kept[:5]
        hit = [dict(rank=i + 1, expr=info[b]["expr"], corr50=round(float(np.mean(r_["corr"])), 4),
                    is_s=b == pb) for i, (v, b, _, r_) in enumerate(top)]
        rank_s = next((i + 1 for i, z in enumerate(kept) if z[1] == pb), None)
        rank_hi = [i + 1 for i, z in enumerate(kept) if float(np.mean(z[3]["corr"])) > 0.7]
        rt = table(rand + [pb], key)
        rank_s_rand = next((i + 1 for i, z in enumerate(rt) if z[1] == pb), None)
        return dict(passed=any(h["is_s"] or h["corr50"] > 0.7 for h in hit), top5=[
            dict(h, value=round(top[i][0], 6), src=sorted(info[top[i][1]]["src"])) for i, h in enumerate(hit)],
            pool_scored=len(kept), rank_s=rank_s, ranks_corr_gt07=rank_hi[:20],
            rank_s_in_random=rank_s_rand, random_scored=len(rt))
    g2c = g2_on("cn")
    g2t = g2_on("fg")
    rep["G2"] = dict(world=chosen, pool_rule="frozen T1 pools of both runs (seed 20261108, 256) + s + logged "
                                             "corr>0.7 + logged gross_t top 100 + attractors",
                     conc_k10_net=g2c, full_gross_t=g2t)
    log("G2 conc %s (s rank %s, corr>.7 ranks %s) | gross_t %s (s rank %s, corr>.7 ranks %s)"
        % ("PASS" if g2c["passed"] else "fail", g2c["rank_s"], g2c["ranks_corr_gt07"][:6],
           "PASS" if g2t["passed"] else "fail", g2t["rank_s"], g2t["ranks_corr_gt07"][:6]))
    neut = []
    for b in [pb] + attractors + list(hi_corr):
        r_ = byb.get(b)
        if not r_ or not r_["ok"] or "neutral" not in r_:
            continue
        x = r_["w"][chosen]
        neut.append(dict(expr=info[b]["expr"], src=sorted(info[b]["src"]), corr50=round(float(np.mean(r_["corr"])), 4),
                         gross_t50=round(F.gross_t(x["fg"]), 4), gross_bpd=round(F.bpd(np.mean(x["fg"])), 3),
                         neutral_gross_t50=round(F.gross_t(r_["neutral"][chosen]), 4),
                         neutral_gross_bpd=round(F.bpd(np.mean(r_["neutral"][chosen])), 3),
                         conc_net_bpd=round(F.bpd(np.mean(x["cn"])), 3)))
    rep["neutral_column"] = dict(note="not gating; signal residualised per hour on rank(size), rank(log price), "
                                      "beta (720 h trailing)", world=chosen, rows=neut)
    if g2c["passed"]:
        hof = "conc_k10_net"
    elif g2t["passed"]:
        hof = "full_gross_t"
    else:
        return finish(rep, dict(proceed=False, plant=chosen, reason="G2 fails under both rulers: neither the "
                                "freeze ruler nor the training ruler puts s or a corr>0.7 relative in the "
                                "top 5 (AGENTS.md 10-13 veto)"), t_all)
    return finish(rep, dict(proceed=True, plant=chosen, alpha=F.alpha_of(chosen)["alpha"], hof_rank=hof,
                            reason="G0 pass; G1 pass at %s; G2 %s" % (chosen, "pass (conc)" if hof == "conc_k10_net"
                                                                        else "fail on conc, pass on gross t")),
                  t_all)


def finish(rep, dec, t_all):
    dec.update(schema="B5_GATE1_DECISION_V1", limit=rep.get("limit"), written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    rep["decision"] = dec
    rep["seconds"] = round(time.time() - t_all, 1)
    json.dump(rep, open(os.path.join(OUT_DIR, "GATE1_REPORT.json"), "w", encoding="utf-8"), indent=1,
              ensure_ascii=False, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    json.dump(dec, open(os.path.join(OUT_DIR, "DECISION.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log("DECISION %s %s" % ("PROCEED" if dec["proceed"] else "STOP", json.dumps(dec, ensure_ascii=False)))
    return 0 if dec["proceed"] else 20


if __name__ == "__main__":
    sys.exit(main())
