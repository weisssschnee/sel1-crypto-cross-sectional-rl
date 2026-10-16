"""TV1 A table: reward signal-to-noise, 20 cells (TV1_ORDER s2 + A2 + TV1_A2_BASIN4_ORDER s1).

Formulas: BASIN3_H1_20261009 update 0 (256, three families). Each is scored on all 50 TRAIN
tiles at H168 (hold token overridden) and at every hold (for the own-hold diagnostic), storing
per tile net (flat6), gross (cost0), cost (flat6 charge) in bp/day and IC (D.ic_tile).
Only formulas valid on all 50 tiles enter. Tiles split by parity: the 25 even tiles simulate
the training reward, the 25 odd tiles are the truth. 2000 simulations; per simulation k tiles
are drawn from the 25 (common draws across cells), the cell's gradient weights are formed, and
Pearson(weight, truth) is taken inside each family and averaged over the three families.

Cells: {net, gross, IC} x {per-family q95, whole-batch q95, full-sample z+-3} x {4, 16}
       + C1 = z(0.5 z(IC) + 0.5 z(gross)), C2 = same with net (full sample, 16 tiles).
Diagnostics (not candidates): net at the formula's own hold token x {fam q95 x 4 = BASIN3's
own reward, full x 16}; each cell's ceiling = the same rule on all 25 simulation tiles.
B: highest corr with truth gross among {gross, IC} x {3 selections} x 16 + C1; all < 0.1 -> C1.
Reads TRAIN only. Writes A_TABLE.json + A_SCORES.npz to the TV1 dir; prints one line.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m2h_trainer as TR            # first: puts the v4 / model roots on sys.path
import m2h_common as M
import m2h_dense_common as D
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import XsTsCsStackVM, CandidateLocalInvalid
from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod

SRC = os.path.join(TR.ROOT, "artifacts", "m2h_v1", "BASIN3_H1_20261009", "FORMULAS.jsonl")
OUT = os.path.join(TR.ROOT, "artifacts", "m2h_v1", "TV1_20261010")
N_SIM = 2000
SIM_SEED = 20261010
EPS = TR.EPSILON


def make_vm():
    import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vm_mod
    vm_mod.BLOCK_ELEMS = 1 << 25
    vm_mod.EMA_KERNEL_CACHE_MAX = 1
    return XsTsCsStackVM(len(M.TERMINALS), windows=M.WINDOWS)


def score_all(device, limit=0):
    vm = make_vm()
    syms, ts, grids = M.load_train_grids(TR.PANEL)
    w = M.build_world(grids)
    starts = M.tile_starts(w["feats"].shape[2])
    assert len(starts) == 50, len(starts)
    slices = M.tile_slices(M.pad_feats(w["feats"]), starts, device)
    rows = []
    with open(SRC, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["update"] != 0:
                break
            rows.append(r)
    assert len(rows) == 256, len(rows)
    rows.sort(key=lambda r: M.BASIN_NAMES.index(r["basin"]))      # stable: family blocks
    if limit:                                                       # debug: first N per family
        rows = [r for b in M.BASIN_NAMES for r in [x for x in rows if x["basin"] == b][:limit]]
    n = len(rows)
    H = list(M.HOLDS)
    net = np.full((n, len(H), 50), np.nan)
    gross = np.full((n, len(H), 50), np.nan)
    cost = np.full((n, len(H), 50), np.nan)
    ic = np.full((n, 50), np.nan)
    valid = np.zeros(n, dtype=bool)
    t0 = time.time()
    for i, r in enumerate(rows):
        body = list(TR.strip_hold(vm, list(r["tokens"])))
        try:
            res = D.score_dense(vm, body, slices, starts, w["univ"], w["shard"], policy_mod, holds=H)
        except (M.ZeroCoverage, CandidateLocalInvalid):
            continue
        for j, h in enumerate(H):
            net[i, j] = [D.bpd(x) for x in res["per_hold"][h]["flat6"]]
            gross[i, j] = [D.bpd(x) for x in res["per_hold"][h]["cost0"]]
            cost[i, j] = [1e4 * p["cost_flat6"] / (M.EPISODE / 24.0) for p in res["parts"][h]]
        ic[i] = res["ic"]
        valid[i] = True
    fam = np.array([r["basin"] for r in rows])
    own = np.array([H.index(int(r["hold_h"])) for r in rows])
    np.savez(os.path.join(OUT, "A_SCORES.npz"), net=net, gross=gross, cost=cost, ic=ic, valid=valid,
             fam=fam, own=own, holds=np.array(H), starts=np.array(starts))
    return dict(net=net, gross=gross, cost=cost, ic=ic, valid=valid, fam=fam, own=own,
                seconds=time.time() - t0)


def weights(r, sel, groups):
    if sel == "famq95":
        return D.q95_family(r, groups, EPS)
    if sel == "globq95":
        return D.q95_global(r, EPS)
    return D.zfam(r, np.ones(len(r), dtype=bool), groups)


def table(S):
    v = S["valid"]
    fam = S["fam"][v]
    groups = [int((fam == b).sum()) for b in M.BASIN_NAMES]
    i168 = list(M.HOLDS).index(168)
    own = S["own"][v]
    pick = lambda a: a[v][:, i168]
    met = dict(net=pick(S["net"]), gross=pick(S["gross"]), ic=S["ic"][v])
    a_own = np.arange(int(v.sum()))
    met_own = S["net"][v][a_own, own]
    cost_own = S["cost"][v][a_own, own]
    ev, od = np.arange(0, 50, 2), np.arange(1, 50, 2)
    truth_g = met["gross"][:, od].mean(1)
    truth_c = pick(S["cost"])[:, od].mean(1)
    truth_c_own = cost_own[:, od].mean(1)

    cells = [(m, s, k) for k in (4, 16) for m in ("net", "gross", "ic")
             for s in ("famq95", "globq95", "full")]
    names = ["%s|%s|%d" % c for c in cells] + ["C1|full|16", "C2|full|16"]
    diag = ["netown|famq95|4", "netown|full|16"]
    acc = {n: ([], []) for n in names + diag}
    c1_vs = dict(ic=[], gross=[])

    def wfor(name, tiles):
        m, s, k = name.split("|")
        mean = lambda a: a[:, tiles].mean(1)
        if m == "C1":
            return D.combo(mean(met["ic"]), mean(met["gross"]), np.ones(len(fam), bool), groups)
        if m == "C2":
            return D.combo(mean(met["ic"]), mean(met["net"]), np.ones(len(fam), bool), groups)
        if m == "netown":
            return weights(mean(met_own), s, groups)
        return weights(mean(met[m]), s, groups)

    rng = np.random.default_rng(SIM_SEED)
    for _ in range(N_SIM):
        draws = {k: ev[np.sort(rng.choice(25, size=k, replace=False))] for k in (4, 16)}
        ws = {}
        for name in names + diag:
            k = int(name.split("|")[2])
            w = wfor(name, draws[k])
            ws[name] = w
            tc = truth_c_own if name.startswith("netown") else truth_c
            acc[name][0].append(D.wcorr_fam(w, truth_g, groups))
            acc[name][1].append(D.wcorr_fam(w, tc, groups))
        c1_vs["ic"].append(D.wcorr_fam(ws["C1|full|16"], ws["ic|full|16"], groups))
        c1_vs["gross"].append(D.wcorr_fam(ws["C1|full|16"], ws["gross|full|16"], groups))

    out = []
    for name in names + diag:
        g, c = np.asarray(acc[name][0]), np.asarray(acc[name][1])
        w25 = wfor(name, ev)
        tc = truth_c_own if name.startswith("netown") else truth_c
        out.append(dict(cell=name, diag=name in diag,
                        corr_gross=[round(float(g.mean()), 4), round(float(g.std()), 4)],
                        corr_cost=[round(float(c.mean()), 4), round(float(c.std()), 4)],
                        ceiling_k25=dict(gross=round(D.wcorr_fam(w25, truth_g, groups), 4),
                                         cost=round(D.wcorr_fam(w25, tc, groups), 4))))
    cand = [n for n in names if n.split("|")[0] in ("gross", "ic") and n.endswith("|16")] + ["C1|full|16"]
    by = {o["cell"]: o["corr_gross"][0] for o in out}
    best = max(cand, key=lambda n: by[n])
    choice = best if by[best] >= 0.1 else "C1|full|16"
    return dict(cells=out, groups_valid=groups, n_valid=int(v.sum()),
                c1_vs_ic_full16=[round(float(np.mean(c1_vs["ic"])), 4), round(float(np.std(c1_vs["ic"])), 4)],
                c1_vs_gross_full16=[round(float(np.mean(c1_vs["gross"])), 4), round(float(np.std(c1_vs["gross"])), 4)],
                b_candidates={n: by[n] for n in cand}, b_best=best, b_choice=choice,
                b_fallback_used=choice != best)


def main():
    import argparse
    global OUT, N_SIM
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="debug: N formulas per family, n_sim 50")
    args = ap.parse_args()
    if args.limit:
        OUT, N_SIM = os.path.join(OUT, "debug"), 50
    os.makedirs(OUT, exist_ok=True)
    device = torch.device("cuda")
    S = score_all(device, args.limit)
    t = table(S)
    t.update(schema="TV1_A_TABLE_V1", source=SRC, n_sim=N_SIM, sim_seed=SIM_SEED,
             split="even tiles simulate, odd tiles are the truth", hold=168,
             corr="Pearson(weight, truth) inside each family, mean over T1/T2/T3",
             score_seconds=round(S["seconds"], 1), dev_read=False, oos_read=False,
             written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(t, open(os.path.join(OUT, "A_TABLE.json"), "w", encoding="utf-8"), indent=1)
    print("A done valid=%d B=%s (%.3f) fallback=%s %.0fs" % (t["n_valid"], t["b_choice"],
          t["b_candidates"][t["b_best"]], t["b_fallback_used"], S["seconds"]), flush=True)


if __name__ == "__main__":
    main()
