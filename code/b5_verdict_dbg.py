"""BASIN5 formal-round verdict with the hall-of-fame freeze (merged plan 10-13, ADR-0009).

    python b5_verdict_dbg.py <RUN_NAME>          e.g. B5_DBG_REAL

b5_verdict.py is imported and NOT modified; two things are swapped in before its main() runs:
  TR       m2h_b5_trainer_dbg (the grammar-masked sampler the run trained with) for the final
           batch and the checkpoint load
  freeze   the TRAIN freeze takes its pool from HOF_RANKED.json (b5_round_dbg.py --mode real: every
           logged formula, top by logged reward + frequent + the final pool, re-scored on the 50
           fixed TRAIN tiles, ranked by the step-1 ruler, deduped, top 30 per family); level-flagged
           formulas are dropped as b5_verdict does; the first 5 that score on all 50 tiles with
           b5_verdict's books are FROZEN_B5_TOP5.json. DEV is opened only after it is on disk.
Everything else (port check, DEV tiling, paired init/final, gates, K5/K20, full pool, OKX column)
is b5_verdict.main unchanged. No OOS.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if len(sys.argv) < 2:
    raise SystemExit("usage: b5_verdict_dbg.py <RUN_NAME>")
import b5_verdict as BV                 # reads sys.argv[1] as the run name
import m2h_b5_trainer_dbg as TD

BV.TR = TD
# B5V_DBG_TEST=1: freeze-only pipeline test on a plant smoke dir (any world, the 50 TRAIN
# tiles, FROZEN_B5_TOP5_DBGTEST.json, then exit; DEV is never opened)
DBG_TEST = os.environ.get("B5V_DBG_TEST") == "1"
if DBG_TEST:
    BV.FROZEN = os.path.join(BV.RUN_DIR, "FROZEN_B5_TOP5_DBGTEST.json")


def hof_freeze(vm, device, pool_t):
    M, V, D, X = BV.M, BV.V, BV.D, BV.X
    hp = os.path.join(BV.RUN_DIR, "HOF_RANKED.json")
    if not os.path.exists(hp):
        raise SystemExit("HOF_RANKED.json missing: run b5_round_dbg.py %s --mode real first" % BV.RUN_NAME)
    hof = json.load(open(hp, encoding="utf-8"))
    if hof.get("world") != "real" and not DBG_TEST:
        raise SystemExit("HOF_RANKED.json is not from the real world")
    print("== HoF freeze on TRAIN (rank %s)" % hof["hof_rank"], flush=True)
    syms, ts, grids = M.load_train_grids(V.TRAIN_PANEL)
    w = V.world_of(grids, M.LOOKBACK)
    assert len(w["starts"]) == 50, len(w["starts"])
    slices = M.tile_slices(w["feats_pad"], w["starts"], device)
    years = [int(ts[s].year) for s in w["starts"]]
    fz = dict(schema="B5_FROZEN_TOP5_V1", run=BV.RUN_NAME, freeze="hall of fame (ADR-0009)",
              pool="HOF_RANKED.json: top 30 per family by the step-1 ruler on the 50 fixed TRAIN tiles",
              hof_rank=hof["hof_rank"], n_tiles=50, rank_ruler=hof["hof_rank"],
              conc_pool_train=int(pool_t.sum()), tile_years=years, families={}, dropped={},
              dev_read=False, oos_read=False)
    train_rows = {}
    for fam in M.BASIN_NAMES:
        t0 = time.time()
        kept, n_flag, n_bad = [], 0, 0
        for h in hof["families"][fam]:
            body = list(h["body"])
            if not BV.legal(vm, body):
                n_bad += 1
                continue
            if D.level_flag(vm, body):
                n_flag += 1
                continue
            rec = BV.score_body(vm, body, slices, w["starts"], w["univ"], w["shard"], BV.BOOKS, pool_t)
            if rec is None:
                n_bad += 1
                continue
            kept.append((h, body, rec))
            if len(kept) == 5:
                break
        fz["dropped"][fam] = dict(hof_listed=len(hof["families"][fam]), level_flag=n_flag, invalid=n_bad)
        rows = []
        for i, (h, b, rec) in enumerate(kept):
            pr = V.summary(rec[BV.bk(BV.PRIMARY)])
            rows.append(dict(rank=i + 1, hof_rank_pos=h["rank"], body=b, hold=BV.HOLD,
                             expr=vm.describe(b, list(M.TERMINALS)), n_tokens=len(b) - 1,
                             level_flag=bool(D.level_flag(vm, b)), explain=X.explain(vm, b),
                             hof_ruler_value=h["ruler_value"], hof_src=h["src"],
                             train50_conc_k10_net=round(float(np.mean(rec[BV.bk(BV.PRIMARY)]["net"])), 4),
                             train50_conc_k10_gross=pr["gross"],
                             train50_conc_k5_net=V.summary(rec["conc_k5"])["net"],
                             train50_conc_k20_net=V.summary(rec["conc_k20"])["net"],
                             train50_full_net=V.summary(rec["full"])["net"],
                             train50_full_gross=V.summary(rec["full"])["gross"],
                             conc_turnover_per_day=pr["turn"],
                             flag_A=bool(pr["gexp"] is not None and pr["gexp"] < 0.5),
                             flag_B=bool((pr["beta"] is not None and abs(pr["beta"]) > 0.3)
                                         or (pr["nexp"] is not None and abs(pr["nexp"]) > 0.1))))
        print("   %s: kept %d (level-flagged %d, invalid %d) %.0fs" % (fam, len(rows), n_flag, n_bad,
                                                                     time.time() - t0), flush=True)
        fz["families"][fam] = rows
        train_rows[fam] = [rec for _, _, rec in kept]
    blob = json.dumps(fz["families"], sort_keys=True).encode()
    fz["sha256"] = hashlib.sha256(blob).hexdigest()
    fz["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(fz, open(BV.FROZEN, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("FROZEN sha256", fz["sha256"], flush=True)
    del slices
    import torch
    torch.cuda.empty_cache()
    return fz, train_rows, years, (syms, grids, w, ts)


BV.freeze = hof_freeze

if __name__ == "__main__":
    if DBG_TEST:
        import torch
        import pyarrow.parquet as pq
        tsyms = sorted(pq.read_table(BV.V.TRAIN_PANEL, columns=["symbol"]).column(0).unique().to_pylist())
        vm_ = BV.V.make_vm()
        import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vmm
        vmm.BLOCK_ELEMS = 1 << 16                     # CPU test: bit-identical, faster
        fz = hof_freeze(vm_, torch.device("cpu"), BV.W.pool_cols(tsyms, BV.POOL_JSON))[0]
        print("DBGTEST freeze ok", {f: len(v) for f, v in fz["families"].items()}, fz["sha256"][:12])
    else:
        BV.main()
