"""Ruler winsor impact (QuantMind 10-14 message, part 2; report only, CPU, V4 TRAIN, no DEV, no OOS).
Same V4 TRAIN world, same signals and the same COMBO_V1 walk-forward signs (combo_v1.stage1, unmodified); only the
settlement grid differs:
  winsor 1  world()'s fwd_wx (m1_ruler.prepare_shard winsor_pct=1 + delisting layer) -- every earlier receipt
  winsor 0  m1_ruler.prepare_shard(winsor_pct=0) + v4_world.install_layer, asserted == world()'s fwd_x
The IC used to fit COMBO_V1's yearly signs stays on the winsorised forward returns in both runs (it is part of the
model, not of the settlement), so the difference is settlement only. No conclusion file is touched.
FROZEN_H2: not done -- gbm_xs_h1 stores only summary JSON (no DEV positions or scores); rebuilding the DEV signal
needs the LightGBM model in the alpha311 env, which is off limits.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import combo_v1 as CB
import m1_ruler as R
import v4_world as X

OUT = os.path.join(X.ROOT, "artifacts", "m2h_v1", "ENR1", "WINSOR_IMPACT.json")


def pick(rep):
    out = dict(combo={b: rep["combo"][b]["net"] for b in ("conc_k10", "full")},
               mom1w={b: rep["mom1w_fixed_plus"][b]["net"] for b in ("conc_k10", "full")},
               combo_k10_by_year={str(y): rep["by_year"][y]["combo"]["conc_k10"]["net"] for y in CB.YEARS},
               mom1w_k10_by_year={str(y): rep["by_year"][y]["mom1w"]["conc_k10"]["net"] for y in CB.YEARS},
               turn_k10=rep["combo"]["conc_k10"]["turn"], gate_passed=rep["gate"]["passed"])
    return out


def main():
    t0 = time.time()
    if os.path.exists(OUT):
        raise SystemExit("%s exists" % OUT)
    w = X.load(X.GRID)
    X.world(w, feats=False)
    sh1 = w["shard"]
    sh0 = dict(fwd=sh1["fwd"], eligible=sh1["eligible"], fi=sh1["fi"], fr=sh1["fr"])
    R.prepare_shard(sh0, 0)
    X.install_layer(sh0, w["dead"])
    if not np.array_equal(sh0["fwd_wx"], sh1["fwd_x"], equal_nan=True):
        raise SystemExit("winsor 0 path != fwd_x")
    S = CB.V.handwritten(w["g"], w["ts"])
    S = {k: S[k] for k in CB.CELLS}
    del w["g"]
    fut = CB.fwd_cum(sh1["fwd_wx"])
    Z = CB.zsets(S, w["univ"])
    res = {}
    for name, sh in (("winsor1", sh1), ("winsor0", sh0)):
        w["shard"] = sh
        res[name] = pick(CB.stage1(w, fut, S, Z))
    w["shard"] = sh1
    d = lambda a, b: round(a - b, 3) if a is not None and b is not None else None
    diff = dict(combo={b: d(res["winsor0"]["combo"][b], res["winsor1"]["combo"][b]) for b in ("conc_k10", "full")},
                mom1w={b: d(res["winsor0"]["mom1w"][b], res["winsor1"]["mom1w"][b]) for b in ("conc_k10", "full")})
    rep = dict(schema="WINSOR_IMPACT_V1", unit="bp/day net flat6, COMBO_V1 tiles 2022-01..2025-12 (104 x 336 h)",
               code_sha16=CB.sha16(os.path.abspath(__file__)), grid_sha16=CB.sha16(X.GRID), results=res,
               diff_winsor0_minus_winsor1=diff,
               frozen_h2="没做：gbm_xs_h1 只存了汇总 JSON，没有 DEV 仓位或打分；重建要跑 alpha311 环境里的 LightGBM，红线不碰",
               notes=__doc__, dev_read=False, oos_read=False, seconds=round(time.time() - t0, 1),
               written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print("winsor1", res["winsor1"]["combo"], res["winsor1"]["mom1w"], flush=True)
    print("winsor0", res["winsor0"]["combo"], res["winsor0"]["mom1w"], flush=True)
    print("wrote %s %.0fs" % (OUT, time.time() - t0), flush=True)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
