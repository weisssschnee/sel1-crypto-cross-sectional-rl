"""Stage-1 review table: OLD ruler vs NEW ruler, x {flat6, depth, maker2}, x HOLD.

All three rulers score the SAME lag-2 windows of the SAME shards, so the only
difference between columns is the ruler:
  old       policy sees real eligibility (a missing 1-min bar forces a retarget),
            raw forward returns                      -> the pre-stage-1 behaviour
  rule0     policy eligible=all-True (freeze through gaps), raw returns
  new       rule0 + 1/99 cross-sectional winsorisation  -> m1_ruler.eval_formula

Rows: the N=1 handwritten baseline at every HOLD, and the M1_17COL top-N at
their own HOLD. Units: net bp/day. Read-only; writes one JSON.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'D:/riskpg_stateful_v5s_governor_nextgen_beem/src')
sys.path.insert(0, r'D:/riskpg_stateful_v5s_governor_nextgen_beem/runtime')

import m1_common as C
import m1_ruler as R
import m2_trainer as T
import dev_score as D
from oos_guard import guard_shard_dir

_ALL = {'2025-07': (r'D:/ChengboRemote/data/m1_shards_v2', 'TRAIN')}
_ALL.update({f'2026-0{k}': (r'D:/ChengboRemote/data/m1_shards_dev', 'DEV') for k in range(1, 7)})
MONTHS = {m: _ALL[m] for m in os.environ.get('S1_TABLE_MONTHS', '2025-07,2026-03').split(',')}
DEPTH_DIR = r'D:/ChengboRemote/runtime/p1'
RUN = os.environ.get('S1_TABLE_RUN', 'M1_17COL_20261007')
TOPN = int(os.environ.get('S1_TABLE_TOPN', '15'))
COSTS = ('flat6', 'depth', 'maker2')
OUT = os.environ.get('S1_TABLE_OUT', r'D:/ChengboRemote/runtime/m1_stage1/s1_table_result.json')


def old_ruler(vm, body, hold, slices, shard, starts, policy_mod, invalid_cls, nav):
    """bp/day per cost under the pre-stage-1 behaviour, same windows as the new ruler."""
    E = R.EPISODE_MINUTES
    specs = {c: R.cost_specs(nav)[c] for c in COSTS}
    acc = {c: [] for c in COSTS}
    for s0, sl in zip(starts, slices):
        sigT = R.window_signal(vm, list(body), sl, E)
        el = shard['eligible'][:, s0:s0 + E].T
        desired = C.rank_weights(sigT, el)
        held = np.asarray(policy_mod.apply_stateful_action_policy(
            desired_weights=desired, eligible=el,
            contract=policy_mod.StatefulActionContract(minimum_hold_hours=int(hold)))[0]).reshape(desired.shape)
        res, _ = R.settle_multi(held, shard['fwd'][:, s0:s0 + E].T, shard['fi'][:, s0:s0 + E].T,
                                shard['fr'][:, s0:s0 + E].T, specs,
                                shard['bid'][s0:s0 + E], shard['ask'][s0:s0 + E])
        for c in COSTS:
            acc[c].append(res[c]['log_nav'])
    return {c: R.bp_per_day(float(np.mean(acc[c]))) for c in COSTS}


def score(vm, body, hold, mo, policy_mod, invalid_cls, nav):
    old = old_ruler(vm, body, hold, mo['slices'], mo['shard'], mo['starts'], policy_mod, invalid_cls, nav)
    res = R.eval_formula(vm, list(body), [hold], mo['slices'], mo['shard'], mo['starts'], policy_mod,
                         invalid_cls, costs=COSTS, variants=('win', 'raw'), nav_usd=nav, ic=False)
    out = dict(old=old)
    for tag, v in (('rule0', 'raw'), ('new', 'win')):
        out[tag] = {c: R.bp_per_day(float(np.mean(res['per_hold'][hold][v][c]))) for c in COSTS}
    p = res['parts'][hold]['win']
    out['turn_day_new'] = float(np.mean([x['turnover'] for x in p])) / (R.EPISODE_MINUTES / 1440.0)
    return out


def main():
    import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vm_mod
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import XsTsCsStackVM, CandidateLocalInvalid
    from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod
    vm_mod.BLOCK_ELEMS = 1 << 27
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vm = XsTsCsStackVM(len(C.TERMINALS), windows=C.WINDOWS)
    cands = D.load_candidates(os.path.join(D.ART, RUN), TOPN)
    forms = [('BASELINE', D.baseline_body(vm), h, 'CS_RANK(NEG(close_to_open_bps))') for h in C.HOLDS]
    forms += [(f'R{i+1}', b, h, e[:60]) for i, (_, b, h, e) in enumerate(cands)]
    out = dict(holds=list(C.HOLDS), episode=R.EPISODE_MINUTES, costs=list(COSTS), months={})
    for month, (sdir, role) in MONTHS.items():
        guard_shard_dir(sdir, role, expect_months=None)
        shard = R.prepare_shard(T.load_shard(month, sdir), R.WINSOR_PCT, depth_dir=DEPTH_DIR)
        starts = R.tile_starts(shard['n_min'])
        mo = dict(shard=shard, starts=starts, slices=T.gpu_episode_slices(shard, starts, dev))
        rows = []
        for name, body, hold, expr in forms:
            try:
                s = score(vm, body, hold, mo, policy_mod, CandidateLocalInvalid, R.NAV_USD)
            except CandidateLocalInvalid as exc:
                rows.append(dict(name=name, hold=hold, expr=expr, invalid=str(exc)))
                continue
            rows.append(dict(name=name, hold=hold, expr=expr, **s))
            print(f"{month} {name:<8} h={hold:<5} flat6 old {s['old']['flat6']:>+9.1f} rule0 {s['rule0']['flat6']:>+8.1f} "
                  f"new {s['new']['flat6']:>+8.1f} | depth new {s['new']['depth']:>+8.1f} | maker new {s['new']['maker2']:>+8.1f}",
                  flush=True)
        out['months'][month] = dict(role=role, windows=len(starts), rows=rows)
        del mo
        torch.cuda.empty_cache()
    json.dump(out, open(OUT, 'w'), indent=1, default=str)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
