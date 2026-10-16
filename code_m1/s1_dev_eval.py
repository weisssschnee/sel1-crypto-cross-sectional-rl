"""Post-run DEV evaluation of a stage-1 run. DEV 2026-01..06 only; evaluation, never selection input to the trainer.

Same scorer as the trainer (m1_ruler.eval_formula) over back-to-back DEV windows. Every formula runs at its
OWN hold under four cost models in one VM pass: flat6 (the training cost), cost0 (gross), maker2 and depth
(optimistic bounds, final comparison table only). Per-month book diagnostics ride along: net exposure,
gross exposure, beta to the equal-weight market, turnover/day.

Candidates = top-K distinct (body, hold) of the run's FORMULAS.jsonl by the trainer's HOLDOUT reward; the
N=1 handwritten baseline is scored at every hold through identical code. Behaviour dedup and the verdict
rules live in s1_dev_report.py (pure python, no GPU).

env: S1_RUN (run dir name under artifacts/m2_v1), S1_TOPK (default 60), S1_MONTHS (default 2026-01..06),
     S1_OUT (result json)
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

DEV_DIR = r'D:/ChengboRemote/data/m1_shards_dev'
DEPTH_DIR = r'D:/ChengboRemote/runtime/p1'
RUN = os.environ['S1_RUN']
TOPK = int(os.environ.get('S1_TOPK', '60'))
MONTHS = os.environ.get('S1_MONTHS', ','.join(f'2026-0{k}' for k in range(1, 7))).split(',')
OUT = os.environ.get('S1_OUT', rf'D:/ChengboRemote/runtime/m1_stage1/s1_dev_eval_{RUN}.json')
COSTS = ('flat6', 'cost0', 'maker2', 'depth')


def month_stats(res, hold):
    days = R.EPISODE_MINUTES / 1440.0
    parts = res['parts'][hold]['win']
    betas = [p['beta_mkt'] for p in parts if np.isfinite(p['beta_mkt'])]
    o = {c: R.bp_per_day(float(np.mean(res['per_hold'][hold]['win'][c]))) for c in COSTS}
    o.update(turn_day=float(np.mean([p['turnover'] for p in parts])) / days,
             gross_bp_day=1e4 * float(np.mean([p['gross'] for p in parts])) / days,
             net_exposure=float(np.mean([p['net_exposure_mean'] for p in parts])),
             gross_exposure=float(np.mean([p['gross_exposure_mean'] for p in parts])),
             beta_mkt=float(np.mean(betas)) if betas else None,
             windows=len(parts))
    return o


def main():
    import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vm_mod
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import XsTsCsStackVM, CandidateLocalInvalid
    from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod
    vm_mod.BLOCK_ELEMS = 1 << 25
    vm_mod.EMA_KERNEL_CACHE_MAX = 1
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vm = XsTsCsStackVM(len(C.TERMINALS), windows=C.WINDOWS)
    run_dir = os.path.join(D.ART, RUN)
    cands = D.load_candidates(run_dir, TOPK)
    forms = [dict(name='BASELINE', kind='BASELINE', body=list(D.baseline_body(vm)), hold=h,
                  expr='CS_RANK(NEG(close_to_open_bps))', train_reward=None) for h in C.HOLDS]
    forms += [dict(name=f'C{i + 1:02d}', kind='CAND', body=list(b), hold=int(h), expr=e, train_reward=float(r))
              for i, (r, b, h, e) in enumerate(cands)]
    print(f'{RUN}: {len(cands)} candidates + {len(C.HOLDS)} baseline holds, months {MONTHS}', flush=True)
    for f in forms:
        f['per_month'] = {}
        f['invalid'] = None
    windows = {}
    for month in MONTHS:
        guard_shard_dir(DEV_DIR, 'DEV', expect_months=None)
        shard = R.prepare_shard(T.load_shard(month, DEV_DIR), R.WINSOR_PCT, depth_dir=DEPTH_DIR)
        starts = R.tile_starts(shard['n_min'])
        slices = T.gpu_episode_slices(shard, starts, dev)
        windows[month] = len(starts)
        for k, f in enumerate(forms):
            if f['invalid']:
                continue
            try:
                res = R.eval_formula(vm, f['body'], [f['hold']], slices, shard, starts, policy_mod,
                                     CandidateLocalInvalid, costs=COSTS, variants=('win',), ic=False)
            except CandidateLocalInvalid as exc:
                f['invalid'] = f'{month}: {exc}'
                continue
            f['per_month'][month] = month_stats(res, f['hold'])
        print(f'{month}: {len(starts)} windows scored for {len(forms)} formulas', flush=True)
        del slices, shard
        torch.cuda.empty_cache()
    out = dict(run=RUN, topk=TOPK, months=MONTHS, windows=windows, holds=list(C.HOLDS),
               episode=R.EPISODE_MINUTES, costs=list(COSTS), rows=forms)
    json.dump(out, open(OUT, 'w'), indent=1, default=str)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
