"""Paired-batch DEV evaluation (work order 2026-10-08, main verdict).

INITIAL batch = the update-0 rows the trainer recorded in FORMULAS.jsonl (sampled from the untrained policy).
FINAL batch   = FINAL_BATCH.jsonl from s1_final_batch.py (final checkpoint, fixed independent seed).
Both are scored by the SAME scorer as the trainer (m1_ruler.eval_formula) on every DEV window of 2026-01..06,
each formula at its own sampled hold, flat6 and cost0 in one VM pass. Per-WINDOW arrays are stored so the report
can pair the two batches window by window. A formula that is invalid on any DEV month is dropped from its batch
(the same filter for both batches). DEV only; evaluation, never a selection input to the trainer.

env: S1_RUN, S1_FINAL (default <run dir>/FINAL_BATCH.jsonl), S1_MONTHS (default 2026-01..06),
     S1_LIMIT (smoke: first N rows of each batch), S1_OUT
"""
from __future__ import annotations

import json
import os
import sys
import time

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
RUN = os.environ['S1_RUN']
RUN_DIR = os.path.join(D.ART, RUN)
FINAL = os.environ.get('S1_FINAL', os.path.join(RUN_DIR, 'FINAL_BATCH.jsonl'))
MONTHS = os.environ.get('S1_MONTHS', ','.join(f'2026-0{k}' for k in range(1, 7))).split(',')
LIMIT = int(os.environ.get('S1_LIMIT', '0'))
OUT = os.environ.get('S1_OUT', rf'D:/ChengboRemote/runtime/m1_stage1/s1_paired_eval_{RUN}.json')
COSTS = ('flat6', 'cost0')


def key_of(body, hold):
    return f"{int(hold)}|" + ",".join(str(int(t)) for t in body)


def read_rows(path, only_update=None):
    rows = []
    for line in open(path, encoding='utf-8'):
        if not line.strip():
            continue
        r = json.loads(line)
        if only_update is not None and r.get('update') != only_update:
            continue
        rows.append(dict(body=[int(t) for t in r['body']], hold=int(r['hold_min']), expr=r.get('expr')))
    return rows


def window_metrics(res, hold):
    days = R.EPISODE_MINUTES / 1440.0
    ph = res['per_hold'][hold]['win']
    parts = res['parts'][hold]['win']
    return dict(net=[R.bp_per_day(x) for x in ph['flat6']],
                gross=[R.bp_per_day(x) for x in ph['cost0']],
                turn=[p['turnover'] / days for p in parts],
                gE=[p['gross_exposure_mean'] for p in parts],
                nE=[p['net_exposure_mean'] for p in parts],
                beta=[p['beta_mkt'] for p in parts])


def main():
    import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vm_mod
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import XsTsCsStackVM, CandidateLocalInvalid
    from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod
    vm_mod.BLOCK_ELEMS = 1 << 25
    vm_mod.EMA_KERNEL_CACHE_MAX = 1
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vm = XsTsCsStackVM(len(C.TERMINALS), windows=C.WINDOWS)

    init_rows = read_rows(os.path.join(RUN_DIR, 'FORMULAS.jsonl'), only_update=0)
    final_rows = read_rows(FINAL)
    if LIMIT:
        init_rows, final_rows = init_rows[:LIMIT], final_rows[:LIMIT]
    for r in init_rows + final_rows:
        r['key'] = key_of(r['body'], r['hold'])
    uniq = {}
    for r in init_rows + final_rows:
        uniq.setdefault(r['key'], dict(key=r['key'], body=r['body'], hold=r['hold'], expr=r['expr'],
                                       invalid=None, win={k: [] for k in
                                                          ('net', 'gross', 'turn', 'gE', 'nE', 'beta')}))
    print(f'{RUN}: initial {len(init_rows)} rows, final {len(final_rows)} rows, {len(uniq)} unique '
          f'(body, hold); months {MONTHS}', flush=True)
    t0 = time.time()
    win_month = []
    forms = list(uniq.values())
    for month in MONTHS:
        guard_shard_dir(DEV_DIR, 'DEV', expect_months=None)
        shard = R.prepare_shard(T.load_shard(month, DEV_DIR), R.WINSOR_PCT)
        starts = R.tile_starts(shard['n_min'])
        slices = T.gpu_episode_slices(shard, starts, dev)
        win_month += [month] * len(starts)
        for k, f in enumerate(forms):
            if f['invalid']:
                continue
            try:
                res = R.eval_formula(vm, f['body'], [f['hold']], slices, shard, starts, policy_mod,
                                     CandidateLocalInvalid, costs=COSTS, variants=('win',), ic=False)
            except CandidateLocalInvalid as exc:
                f['invalid'] = f'{month}: {exc}'
                continue
            for name, vals in window_metrics(res, f['hold']).items():
                f['win'][name] += [float(v) for v in vals]
            if (k + 1) % 100 == 0:
                print(f'  {month}: {k + 1}/{len(forms)} formulas, {time.time() - t0:.0f}s', flush=True)
        print(f'{month}: {len(starts)} windows x {len(forms)} formulas, elapsed {time.time() - t0:.0f}s', flush=True)
        del slices, shard
        torch.cuda.empty_cache()
    out = dict(run=RUN, months=MONTHS, win_month=win_month, episode=R.EPISODE_MINUTES, costs=list(COSTS),
               initial=[r['key'] for r in init_rows], final=[r['key'] for r in final_rows],
               formulas={f['key']: dict(hold=f['hold'], expr=f['expr'], invalid=f['invalid'], win=f['win'])
                         for f in forms},
               eval_seconds=round(time.time() - t0, 1), oos_read=False)
    json.dump(out, open(OUT, 'w'), default=str)
    print('wrote', OUT, f'({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
