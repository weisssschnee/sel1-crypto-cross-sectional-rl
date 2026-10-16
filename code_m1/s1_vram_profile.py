"""Per-operator peak VRAM / wall-clock on ONE 10140+1440-column window (read-only).

Answers where the batch-256 probe's 16.7 GiB floor comes from. Each windowed
operator is run over a terminal at every window; peak is measured from a reset
after the previous op, with the EMA kernel cache state printed alongside."""
from __future__ import annotations

import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r'D:/riskpg_stateful_v5s_governor_nextgen_beem/src')
sys.path.insert(0, r'D:/riskpg_stateful_v5s_governor_nextgen_beem/runtime')

import m1_common as C
import m1_ruler as R
import m2_trainer as T
import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vm_mod
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import (
    OPERATOR_ARITIES, OPERATOR_NAMES, WINDOWED_OPERATORS, XsTsCsStackVM)

GIB = 2 ** 30
dev = torch.device('cuda')
vm_mod.BLOCK_ELEMS = int(os.environ.get('S1_BLOCK', 1 << 27))
vm = XsTsCsStackVM(len(C.TERMINALS), windows=C.WINDOWS)
shard = R.prepare_shard(T.load_shard('2025-07', r'D:/ChengboRemote/data/m1_smoke_2025-07'), R.WINSOR_PCT)
starts = R.tile_starts(shard['n_min'])[:1]
sl = T.gpu_episode_slices(shard, starts, dev)[0]
print(f'slice {tuple(sl.shape)} {sl.dtype}  block_elems {vm_mod.BLOCK_ELEMS}  resident {torch.cuda.memory_allocated()/GIB:.2f} GiB', flush=True)
arity = dict(OPERATOR_ARITIES)


def run(name, w_idx):
    toks = [0] if arity[name] == 1 else [0, 1]
    toks += [vm.window_base + w_idx, vm.operator_base + OPERATOR_NAMES.index(name)]
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    out = vm.execute(toks + [vm.end_token], sl)
    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() - base) / GIB
    del out
    return peak, dt


print(f'{"op":<9}{"win":>6} {"peak+GiB":>9} {"sec":>6}   ema_cache_n  resident_GiB')
for name in WINDOWED_OPERATORS:
    for w_idx, w in enumerate(C.WINDOWS):
        try:
            peak, dt = run(name, w_idx)
        except Exception as exc:
            print(f'{name:<9}{w:>6}  ERR {type(exc).__name__}: {str(exc)[:60]}', flush=True)
            continue
        print(f'{name:<9}{w:>6} {peak:>9.2f} {dt:>6.2f}   {len(vm_mod._EMA_KERNEL_CACHE):>4}        '
              f'{torch.cuda.memory_allocated()/GIB:>6.2f}', flush=True)
print('max over all ops (above resident):', end=' ')
print(f'{torch.cuda.max_memory_allocated()/GIB:.2f} GiB absolute peak')
