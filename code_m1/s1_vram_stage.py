"""Where does the batch-256 peak come from? Runs m2_trainer.main() unchanged for 1
update and records peak VRAM per stage by wrapping three call sites:
  sample_batch_with_hold   (sampler + model forward, autograd live)
  R.eval_formula           (VM + settlement, one formula x one window set)
  novelty_weighted_risk_update (backward + optimiser step)
Read-only diagnostic; nothing in the trainer is modified."""
from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem"
LEGACY = r"D:\v4r_groot\Chengbo\workspaces\alphagpt-rl-dl-base"
for p in (os.path.join(ROOT, "src"), os.path.join(ROOT, "runtime"), LEGACY, os.path.join(LEGACY, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import m1_ruler as R
import m2_trainer as T
import experiments.rl_v2_autonomous_novelty_search_v4 as legacy

GIB = 2 ** 30
stage = {}          # name -> dict(n, max_delta, max_abs, after_alloc)
evals = []          # (delta_gib, abs_peak_gib, n_tokens)


def _wrap(name, fn):
    def inner(*a, **k):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        s = stage.setdefault(name, dict(n=0, max_delta=0.0, max_abs=0.0, before=0.0, after=0.0))
        s["n"] += 1
        s["max_delta"] = max(s["max_delta"], (peak - base) / GIB)
        s["max_abs"] = max(s["max_abs"], peak / GIB)
        s["before"], s["after"] = base / GIB, torch.cuda.memory_allocated() / GIB
        if name == "eval_formula":
            evals.append(((peak - base) / GIB, peak / GIB, len(a[1])))
        return out
    return inner


T.sample_batch_with_hold = _wrap("sample_batch_with_hold", T.sample_batch_with_hold)
R.eval_formula = _wrap("eval_formula", R.eval_formula)
legacy.novelty_weighted_risk_update = _wrap("risk_update", legacy.novelty_weighted_risk_update)

import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vm_mod  # noqa: E402
if os.environ.get("S1_EMA_MAX"):
    vm_mod.EMA_KERNEL_CACHE_MAX = int(os.environ["S1_EMA_MAX"])
print("EMA_KERNEL_CACHE_MAX =", vm_mod.EMA_KERNEL_CACHE_MAX, flush=True)

sys.argv = ["m2_trainer.py", "S1_STAGE_PROBE_20261008" + os.environ.get("S1_TAG", ""), "--shard-dir", r"D:\ChengboRemote\data\m1_smoke_2025-07",
            "--updates", "1", "--batch", "256", "--windows", "4", "--elite-recheck", "0", "--cost", "flat6",
            "--vm-block-elems", str(1 << 25)]
T.main()

print("\n=== per-stage peak (GiB) ===")
print(f'{"stage":<26}{"calls":>6}{"max +over base":>16}{"max absolute":>14}{"alloc before":>14}{"alloc after":>13}')
for k, s in stage.items():
    print(f'{k:<26}{s["n"]:>6}{s["max_delta"]:>16.2f}{s["max_abs"]:>14.2f}{s["before"]:>14.2f}{s["after"]:>13.2f}')
evals.sort(reverse=True)
print("\n=== eval_formula: 8 largest per-call peaks (+GiB over base, absolute GiB, n_tokens) ===")
for d, a, n in evals[:8]:
    print(f"  +{d:6.2f}  abs {a:6.2f}  tokens {n}")
print(f"eval calls {len(evals)}; median +{evals[len(evals)//2][0]:.2f} GiB")
print(f"global max_memory_allocated {torch.cuda.max_memory_allocated()/GIB:.2f} GiB")
