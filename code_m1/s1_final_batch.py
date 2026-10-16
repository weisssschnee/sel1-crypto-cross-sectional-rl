"""Sample the FINAL policy's batch for the paired-batch verdict (work order 2026-10-08, rule 1).

Loads a trainer checkpoint, rebuilds the exact model the trainer used, and draws one batch of formulas with a
FIXED seed that is independent of the training seed. The INITIAL batch is not drawn here: it is the update-0
rows the trainer already recorded in FORMULAS.jsonl.

Sampling mirrors the trainer: same sampler (sample_batch_with_hold), same legality table, model left in its
default (train) mode because the trainer never calls .eval(), so dropout is active in both batches. Under
no_grad; the seed fixes torch's global RNG (dropout) and the multinomial generator.

env: S1_RUN (run dir name under artifacts/m2_v1), S1_CKPT (checkpoint path; default newest in the run dir),
     S1_SEED (default 20261108), S1_N (default 256), S1_OUT_DIR (default the run dir)
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem"
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "runtime"))
LEGACY_ROOT = r"D:\v4r_groot\Chengbo\workspaces\alphagpt-rl-dl-base"
for p in (LEGACY_ROOT, os.path.join(LEGACY_ROOT, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import m1_common as C
import m1_sample
from m1_sample import sample_batch_with_hold, strip_hold, hold_of
import m2_trainer as T

RUN = os.environ["S1_RUN"]
SEED = int(os.environ.get("S1_SEED", "20261108"))
N = int(os.environ.get("S1_N", "256"))
RUN_DIR = os.path.join(ROOT, "artifacts", "m2_v1", RUN)
OUT_DIR = os.environ.get("S1_OUT_DIR", RUN_DIR)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck_path = os.environ.get("S1_CKPT")
    if not ck_path:
        cks = sorted(glob.glob(os.path.join(RUN_DIR, "checkpoint_*.pt")))
        if not cks:
            raise SystemExit(f"no checkpoint in {RUN_DIR}")
        ck_path = cks[-1]
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)

    import model_core.alphagpt as alphagpt_mod
    from model_core.config import ModelConfig
    from model_core.vocab import FormulaVocab
    from train_basin_riskpg_v1 import make_basin_model, apply_output_bias
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import XsTsCsStackVM, OPERATOR_NAMES
    from okx_alpha_discovery.adapters.xs_tscs_sampler_v2 import build_legality_table
    from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod
    ModelConfig.DEVICE = device

    mods = dict(m1_ruler=T._sha(T.R.__file__), m1_common=T._sha(C.__file__),
                m1_sample=T._sha(m1_sample.__file__), stateful_action_policy=T._sha(policy_mod.__file__))
    ck_mods = {k: v for k, v in ck["modules"].items() if k in mods}
    if ck_mods != mods:
        raise SystemExit(f"code differs from the checkpointed run: {ck_mods} vs {mods}")

    vm = XsTsCsStackVM(len(C.TERMINALS), windows=C.WINDOWS)
    op_names = list(OPERATOR_NAMES) + [f"W{w}" for w in C.WINDOWS] + ["END"] + m1_sample.HOLD_NAMES
    model, _ = make_basin_model(
        alphagpt_mod=alphagpt_mod, ModelConfig=ModelConfig, FormulaVocab=FormulaVocab,
        feature_names=list(C.TERMINALS), operator_token_names=op_names,
        max_len=C.MAX_PROGRAM_LEN, device=device)
    apply_output_bias(model, len(C.TERMINALS), len(OPERATOR_NAMES), len(C.WINDOWS), T.OP_BIAS, T.END_BIAS)
    model.load_state_dict(ck["model"])
    table = build_legality_table(vm, max_len=C.MAX_PROGRAM_LEN, available_windows=C.WINDOWS).to(device)

    torch.manual_seed(SEED)
    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    with torch.no_grad():
        batch = sample_batch_with_hold(model, vm, table, N, device=device, generator=gen)

    rows = []
    for toks in batch.formulas:
        toks = list(toks)
        body = list(strip_hold(vm, toks))
        rows.append(dict(update=-1, batch="FINAL", tokens=toks, body=body, hold_min=hold_of(vm, toks),
                         expr=vm.describe(body, list(C.TERMINALS))))
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "FINAL_BATCH.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    holds = [r["hold_min"] for r in rows]
    receipt = dict(schema="S1_FINAL_BATCH_V1", run=RUN, checkpoint=ck_path, checkpoint_sha256=sha256(ck_path),
                   checkpoint_update=int(ck["update"]), seed=SEED, n=len(rows),
                   n_distinct_tokens=len({tuple(r["tokens"]) for r in rows}),
                   hold_hist={h: holds.count(h) for h in m1_sample.HOLDS}, model_mode="train (as the trainer; dropout on)",
                   modules=mods, written=time.strftime("%Y-%m-%dT%H:%M:%S%z"), oos_read=False)
    json.dump(receipt, open(os.path.join(OUT_DIR, "FINAL_BATCH_RECEIPT.json"), "w"), indent=1)
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
