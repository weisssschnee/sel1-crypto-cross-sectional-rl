"""Time one 128-formula masked draw vs the plain sampler (init policy, CPU). One line."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import m2h_b5_trainer_dbg as T
import m2h_b5_trainer as P
import m2h_common as M
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import XsTsCsStackVM, OPERATOR_NAMES
from okx_alpha_discovery.adapters.xs_tscs_sampler_v2 import build_legality_table
from train_basin_riskpg_v1 import make_basin_model, apply_output_bias
import model_core.alphagpt as alphagpt_mod
from model_core.config import ModelConfig
from model_core.vocab import FormulaVocab
dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cpu")
ModelConfig.DEVICE = dev
torch.manual_seed(20261009)
vm = XsTsCsStackVM(len(M.TERMINALS), windows=M.WINDOWS)
names = (list(OPERATOR_NAMES) + [f"W{w}" for w in M.WINDOWS] + ["END"] + M.HOLD_NAMES + list(M.BASIN_TOKEN_NAMES))
model, _ = make_basin_model(alphagpt_mod=alphagpt_mod, ModelConfig=ModelConfig, FormulaVocab=FormulaVocab,
                            feature_names=list(M.TERMINALS), operator_token_names=names, max_len=M.MAX_PROGRAM_LEN, device=dev)
apply_output_bias(model, len(M.TERMINALS), len(OPERATOR_NAMES), len(M.WINDOWS), T.OP_BIAS, T.END_BIAS)
tabs = [build_legality_table(vm, max_len=M.MAX_PROGRAM_LEN, available_terminals=M.basin_terms(b),
                             available_windows=M.WINDOWS).to(dev) for b in M.BASIN_NAMES]
cnt = M.quota(128)
out = []
for name, mod in (("plain", P), ("gmask", T)):
    t0 = time.time()
    b = mod.sample_batch_with_hold_basins(model, vm, tabs, cnt, device=dev)
    out.append("%s %.1fs maxlen %d" % (name, time.time() - t0, b.context["tokens"].shape[1]))
print("GTIME " + " | ".join(out))
