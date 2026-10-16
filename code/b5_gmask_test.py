"""Quick CPU check of m2h_b5_trainer_dbg's grammar-masked sampler (no training, no data):
init policy (seed 20261009), 512 draws per family. Prints one line: per family legal rate under
m2h_b5_grammar.check (must be 1.0), mean length, and whether the plant body's path is unmasked."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch

import m2h_b5_trainer_dbg as T
import m2h_common as M
import m2h_dense_common as D
import m2h_b5_grammar as G
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import XsTsCsStackVM, OPERATOR_NAMES
from okx_alpha_discovery.adapters.xs_tscs_sampler_v2 import build_legality_table
from train_basin_riskpg_v1 import make_basin_model, apply_output_bias
import model_core.alphagpt as alphagpt_mod
from model_core.config import ModelConfig
from model_core.vocab import FormulaVocab

dev = torch.device("cpu")
ModelConfig.DEVICE = dev
torch.manual_seed(20261009)
vm = XsTsCsStackVM(len(M.TERMINALS), windows=M.WINDOWS)
names = (list(OPERATOR_NAMES) + [f"W{w}" for w in M.WINDOWS] + ["END"] + M.HOLD_NAMES + list(M.BASIN_TOKEN_NAMES))
model, _ = make_basin_model(alphagpt_mod=alphagpt_mod, ModelConfig=ModelConfig, FormulaVocab=FormulaVocab,
                            feature_names=list(M.TERMINALS), operator_token_names=names,
                            max_len=M.MAX_PROGRAM_LEN, device=dev)
apply_output_bias(model, len(M.TERMINALS), len(OPERATOR_NAMES), len(M.WINDOWS), T.OP_BIAS, T.END_BIAS)
out = []
for fam in M.BASIN_NAMES:
    tab = build_legality_table(vm, max_len=M.MAX_PROGRAM_LEN, available_terminals=M.basin_terms(fam),
                               available_windows=M.WINDOWS).to(dev)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(7)
    with torch.no_grad():
        b = T.sample_batch_with_hold_basins(model, vm, [tab], [512], device=dev, generator=gen, basins=(fam,))
    bodies = [list(T.strip_hold(vm, list(t))) for t in b.formulas]
    legal = np.mean([G.check(vm, x) is None for x in bodies])
    holds = set(T.hold_of(vm, list(t)) for t in b.formulas)
    ln = np.mean([len(x) - 1 for x in bodies])
    distinct = len({tuple(x) for x in bodies})
    out.append("%s legal %.3f len %.1f distinct %d holds %s" % (fam, legal, ln, distinct, sorted(holds)))
# the plant body's path through the T1 mask
tab = build_legality_table(vm, max_len=M.MAX_PROGRAM_LEN, available_terminals=M.basin_terms("T1"),
                           available_windows=M.WINDOWS)
mf = tab.min_finish.tolist()
ent, depth, win, ok = [], 0, 0, True
import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vmm
ar = dict(vmm.OPERATOR_ARITIES)
for step, t in enumerate(D.plant_body(vm)):
    rem = M.MAX_PROGRAM_LEN - step
    fin = torch.isfinite(tab.mask[depth, win, rem]).numpy()
    gm = T._g_mask_row(ent, fin, rem, vm, mf)
    if not (fin[t] and np.isfinite(gm[t])):
        ok = False
        break
    if t == vm.end_token:
        break
    ent, _ = T._g_step(ent, t, vm)
    if vm.is_terminal(t):
        depth, win = depth + 1, 0
    elif vm.is_window(t):
        win = 1
    else:
        depth, win = depth - ar[vm.operator_name(t)] + 1, 0
print("GMASK " + " | ".join(out) + " | plant path ok %s" % ok)
