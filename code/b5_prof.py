"""Profile one B5 update's pieces in the trainer's own configuration (GPU VM 1<<25, default
threads): VM signal per formula x 16 tiles, ruler score_window, IC, diagnostics, corr, plus the
masked sampler for 128. Formulas = the last 40 valid rows of B5_DBG_REAL. Real world. One line."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import b5_fixed as F
TR, M, D, W, R = F._imports()
import m2h_b5_trainer_dbg as TD
import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vmm
from okx_alpha_discovery.evaluation import stateful_action_policy as pol
vmm.BLOCK_ELEMS = 1 << 25
vmm.EMA_KERNEL_CACHE_MAX = 1
dev = torch.device("cuda")
man, arr = F.load_cache()
sh = F.make_shard("real", arr)
univ = np.array(arr["univ"])
starts = list(man["starts"][::3][:16])
vm = vmm.XsTsCsStackVM(len(M.TERMINALS), windows=M.WINDOWS)
slices = M.tile_slices(np.asarray(arr["feats_pad"]), starts, dev)
rows = []
for l in open(os.path.join(F.A_ROOT, "B5_DBG_REAL", "FORMULAS.jsonl"), encoding="utf-8"):
    r = json.loads(l)
    if r.get("invalid_reason") is None:
        rows.append(r)
rows = rows[-40:]
tv = tr = ti = td = 0.0
E = M.EPISODE
for r in rows:
    body = list(TD.strip_hold(vm, list(r["tokens"])))
    for s0, sl in zip(starts, slices):
        torch.cuda.synchronize(); t0 = time.time()
        sig = M.window_signal(vm, body, sl, s0, univ)
        torch.cuda.synchronize(); tv += time.time() - t0
        t0 = time.time()
        sc = R.score_window(sig, sh, s0, E, [168], pol, costs=("flat6", "cost0"), variants=("win",),
                            band=R.BAND, nav_usd=R.NAV_USD)
        tr += time.time() - t0
        t0 = time.time()
        D.ic_tile(sig, sh, s0)
        ti += time.time() - t0
        t0 = time.time()
        M.diagnostics([sc[168]["win"][1]])
        td += time.time() - t0
n = len(rows) * 16
# sampler
from train_basin_riskpg_v1 import make_basin_model, apply_output_bias
import model_core.alphagpt as alphagpt_mod
from model_core.config import ModelConfig
from model_core.vocab import FormulaVocab
from okx_alpha_discovery.adapters.xs_tscs_sampler_v2 import build_legality_table
ModelConfig.DEVICE = dev
names = (list(vmm.OPERATOR_NAMES) + [f"W{w}" for w in M.WINDOWS] + ["END"] + M.HOLD_NAMES + list(M.BASIN_TOKEN_NAMES))
model, _ = make_basin_model(alphagpt_mod=alphagpt_mod, ModelConfig=ModelConfig, FormulaVocab=FormulaVocab,
                            feature_names=list(M.TERMINALS), operator_token_names=names, max_len=M.MAX_PROGRAM_LEN, device=dev)
apply_output_bias(model, len(M.TERMINALS), len(vmm.OPERATOR_NAMES), len(M.WINDOWS), TD.OP_BIAS, TD.END_BIAS)
tabs = [build_legality_table(vm, max_len=M.MAX_PROGRAM_LEN, available_terminals=M.basin_terms(b),
                             available_windows=M.WINDOWS).to(dev) for b in M.BASIN_NAMES]
t0 = time.time()
TD.sample_batch_with_hold_basins(model, vm, tabs, M.quota(128), device=dev)
ts_ = time.time() - t0
print("PROF per formula-tile ms: vm %.1f ruler %.1f ic %.1f diag %.2f | per update (118x16): vm %.0fs ruler %.0fs ic %.0fs | sampler %.1fs | torch threads %d"
      % (1e3 * tv / n, 1e3 * tr / n, 1e3 * ti / n, 1e3 * td / n, tv / n * 1888, tr / n * 1888, ti / n * 1888, ts_,
         torch.get_num_threads()))
