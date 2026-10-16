"""Time b5_fixed's per-tile pieces on CPU: VM window_signal at two block sizes, score_window full/conc."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, torch
torch.set_num_threads(1)
import b5_fixed as F
TR, M, D, W, R = F._imports()
import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vmm
from okx_alpha_discovery.evaluation import stateful_action_policy as pol
man, arr = F.load_cache()
st = man["starts"][:3]
sl = F.tile_slices(arr, st)
univ = np.array(arr["univ"])
vm = F.make_vm()
body = [3, 38, 26, 41]          # TS_ZSCORE(log_quote_volume, 336)
pb = list(D.plant_body(vm))
out = []
for be in (1 << 25, 1 << 20, 1 << 16):
    vmm.BLOCK_ELEMS = be
    t0 = time.time()
    for s0, x in zip(st, sl):
        for b in (body, pb):
            sig = M.window_signal(vm, b, x, s0, univ)
    out.append("vm be%d %.2fs/6" % (be.bit_length() - 1, time.time() - t0))
sh = F.make_shard("p32", arr)
t0 = time.time()
for s0, x in zip(st, sl):
    sig = M.window_signal(vm, body, x, s0, univ)
    R.score_window(sig, sh, s0, M.EPISODE, [168], pol, costs=M.COSTS, variants=("win",), band=R.BAND, nav_usd=R.NAV_USD)
out.append("vm+score %.2fs/3" % (time.time() - t0))
print("FTIME " + " | ".join(out))
