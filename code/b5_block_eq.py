"""Check VM block size gives bit-identical signals (CPU): 4 formulas x 2 tiles. One line."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
torch.set_num_threads(4)
import b5_fixed as F
TR, M, D, W, R = F._imports()
import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vmm
man, arr = F.load_cache()
st = [man["starts"][0], man["starts"][30]]
sl = F.tile_slices(arr, st)
univ = np.array(arr["univ"])
vm = F.make_vm()
bodies = [[3, 38, 26, 41], list(D.plant_body(vm)), [0, 37, 21, 41], [8, 40, 25, 41],
          [3, 38, 27, 3, 37, 32, 31, 1, 0, 22, 33, 40, 28, 41]]
bad = 0
for b in bodies:
    try:
        for s0, x in zip(st, sl):
            vmm.BLOCK_ELEMS = 1 << 25
            a = M.window_signal(vm, b, x, s0, univ)
            vmm.BLOCK_ELEMS = 1 << 16
            c = M.window_signal(vm, b, x, s0, univ)
            same = np.array_equal(np.isnan(a), np.isnan(c)) and np.array_equal(a[~np.isnan(a)], c[~np.isnan(c)])
            bad += not same
    except Exception as e:
        print("err", vm.describe(b, list(M.TERMINALS)), str(e)[:60])
print("BLOCKEQ bad %d of %d" % (bad, 2 * len(bodies)))
