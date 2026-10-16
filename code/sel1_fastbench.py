"""XS_SEL1 补充 A: per-operator timing, original vs sel1_fastts, on year chunk 0 of the V4 TRAIN features (GPU, no
returns). New file, diagnostics only. -> one line per (op, window): ms original, ms fused.
"""
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import sel1_common as S
import cr1_common as Q
import sel1_fastts as F


def ms(fn, *a):
    fn(*a)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(3):
        fn(*a)
    torch.cuda.synchronize()
    return 1000 * (time.time() - t) / 3


def main():
    warnings.simplefilter("ignore", RuntimeWarning)
    vm = Q.make_vm()
    ctx = S.build("real", torch.device("cuda"), vm, log=lambda m: None)
    in0, o0, o1 = ctx["chunks"][0]
    x = ctx["feats_gpu"][:, 0, in0:o1].contiguous()
    y = ctx["feats_gpu"][:, 3, in0:o1].contiguous()
    F.install()
    print("shape %s dtype %s cuda %s block %s fused_path %s kernels %d" % (
        tuple(x.shape), x.dtype, x.is_cuda, F.V.BLOCK_ELEMS, F._ok(24, x), len(F._KERN)), flush=True)
    F.FAST["TS_MEAN"](x.float(), 24)
    torch.cuda.synchronize()
    print("after float32 call kernels %d" % len(F._KERN), flush=True)
    for name in ("TS_MEAN", "TS_RANK", "TS_SLOPE", "TS_CORR"):
        for w in (24, 168, 504):
            a = (x, y) if name == "TS_CORR" else (x,)
            print("%s w%d orig %.1fms fast %.1fms" % (name, w, ms(F.ORIG[name], *a, w), ms(F.FAST[name], *a, w)),
                  flush=True)


if __name__ == "__main__":
    main()
