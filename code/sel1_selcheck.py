"""XS_SEL1 补充 10-15 step 2: zero-diff check and timing of sel1_common.vm_select_gpu against vm_select (GPU VM,
V4 TRAIN features only, no returns). New file, diagnostics only. Real and null worlds, K in {5, 10, 20}, the 60
CR1_PC_P8 HOF bodies, sel1_fastts installed in both paths (as the trainer runs with --fastts): every (hi, lo, n)
array of every A month must be equal, Invalid outcomes equal. Seconds per body for both paths.
-> SEL1_SELCHECK.json, one line per world; rc 0 only if everything is identical.
"""
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import torch
import sel1_common as S
import cr1_common as Q
import m2h_common as M
import sel1_fastts as F
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid

OUT = os.path.join(S.OUT_ROOT, "SEL1_SELCHECK.json")


def run(fn, ctx, vm, b):
    torch.cuda.synchronize()
    t = time.time()
    try:
        r = fn(ctx, vm, b)
    except (S.Invalid, CandidateLocalInvalid, M.ZeroCoverage) as e:
        r = "INVALID:" + str(e)
    torch.cuda.synchronize()
    return r, time.time() - t


def equal(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return len(a) == len(b) and all(all(np.array_equal(x, y) for x, y in zip(p, q)) for p, q in zip(a, b))


def main():
    warnings.simplefilter("ignore", RuntimeWarning)
    vm = Q.make_vm()
    F.install()
    hof = json.load(open(os.path.join(S.A_ROOT, "CR1_PC_P8", "HOF.json"), encoding="utf-8"))["entries"]
    bodies = [tuple(e["body"]) for e in hof[:60]]
    rep, ok = {}, True
    for mode in ("real", "null"):
        ctx = S.build(mode, torch.device("cuda"), vm, log=lambda m: None)
        run(S.vm_select, ctx, vm, bodies[0])                    # compile / warm both paths
        run(S.vm_select_gpu, ctx, vm, bodies[0])
        res = {}
        for k in (5, 10, 20):
            S.K = k
            diff, ta, tb, inv = [], [], [], 0
            for i, b in enumerate(bodies):
                a, sa = run(S.vm_select, ctx, vm, b)
                g, sb = run(S.vm_select_gpu, ctx, vm, b)
                ta.append(sa)
                tb.append(sb)
                inv += isinstance(a, str)
                if not equal(a, g):
                    diff.append(i)
            res[k] = dict(diff=diff, invalid=inv, cpu_s=float(np.mean(ta)), gpu_s=float(np.mean(tb)),
                          cpu_p50=float(np.median(ta)), gpu_p50=float(np.median(tb)))
            ok = ok and not diff
            print("%s K%-2d diff %d/%d invalid %d | s/body mean %.3f -> %.3f p50 %.3f -> %.3f" % (
                mode, k, len(diff), len(bodies), inv, res[k]["cpu_s"], res[k]["gpu_s"], res[k]["cpu_p50"],
                res[k]["gpu_p50"]), flush=True)
        rep[mode] = res
        del ctx
        torch.cuda.empty_cache()
    S.K = 5
    json.dump(dict(schema="SEL1_SELCHECK_V1", worlds=rep, bit_identical=ok, bodies=len(bodies),
                   common_sha16=S.sha16f(S.__file__), fastts_sha16=S.sha16f(F.__file__)),
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("selcheck identical %s" % ok, flush=True)
    sys.exit(0 if ok else 7)


if __name__ == "__main__":
    main()
