"""XS_SEL1 补充 A: bit-for-bit check and timing of sel1_fastts (GPU, V4 TRAIN features only, no returns). New file.
  1. every replaced operator x every window on the 11 terminals of the first year chunk (first 96 coins; the
     reduction order per output does not depend on the coin count), plus a NaN / inf / -0.0 laced copy and, for
     TS_CORR, terminal pairs: original vs fast, compared as int32 bit patterns.
  2. the 60 CR1_PC_P8 HOF bodies through S.raw_signal, original VM (trainer config, make_vm) then fused, in this process:
     signal sha256 per body must match; seconds per body before and after.
Fused kernels must actually launch (CALLS), the original run must launch none.
-> SEL1_FASTCHECK.json, one summary line; rc 0 only if everything is bit-identical.
"""
import hashlib
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
import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as V

OUT = os.path.join(S.OUT_ROOT, "SEL1_FASTCHECK.json")
N_COINS = 96


def same(a, b):
    return a.shape == b.shape and bool(torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)))


def laced(x, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.rand(x.shape, generator=g).to(x.device)
    y = x.clone()
    y[r < 0.03] = float("nan")
    y[(r >= 0.03) & (r < 0.035)] = float("inf")
    y[(r >= 0.035) & (r < 0.04)] = float("-inf")
    y[(r >= 0.04) & (r < 0.05)] = -0.0
    y[(r >= 0.05) & (r < 0.06)] = 0.0
    return y


def op_check(feats):
    inputs = [feats[:, j, :].contiguous() for j in range(feats.shape[1])]
    inputs += [laced(inputs[0], 1), laced(inputs[3], 2), laced(inputs[6], 3)]
    pairs = [(inputs[i], inputs[j]) for i, j in ((0, 3), (3, 7), (8, 9), (11, 12), (0, 13))]
    bad, n = [], 0
    for name in F.REPLACED:
        for w in M.WINDOWS:
            cases = pairs if name == "TS_CORR" else [(x,) for x in inputs]
            for k, args in enumerate(cases):
                a = F.ORIG[name](*args, w)
                b = F.FAST[name](*args, w)
                torch.cuda.synchronize()
                n += 1
                if not same(a, b):
                    d = (a != b) & ~(torch.isnan(a) & torch.isnan(b))
                    bad.append(dict(op=name, w=w, case=k, n_diff=int(d.sum()),
                                    max_abs=float((a - b).abs().nan_to_num(0)[d].max()) if bool(d.any()) else 0.0))
    return n, bad


def main():
    warnings.simplefilter("ignore", RuntimeWarning)
    vm = Q.make_vm()
    ctx = S.build("real", torch.device("cuda"), vm, log=lambda m: None)
    in0, o0, o1 = ctx["chunks"][0]
    F.install()
    t = time.time()
    x = torch.ones((2, 8), dtype=torch.float32, device="cuda")
    for name in F.REPLACED:                             # NVRTC: one compile per kernel, kept out of the timings
        F.FAST[name](*((x, x) if name == "TS_CORR" else (x,)), 24)
    torch.cuda.synchronize()
    compile_s = time.time() - t
    F.CALLS.clear()
    n_op, bad = op_check(ctx["feats_gpu"][:N_COINS, :, in0:o1])
    op_calls = dict(F.CALLS)
    print("block %d compile %.1fs | ops %d checked, %d differ | kernel launches %s" % (
        V.BLOCK_ELEMS, compile_s, n_op, len(bad), op_calls), flush=True)
    hof = json.load(open(os.path.join(S.A_ROOT, "CR1_PC_P8", "HOF.json"), encoding="utf-8"))["entries"]
    bodies = [tuple(e["body"]) for e in hof[:60]]
    runs = {}
    for tag in ("before", "after"):                     # original VM, then the fused operators, same process
        F.uninstall() if tag == "before" else F.install()
        F.CALLS.clear()
        per, sig = [], []
        for b in bodies:
            torch.cuda.synchronize()
            t = time.time()
            try:
                sig.append(hashlib.sha256(np.ascontiguousarray(S.raw_signal(ctx, vm, b)).tobytes()).hexdigest())
            except Exception as e:
                sig.append("ERR " + type(e).__name__ + " " + str(e)[:60])
            torch.cuda.synchronize()
            per.append(time.time() - t)
        runs[tag] = dict(sig=sig, per=per, calls=dict(F.CALLS), s=dict(mean=float(np.mean(per)), p50=float(np.median(per)),
                                                  p90=float(np.quantile(per, 0.9))))
    sig_bad = [i for i, (x, y) in enumerate(zip(runs["before"]["sig"], runs["after"]["sig"])) if x != y]
    used = all(op_calls.get(k, 0) > 0 for k in ("k_mom", "k_ext", "k_slope", "k_corr")) and sum(runs["after"]["calls"].values()) > 0
    ok = not bad and not sig_bad and used and not runs["before"]["calls"]
    bf, af = runs["before"]["s"], runs["after"]["s"]
    rep = dict(block_elems=V.BLOCK_ELEMS, compile_s=compile_s, op_kernel_calls=op_calls, fused_used=used,
               body_kernel_calls=dict((k, v["calls"]) for k, v in runs.items()), ops_checked=n_op, ops_differ=bad, bodies=len(bodies), bodies_differ=sig_bad,
               bodies_error=sum(s.startswith("ERR") for s in runs["after"]["sig"]),
               before_s=bf, after_s=af, per_body_s=dict((k, [round(x, 4) for x in v["per"]]) for k, v in runs.items()),
               peak_vram_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
               fastts_sha16=S.sha16f(F.__file__), bit_identical=ok)
    json.dump(rep, open(OUT, "w"), indent=1)
    print("bit_identical %s used %s | ops %d diff %d | bodies %d diff %d | s/body mean %.3f -> %.3f p50 %.3f -> %.3f"
          % (ok, used, n_op, len(bad), len(bodies), len(sig_bad), bf["mean"], af["mean"], bf["p50"], af["p50"]), flush=True)
    sys.exit(0 if ok else 7)


if __name__ == "__main__":
    main()
