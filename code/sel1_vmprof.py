"""XS_SEL1 VM cost profile (read-only diagnostics; GPU, V4 TRAIN features only, no returns). New file.
Times every operator call of xs_tscs_vm_v2 (wrapped, synchronised) while the bodies of a finished run's FORMULAS
would run on the year chunks; bodies are taken from CR1_PC_P8 HOF (tokens) since FORMULAS keeps only expressions.
Also counts the coins that are ever in TOP100 during 2021-2025.  -> SEL1_VM_PROFILE.json, one summary line.
"""
import collections
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
import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as V

BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 0          # V.BLOCK_ELEMS for this run (0 = whole panel)
FAST = len(sys.argv) > 2 and sys.argv[2] == "fast"            # sel1_fastts operators installed before timing
OUT = os.path.join(S.OUT_ROOT, "SEL1_VM_PROFILE%s%s.json" % ("_b%d" % BLOCK if BLOCK else "", "_fast" if FAST else ""))
STAT = collections.defaultdict(lambda: [0, 0.0])


def wrap(table, windowed):
    for name, fn in list(table.items()):
        def w(*a, _fn=fn, _name=name):
            torch.cuda.synchronize()
            t = time.time()
            r = _fn(*a)
            torch.cuda.synchronize()
            key = "%s|w%d" % (_name, a[-1]) if windowed else _name
            STAT[key][0] += 1
            STAT[key][1] += time.time() - t
            return r
        table[name] = w


def main():
    warnings.simplefilter("ignore", RuntimeWarning)
    vm = Q.make_vm()
    ctx = S.build("real", torch.device("cuda"), vm, log=lambda m: None)
    i0 = int(ctx["ts"].searchsorted(S.TRAIN_FIRST))
    ever = int(ctx["univ"][:, i0:].any(axis=1).sum())
    if FAST:
        import sel1_fastts
        import m2h_common as M
        sel1_fastts.install()
        tw = time.time()
        x = torch.ones((2, 8), dtype=torch.float32, device="cuda")
        for name in sel1_fastts.REPLACED:          # NVRTC compiles each (kernel, window) once; keep it out of the timing
            for w in M.WINDOWS:
                sel1_fastts.FAST[name](*((x, x) if name == "TS_CORR" else (x,)), w)
        torch.cuda.synchronize()
        print("fastts compile %.1fs" % (time.time() - tw), flush=True)
    wrap(V._WINDOWED_IMPL, True)
    wrap(V._PLAIN_IMPL, False)
    hof = json.load(open(os.path.join(S.A_ROOT, "CR1_PC_P8", "HOF.json"), encoding="utf-8"))["entries"]
    bodies = [tuple(e["body"]) for e in hof[:60]]
    V.BLOCK_ELEMS = BLOCK
    t0 = time.time()
    per, sig = [], []
    t_raw = 0.0
    for b in bodies:
        torch.cuda.synchronize()
        t = time.time()
        try:
            s_ = S.raw_signal(ctx, vm, b)
            t_raw += time.time() - t
            sig.append(hashlib.sha256(np.ascontiguousarray(s_).tobytes()).hexdigest())
        except Exception as e:
            sig.append("ERR " + type(e).__name__ + " " + str(e)[:60])
        torch.cuda.synchronize()
        per.append(time.time() - t)
    tot = sum(v[1] for v in STAT.values())
    top = sorted(STAT.items(), key=lambda kv: -kv[1][1])[:25]
    rep = dict(block_elems=BLOCK, peak_vram_gib=torch.cuda.max_memory_allocated() / 2 ** 30, signal_sha=sig,
               coins=len(ctx["syms"]), coins_ever_top100_2021_25=ever, bodies=len(bodies),
               body_s=dict(mean=float(np.mean(per)), p50=float(np.median(per)), p90=float(np.quantile(per, 0.9))),
               op_total_s=tot, raw_signal_s=t_raw, wall_s=time.time() - t0, fast=FAST,
               top_ops=[dict(op=k, calls=v[0], s=round(v[1], 3), share=round(v[1] / tot, 4)) for k, v in top])
    json.dump(rep, open(OUT, "w"), indent=1)
    print("fast %s ops %.1fs raw %.1fs | block %d peak %.1fGiB | coins %d ever-top100 %d | body mean" % (FAST, tot, t_raw, BLOCK, rep["peak_vram_gib"], rep["coins"], ever) + " %.2fs p50 %.2fs | top %s" % (rep["body_s"]["mean"],
          rep["body_s"]["p50"], ", ".join("%s %.0f%%" % (x["op"], 100 * x["share"]) for x in rep["top_ops"][:8])), flush=True)


if __name__ == "__main__":
    main()
