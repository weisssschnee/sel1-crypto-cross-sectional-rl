"""XS_SEL1 补充 10-15 step 2, second check: SEL1_SPEED_V4 (--gpu-select) matches V0 on every FORMULAS row but
pg_loss of update 5 differs in the last float32 bit. Here, on the trainer's own candidates (every body in the hall
of SEL1_SPEED_V4 checkpoint_0005, null world, K 5, sel1_fastts installed as in the run): vm_select vs vm_select_gpu
selections, and S.bpd of the A-half book from each, compared exactly; plus the max |weight difference| of
checkpoint_0005 V0 vs V3 vs V4. New file, diagnostics only, no returns beyond the trainer's A-half reward.
-> SEL1_SELCHECK2.json, one line.
"""
import json
import os
import sys
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

OUT = os.path.join(S.OUT_ROOT, "SEL1_SELCHECK2.json")


def ck(run):
    return torch.load(os.path.join(S.OUT_ROOT, run, "checkpoint_0005.pt"), map_location="cpu", weights_only=False)


def wdiff(a, b):
    return max(float((a["model"][k].float() - b["model"][k].float()).abs().max()) for k in a["model"])


def sel(fn, ctx, vm, b):
    try:
        return fn(ctx, vm, b)
    except (S.Invalid, CandidateLocalInvalid, M.ZeroCoverage) as e:
        return "INVALID:" + str(e)


def main():
    warnings.simplefilter("ignore", RuntimeWarning)
    c0, c3, c4 = ck("SEL1_SPEED_V0"), ck("SEL1_SPEED_V3"), ck("SEL1_SPEED_V4")
    w = dict(v0_v3=wdiff(c0, c3), v0_v4=wdiff(c0, c4), v3_v4=wdiff(c3, c4))
    print("weights max|diff| %s" % w, flush=True)
    cands = [tuple(tuple(x) if isinstance(x, list) else x for x in k) for k, _ in c4["hof"]]
    vm = Q.make_vm()
    F.install()
    ctx = S.build("null", torch.device("cuda"), vm, log=lambda m: None)
    bodies = list(dict.fromkeys(c[0] for c in cands))
    sel_c, sel_g, bad_sel = {}, {}, []
    for i, b in enumerate(bodies):
        sel_c[b], sel_g[b] = sel(S.vm_select, ctx, vm, b), sel(S.vm_select_gpu, ctx, vm, b)
        a, g = sel_c[b], sel_g[b]
        same = (a == g) if isinstance(a, str) or isinstance(g, str) else all(
            all(np.array_equal(x, y) for x, y in zip(p, q)) for p, q in zip(a, g))
        if not same:
            bad_sel.append(i)
    bad_rew = []
    for c in cands:
        a, g = sel_c[c[0]], sel_g[c[0]]
        if isinstance(a, str) or isinstance(g, str):
            continue
        ra, rg = S.bpd(S.score_sel(a, ctx, *c[1:])), S.bpd(S.score_sel(g, ctx, *c[1:]))
        if not (ra == rg or (np.isnan(ra) and np.isnan(rg))):
            bad_rew.append(dict(expr=vm.describe(list(c[0]), list(M.TERMINALS)), cpu=ra, gpu=rg))
    ok = not bad_sel and not bad_rew
    json.dump(dict(schema="SEL1_SELCHECK2_V1", weights_max_abs_diff=w, bodies=len(bodies), candidates=len(cands),
                   selections_differ=bad_sel, rewards_differ=bad_rew[:20], identical=ok),
              open(OUT, "w", encoding="utf-8"), indent=1)
    print("selcheck2 bodies %d cands %d | selections differ %d rewards differ %d | identical %s"
          % (len(bodies), len(cands), len(bad_sel), len(bad_rew), ok), flush=True)
    sys.exit(0 if ok else 7)


if __name__ == "__main__":
    main()
