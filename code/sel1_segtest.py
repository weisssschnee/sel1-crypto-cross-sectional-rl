"""XS_SEL1 补充 A: segment-sampled scoring vs the full A half (GPU VM, V4 TRAIN real world, A months only; no B,
no DEV, no OOS). New file, diagnostics only.
Candidates: the unique valid (expr, h, state, dir) of SEL1_SPEED_V0 FORMULAS (768 sampled by the trainer), rescored
here in the real world. Year chunks run on their own (raw_signal) and the stateful book resets every month, so a
segment's per-month rows equal the full run's rows for those months: segment reward = cost0 bp/day over the
segment's A months, read off one full run. Segments: each single year (5), each pair of years (10).
Seconds: VM per chunk measured; book measured on all 30 A months and pro-rated by month count.
Validity / coverage are the full run's.  -> SEL1_SEGTEST.json, one summary line.
"""
import itertools
import json
import os
import re
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
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid, OPERATOR_NAMES

OUT = os.path.join(S.OUT_ROOT, "SEL1_SEGTEST.json")
SRC = os.path.join(S.OUT_ROOT, "SEL1_SPEED_V0", "FORMULAS.jsonl")
_TOK = re.compile(r"\s*(?:(w=)|([A-Za-z_][A-Za-z0-9_]*)|(\d+)|([(),]))")


def tokens(vm, text):
    """vm.describe text -> postfix VM tokens + END (as b5_reward_pick.parse_expr / m2h_rgp.names_to_tokens)."""
    tk, pos = [], 0
    while pos < len(text):
        m = _TOK.match(text, pos)
        tk.append(next(g for g in m.groups() if g))
        pos = m.end()
    out, i = [], 0

    def node():
        nonlocal i
        name = tk[i]
        i += 1
        win = None
        if i < len(tk) and tk[i] == "(":
            i += 1
            while True:
                if tk[i] == "w=":
                    win = int(tk[i + 1])
                    i += 2
                else:
                    node()
                i += 1
                if tk[i - 1] == ")":
                    break
        if win is not None:
            out.append(vm.window_base + list(M.WINDOWS).index(win))
        if name in M.TERMINALS:
            out.append(M.TERMINALS.index(name))
        else:
            out.append(vm.operator_base + list(OPERATOR_NAMES).index(name))

    node()
    toks = out + [vm.end_token]
    if vm.describe(toks, list(M.TERMINALS)) != text:
        raise ValueError(text)
    return tuple(toks)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    warnings.simplefilter("ignore", RuntimeWarning)
    vm = Q.make_vm()
    F.install()
    ctx = S.build("real", torch.device("cuda"), vm, log=lambda m: None)
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    cands = list(dict.fromkeys((r["expr"], r["h"], S.STATES.index(r["state"]), S.DIRS.index(r["dir"]))
                               for r in rows if r["invalid"] is None))
    am = [m for m in ctx["months"] if m["half"] == "A"]
    years = sorted({m["year"] for m in am})
    sig_cache, t_chunk, full, seg, t_book = {}, [], [], [], []
    for expr, h, st, dr in cands:
        if expr not in sig_cache:
            body = tokens(vm, expr)
            out = np.full(ctx["mask"].shape, np.nan)
            tc = []
            with torch.no_grad():
                for in0, o0, o1 in ctx["chunks"]:
                    torch.cuda.synchronize()
                    t = time.time()
                    s = vm.execute(list(body), ctx["feats_gpu"][:, :, in0:o1]).detach().cpu().numpy()
                    out[:, o0:o1] = s[:, o0 - in0:]
                    tc.append(time.time() - t)
            try:
                sig_cache[expr] = (S.select_months(S.mask_signal(ctx, out), ctx), tc)
            except (S.Invalid, CandidateLocalInvalid, M.ZeroCoverage):
                sig_cache[expr] = (None, tc)
        sels, tc = sig_cache[expr]
        if sels is None:
            continue
        t = time.time()
        r = S.score_sel(sels, ctx, h, st, dr)
        t_book.append(time.time() - t)
        t_chunk.append(tc)
        full.append(S.bpd(r))
        seg.append({y: [x for x in r if x["year"] == y] for y in years})
    full = np.array(full)
    t_chunk = np.array(t_chunk)
    book = float(np.mean(t_book))
    ny = {y: sum(m["year"] == y for m in am) for y in years}
    yi = {y: k for k, (in0, o0, o1) in enumerate(ctx["chunks"]) for y in [S.TRAIN_FIRST.year + k]}
    res = []
    for size in (1, 2):
        for ys in itertools.combinations(years, size):
            v = np.array([S.bpd(sum((s[y] for y in ys), [])) for s in seg])
            ok = np.isfinite(v) & np.isfinite(full)
            sec = float(t_chunk[:, [yi[y] for y in ys]].sum(axis=1).mean() + book * sum(ny[y] for y in ys) / len(am))
            top = full >= np.quantile(full[ok], 0.9)
            pick = v >= np.quantile(v[ok], 0.75)
            res.append(dict(years=list(ys), rho=spearman(v[ok], full[ok]), s_per_formula=sec,
                            full_top10_in_seg_top25=float((top & pick).sum() / top.sum())))
    full_s = float(t_chunk.sum(axis=1).mean() + book)
    rep = dict(candidates=len(cands), scored=len(full), bodies=len(sig_cache), fused=True,
               full_s_per_formula=full_s, vm_s_per_body_chunk=t_chunk.mean(axis=0).round(4).tolist(),
               book_s_per_candidate=book, segments=res, peak_vram_gib=torch.cuda.max_memory_allocated() / 2 ** 30,
               kernels=len(F._KERN), dev_read=False, oos_read=False)
    json.dump(rep, open(OUT, "w"), indent=1)
    for size in (1, 2):
        g = [x for x in res if len(x["years"]) == size]
        print("seg %dy: rho mean %.3f min %.3f | s/formula %.3f | full top10%% in seg top25%% %.2f" % (
            size, np.mean([x["rho"] for x in g]), min(x["rho"] for x in g), np.mean([x["s_per_formula"] for x in g]),
            np.mean([x["full_top10_in_seg_top25"] for x in g])), flush=True)
    print("full A: %.3f s/formula (vm %.3f + book %.3f) | %d candidates, %d bodies" % (
        full_s, t_chunk.sum(axis=1).mean(), book, len(full), len(sig_cache)), flush=True)


if __name__ == "__main__":
    main()
