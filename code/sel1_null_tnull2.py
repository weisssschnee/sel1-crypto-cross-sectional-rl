"""XS_SEL1 mid-run T_null, copy of sel1_null_tnull.py with progress lines and --tag (10-16 rerun after rc=-1). NULL world only; no real data, no DEV, no OOS.
The NULL checkpoint's hall -> top 500 exactly as the trainer writes HOF.json (sorted by -score) -> two numbers:
  T_null_mid       max t_B over that hall, sel1_round's rescore (A+B, the entry's own h/state/dir).
  T_combo_null_mid composite book t_B of that hall, sel1_combo.members + book (as COMBO_NULL.json).
Also prints COMBO_PC.json's book t_B. sel1_round / sel1_combo are imported, not modified.
    python -I sel1_null_tnull.py [--ckpt PATH] [--device cpu|cuda] [--top N]
-> SEL1_NULL\TNULL_MID_<ckpt>.json, one line.
"""
import argparse
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
import sel1_fastts as F
import sel1_combo as C
import sel1_pool as P
import cr1_common as Q
import m2h_common as M
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import CandidateLocalInvalid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(S.OUT_ROOT, "SEL1_NULL", "checkpoint_0527.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--top", type=int, default=500)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(4)
    d = os.path.join(S.OUT_ROOT, "SEL1_NULL")
    man = json.load(open(os.path.join(d, "TRAIN_MANIFEST.json"), encoding="utf-8"))
    S.K = int(man["world"].get("k", 5))
    vm = Q.make_vm()
    F.install()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    hof = {tuple(tuple(x) if isinstance(x, list) else x for x in k): v for k, v in ck["hof"]}
    top = sorted(hof.items(), key=lambda kv: -kv[1].get("score", kv[1]["reward_bpd"]))[:a.top]
    ents = [dict(body=list(c[0]), st=c[2], dr=c[3], **e) for c, e in top]
    ctx = S.build(man["world"]["mode"], torch.device(a.device), vm, alpha=man["world"]["alpha"], log=lambda m: None)
    ctx["dd_max"] = float(man["args"]["dd_max"])
    best, n_ok = None, 0
    print("hall %d world %.0fs" % (len(ents), time.time() - t0), flush=True)
    for i, e in enumerate(ents):                                # sel1_round.rescore's lines, no corr
        if i % 25 == 0:
            print("rescore %d/%d %.0fs best %s" % (i, len(ents), time.time() - t0, None if best is None else round(best[0], 3)), flush=True)
        try:
            sg = S.signal(ctx, vm, tuple(e["body"]))
        except (S.Invalid, CandidateLocalInvalid, M.ZeroCoverage):
            continue
        rows = S.score_sel(S.select_months(sg, ctx, ("A", "B")), ctx, e["h"], e["st"], e["dr"], ("A", "B"), keep=True)
        t_b = float(S.reward([r for r in rows if r["half"] == "B"], "t"))
        if np.isfinite(t_b):
            n_ok += 1
            if best is None or t_b > best[0]:
                best = (t_b, e["expr"], e["h"], e["state"], e["dir"])
        del sg
    print("rescore done %.0fs" % (time.time() - t0), flush=True)
    lay = P.Layout(ctx, ("A", "B"))
    chosen, seen, d_max = C.members(ents, ctx, vm, lay, 0)
    bk = C.book(chosen, ctx, lay) if chosen else None
    pc = json.load(open(os.path.join(S.OUT_ROOT, "SEL1_PC", "COMBO_PC.json"), encoding="utf-8"))
    rep = dict(ckpt=os.path.basename(a.ckpt), update=int(ck["update"]), hall=len(ents), rescored=n_ok,
               T_null_mid=None if best is None else best[0], T_null_entry=None if best is None else best[1:],
               T_combo_null_mid=None if bk is None else bk["t_B"], combo_members=len(chosen), rescore_max_abs_diff=d_max,
               combo_pc_tB=(pc.get("book") or {}).get("t_B"), combo_pc_selfcheck=pc.get("selfcheck_passed"),
               seconds=round(time.time() - t0, 1), dev_read=False, oos_read=False)
    json.dump(rep, open(os.path.join(d, "TNULL_MID_%s%s.json" % (os.path.splitext(rep["ckpt"])[0], a.tag)), "w"), indent=1, default=float)
    print("u%d T_null_mid %.3f | T_combo_null_mid %s (%d members) | COMBO_PC t_B %s selfcheck %s | %.0fs"
          % (rep["update"], rep["T_null_mid"] or float("nan"), rep["T_combo_null_mid"], len(chosen), rep["combo_pc_tB"],
             rep["combo_pc_selfcheck"], rep["seconds"]), flush=True)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
