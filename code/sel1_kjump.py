"""XS_SEL1 补充 B': K-jump check of the power table (check only, no training; V4 TRAIN only, no DEV, no OOS). New file.
The plant formula (SEL1_ALPHA.json body, alpha unchanged) at K in {5, 10, 20}: all 60 combos (hold x state x dir),
pc world and real world, A half and B half: cost0 bp/day, monthly t, mean coin count held (n_pos), mean gross coin
exposure sum|w| and mean |BTC leg|. Worlds built exactly as sel1_power.py.  -> SEL1_KJUMP.json, key lines.
"""
from __future__ import annotations

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
import sel1_calib as C
import cr1_common as Q
import m2h_dense_common as D

OUT = os.path.join(S.OUT_ROOT, "SEL1_KJUMP.json")
KS = (5, 10, 20)


def stats(rows):
    m = np.asarray([1e4 * r["cost0"] / r["days"] for r in rows])
    g, b = [], []
    for r in rows:
        if "held" in r:
            h = r["held"]
            g.append(np.abs(h[:, :-1]).sum(axis=1).mean())
            b.append(np.abs(h[:, -1]).mean())
        else:
            g.append(0.0)
            b.append(0.0)
    return dict(bpd=round(S.bpd(rows), 3), t=round(S.tstat(m), 3), n_pos=round(float(np.mean([r["n_pos"] for r in rows])), 2),
                gross=round(float(np.mean(g)), 3), btc=round(float(np.mean(b)), 3))


def main():
    t0 = time.time()
    vm = Q.make_vm()
    pb = tuple(D.plant_body(vm))
    aj = json.load(open(S.ALPHA_JSON, encoding="utf-8"))
    if list(pb) != aj["plant_body"]:
        raise SystemExit("plant body differs from SEL1_ALPHA.json")
    alpha = float(aj["alpha"])
    real = S.build("real", torch.device("cuda"), vm, log=lambda m: None, keep_base=True)
    add = np.zeros_like(real["shard"]["fwd"])
    add[:, 1:] = np.nan_to_num(real["zp"][:, :-1], nan=0.0)
    pc = dict(real)                                                  # sel1_power / sel1_calib.world(alpha)
    pc["shard"] = S.no_winsor_shard(real["shard"], real["dead"], fwd=real["shard"]["fwd"] + alpha * add)
    pc["mode"] = "pc"
    worlds = dict(pc=(pc, S.signal(pc, vm, pb)), real=(real, S.signal(real, vm, pb)))
    tab = {}
    for k in KS:
        S.K = k
        rows = []
        for h, st, dr in C.COMBOS:
            row = dict(h=h, state=S.STATES[st], dir=S.DIRS[dr])
            for w, (ctx, sig) in worlds.items():
                for half in ("A", "B"):
                    row["%s_%s" % (w, half)] = stats(S.score(sig, ctx, h, st, dr, halves=(half,), keep=True))
            rows.append(row)
        best = max(rows, key=lambda r: r["pc_A"]["bpd"] if np.isfinite(r["pc_A"]["bpd"]) else -1e9)
        tab[k] = dict(rows=rows, pc_A_best=dict(h=best["h"], state=best["state"], dir=best["dir"]))
        for r in rows:
            key = (r["h"], r["state"], r["dir"])
            if key in ((4, "all", "both"), (72, "all", "long"), (best["h"], best["state"], best["dir"])):
                print("K%-2d %3d/%-12s %-5s %s| pc A %7.2f t %5.2f B %7.2f t %5.2f | real A %7.2f t %5.2f B %7.2f t %5.2f"
                      " | n_pos %5.1f gross %.2f btc %.2f" % (
                          k, r["h"], r["state"], r["dir"], "*" if r is best else " ",
                          r["pc_A"]["bpd"], r["pc_A"]["t"], r["pc_B"]["bpd"], r["pc_B"]["t"],
                          r["real_A"]["bpd"], r["real_A"]["t"], r["real_B"]["bpd"], r["real_B"]["t"],
                          r["pc_B"]["n_pos"], r["pc_B"]["gross"], r["pc_B"]["btc"]), flush=True)
        a = np.array([r["pc_A"]["bpd"] for r in rows])
        print("K%-2d pc A rank of 4/all/both %d / 60" % (k, 1 + int((a > next(
            r["pc_A"]["bpd"] for r in rows if (r["h"], r["state"], r["dir"]) == (4, "all", "both"))).sum())), flush=True)
    json.dump(dict(schema="SEL1_KJUMP_V1", alpha=alpha, table=tab, dev_read=False, oos_read=False,
                   seconds=round(time.time() - t0, 1)), open(OUT, "w", encoding="utf-8"), indent=1, default=float)
    print("kjump done %.0fs" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
