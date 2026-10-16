"""BASIN5 debug-round plant wrapper: m2h_plant.install's world swap (fwd += alpha * s[t-1], then
prepare_shard; terminals untouched; build_world may run once) for m2h_b5_trainer_dbg, with the
alpha looked up by b5_fixed.alpha_of: p8 / p32 from TV1_ALPHA.json, p16 from B5_ALPHA_P16.json
(b5_fixed.calibrate_p16). m2h_plant.py is NOT modified. null = D.null_shard as in m2h_plant.

    python m2h_b5_plant_dbg.py --mode p8|p16|p32|null -- <m2h_b5_trainer_dbg args>
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def install(TR, mode, out_name):
    import numpy as np
    import m2h_common as M
    import m2h_dense_common as D
    import b5_fixed as F
    orig = M.build_world
    calls = []

    def build_world(g):
        if calls:
            raise RuntimeError("build_world called twice: plant/null may touch only the TRAIN world")
        w = orig(g)
        base = w["shard"]
        rec = dict(schema="TV1_PLANT_MANIFEST_V1", mode=mode, out_name=out_name, wrapper="m2h_b5_plant_dbg",
                   trainer=os.path.basename(TR.__file__), shape=list(base["fwd"].shape),
                   train_only=True, dev_read=False, oos_read=False)
        if mode == "null":
            sh = D.null_shard(base, w["univ"], D.PLANT_SEED)
            rec.update(null_seed=D.PLANT_SEED, permuted=["fwd", "eligible", "fi", "fr"])
        else:
            cal = F.alpha_of(mode)
            s = np.load(TR.S_FILE)
            if s.shape != base["fwd"].shape:
                raise RuntimeError(f"s shape {s.shape} != fwd {base['fwd'].shape}")
            if F.sha16(s) != cal["s_sha16"]:
                raise RuntimeError("plant signal differs from the calibrated one")
            sh = D.plant_shard(base, s, cal["alpha"])
            rec.update(alpha=cal["alpha"], target_bpd=cal["target_bpd"], lift_bpd=cal["lift_bpd"],
                       s_sha16=cal["s_sha16"], expr=cal["expr"])
        w["shard"], w["fwd"] = sh, sh["fwd"]
        calls.append(1)
        rec["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        out_dir = os.path.join(TR.ROOT, "artifacts", "m2h_v1", out_name)
        os.makedirs(out_dir, exist_ok=True)
        json.dump(rec, open(os.path.join(out_dir, "PLANT_MANIFEST.json"), "w"), indent=1)
        print("plant", mode, rec.get("alpha"), flush=True)
        return w

    M.build_world = build_world


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: m2h_b5_plant_dbg.py --mode p8|p16|p32|null -- <trainer args>")
    k = argv.index("--")
    own, rest = argv[:k], argv[k + 1:]
    opt = dict(zip(own[0::2], own[1::2]))
    mode = opt.get("--mode")
    if mode not in ("p8", "p16", "p32", "null") or len(own) != 2:
        raise SystemExit(f"bad wrapper args {own}")
    import m2h_b5_trainer_dbg as TR
    install(TR, mode, rest[0])
    sys.argv = [TR.__file__] + rest
    TR.main()


if __name__ == "__main__":
    main()
