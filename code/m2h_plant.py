"""TV1 section 4 wrapper: planted signal / shuffled null, then the trainer's own main().

    python m2h_plant.py --trainer old|dense|dense2 --mode p32|p8|null -- <trainer args>

dense2 is m2h_dense_trainer_v2 (the same trainer plus --ent-coef / --adv-deadband, defaulting to
the v1 behaviour); used for R2b.

Imports the trainer, wraps m2h_common.build_world so that, right after the TRAIN world is
built, the ruler shard is replaced by
  p32 / p8  fwd[i,t] += alpha * s[i,t-1]  (s = D.plant_signal of DELTA(TS_MEAN(log_quote_volume,
            168),168) saved as TV1_S.npy, alpha from TV1_ALPHA.json; both written by tv1_selftest
            check 5, and s is checked against the calibrated sha)
  null      per hour one permutation of the pool coins, applied jointly to fwd / eligible /
            funding settlement (fi, fr); terminals untouched
and the new shard is run through prepare_shard, so fwd_w (the ruler's and the IC's winsorised
forward return) comes from the modified fwd. Terminals (feats) are never touched. The TRAIN
panel is the only one this process reads (load_train_grids refuses DEV / OOS); build_world may be
called once. PLANT_MANIFEST.json goes next to the trainer's LAUNCH_MANIFEST.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TV1_DIR = os.path.join(r"D:\riskpg_stateful_v5s_governor_nextgen_beem", "artifacts", "m2h_v1",
                       "TV1_20261010")
ALPHA_FILE = os.path.join(TV1_DIR, "TV1_ALPHA.json")
S_FILE = os.path.join(TV1_DIR, "TV1_S.npy")


def s_sha(s):
    return hashlib.sha256(np.ascontiguousarray(np.nan_to_num(s, nan=-9e9)).tobytes()).hexdigest()[:16]


def install(TR, mode, out_name):
    import torch
    import m2h_common as M
    import m2h_dense_common as D
    import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vm_mod
    orig = M.build_world
    calls = []

    def build_world(g):
        if calls:
            raise RuntimeError("build_world called twice: plant/null may touch only the TRAIN world")
        w = orig(g)
        base = w["shard"]
        rec = dict(schema="TV1_PLANT_MANIFEST_V1", mode=mode, out_name=out_name,
                   trainer=os.path.basename(TR.__file__), shape=list(base["fwd"].shape),
                   train_only=True, dev_read=False, oos_read=False)
        if mode == "null":
            sh = D.null_shard(base, w["univ"], D.PLANT_SEED)
            rec.update(null_seed=D.PLANT_SEED, permuted=["fwd", "eligible", "fi", "fr"])
        else:
            cal = json.load(open(ALPHA_FILE, encoding="utf-8"))[mode]
            s = np.load(S_FILE)                    # the s the self-check calibrated alpha on
            if s.shape != base["fwd"].shape:
                raise RuntimeError(f"s shape {s.shape} != fwd {base['fwd'].shape}")
            sha = s_sha(s)
            if sha != cal["s_sha16"]:
                raise RuntimeError(f"plant signal differs from the calibrated one: {sha} vs {cal['s_sha16']}")
            sh = D.plant_shard(base, s, cal["alpha"])
            rec.update(alpha=cal["alpha"], target_bpd=cal["target_bpd"], lift_bpd=cal["lift_bpd"],
                       s_sha16=sha, expr=cal["expr"])
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
        raise SystemExit("usage: m2h_plant.py --trainer old|dense --mode p32|p8|null -- <trainer args>")
    k = argv.index("--")
    own, rest = argv[:k], argv[k + 1:]
    opt = dict(zip(own[0::2], own[1::2]))
    trainer, mode = opt.get("--trainer"), opt.get("--mode")
    if trainer not in ("old", "dense", "dense2") or mode not in ("p32", "p8", "null"):
        raise SystemExit(f"bad wrapper args {own}")
    if trainer == "old":
        import m2h_trainer as TR
    elif trainer == "dense":
        import m2h_dense_trainer as TR
    else:                                   # v2 knobs (ent-coef / adv-deadband), R2b
        import m2h_dense_trainer_v2 as TR
    install(TR, mode, rest[0])
    sys.argv = [TR.__file__] + rest
    TR.main()


if __name__ == "__main__":
    main()
