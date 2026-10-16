"""BASIN5_PC_ENT plant wrapper: m2h_b5_plant.py with the entropy-floor trainer
(m2h_b5_trainer_ent) in place of m2h_b5_trainer. m2h_plant.install (TV1 plant / null world) is
NOT modified; the planted s, alpha and its sha check are TV1's.

    python m2h_b5_plant_ent.py --mode p8|p32|null -- <m2h_b5_trainer_ent args>
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: m2h_b5_plant_ent.py --mode p8|p32|null -- <trainer args>")
    k = argv.index("--")
    own, rest = argv[:k], argv[k + 1:]
    opt = dict(zip(own[0::2], own[1::2]))
    mode = opt.get("--mode")
    if mode not in ("p32", "p8", "null") or len(own) != 2:
        raise SystemExit(f"bad wrapper args {own}")
    import m2h_b5_trainer_ent as TR      # sets sys.path for m1_* / okx_alpha_discovery
    import m2h_plant as PL
    PL.install(TR, mode, rest[0])
    sys.argv = [TR.__file__] + rest
    TR.main()


if __name__ == "__main__":
    main()
