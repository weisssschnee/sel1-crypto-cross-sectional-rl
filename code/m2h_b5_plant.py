"""BASIN5 plant wrapper: m2h_plant.install (TV1 plant / null world, NOT modified) around
m2h_b5_trainer.main().

    python m2h_b5_plant.py --mode p8|p32|null -- <m2h_b5_trainer args>

m2h_plant.py only knows the old / dense / dense2 trainers; this file passes the B5 trainer to
the same install() so the planted s, alpha and its sha check are TV1's.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: m2h_b5_plant.py --mode p8|p32|null -- <trainer args>")
    k = argv.index("--")
    own, rest = argv[:k], argv[k + 1:]
    opt = dict(zip(own[0::2], own[1::2]))
    mode = opt.get("--mode")
    if mode not in ("p32", "p8", "null") or len(own) != 2:
        raise SystemExit(f"bad wrapper args {own}")
    import m2h_b5_trainer as TR          # sets sys.path for m1_* / okx_alpha_discovery
    import m2h_plant as PL
    PL.install(TR, mode, rest[0])
    sys.argv = [TR.__file__] + rest
    TR.main()


if __name__ == "__main__":
    main()
