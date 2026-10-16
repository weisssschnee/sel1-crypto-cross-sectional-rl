"""SEL2 probe (metadata only): one minute shard's keys / shapes / dtypes, the minute axis head, and SHARD_MANIFEST keys.
TRAIN shard 2024-01 only; no values are summarised."""
import json
import os
import sys

import numpy as np

D = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\runtime\m1_staging\m1_shards"
z = np.load(os.path.join(D, "m1_shard_2024-01.npz"), allow_pickle=False)
print({k: (z[k].shape, str(z[k].dtype)) for k in z.files})
for k in z.files:
    if z[k].ndim == 1 and z[k].shape[0] < 2000000 and z[k].dtype.kind in "iuMUSO":
        print(k, z[k][:3])
m = json.load(open(os.path.join(D, "SHARD_MANIFEST.json"), encoding="utf-8"))
print(sorted(m.keys()) if isinstance(m, dict) else type(m))
print({k: m[k] for k in m if not isinstance(m[k], (list, dict))} if isinstance(m, dict) else "")
sys.path.insert(0, r"D:\ChengboRemote\runtime\m1_stage1")
import m1_common as C
print(len(C.TERMINALS), C.TERMINALS)
