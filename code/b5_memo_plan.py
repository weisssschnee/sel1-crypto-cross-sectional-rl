"""Check-run plan for b5_dbg_memo.bat: the second-latest B5_DBG_REAL checkpoint and the 16 updates
after it (all logged by the plain trainer, and kept by the main resume's truncation). Prints
"CK_PATH U0 U1", or exits 1 if there are fewer than two checkpoints."""
import glob, os, sys

A = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1"
cks = sorted(glob.glob(os.path.join(A, "B5_DBG_REAL", "checkpoint_*.pt")))
if len(cks) < 2:
    sys.exit(1)
n = int(os.path.basename(cks[-2])[len("checkpoint_"):-len(".pt")])
m = int(os.path.basename(cks[-1])[len("checkpoint_"):-len(".pt")])
print(cks[-2], n + 1, m)
