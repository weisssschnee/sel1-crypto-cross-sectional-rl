import os, sys
sys.path.insert(0, r"D:\ChengboRemote\runtime\basin3_h1")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cr1_common as Q
vm = Q.make_vm()
names = ("is_terminal", "is_window", "is_operator", "operator_name", "window_value", "is_end", "describe")
print("vm", {n: hasattr(vm, n) for n in names})
from okx_alpha_discovery.adapters import xs_tscs_vm_v2 as V
print("V", hasattr(V, "OPERATOR_MIN_WINDOW"), hasattr(V, "WINDOWED_OPERATORS"))
import sel1_common as S
print("S", hasattr(S, "plant_add"), hasattr(S, "no_winsor_shard"), hasattr(S, "Invalid"), hasattr(S, "COVER_MIN"), hasattr(S, "MIN_COINS"))
for m in ("sel1_dedup", "sel1_pool", "sel1_gfn", "sel1_world2", "sel1_trainer_x", "sel1_ga"):
    __import__(m)
print("imports ok")
