"""Build S1_STAGE1_20261008_VERDICT next to the original run dir.

Holds only the 256 update-0 rows of the rerun (same init and seed as the original run) as FORMULAS.jsonl, so that
s1_paired_eval.py can read its INITIAL batch. Nothing in the original run dir or the backup is modified.
"""
import hashlib
import json
import os
import sys

ART = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2_v1"
RUN = os.path.join(ART, "S1_STAGE1_20261008")
SRC = os.path.join(RUN, "_INCIDENT_20261009_RERUN_PARTIAL", "FORMULAS.jsonl")
BACKUP = r"D:\ChengboRemote\runtime\m1_stage1\S1_STAGE1_20261008_ORIG_BACKUP"
DST = os.path.join(ART, "S1_STAGE1_20261008_VERDICT")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


if os.path.exists(DST):
    sys.exit(f"{DST} exists, refusing")
keep, upd_hist = [], {}
with open(SRC, "rb") as fh:
    for raw in fh:
        if not raw.strip():
            continue
        r = json.loads(raw)
        upd_hist[r.get("update")] = upd_hist.get(r.get("update"), 0) + 1
        if r.get("update") == 0:
            keep.append(raw if raw.endswith(b"\n") else raw + b"\n")
print("source updates seen:", min(upd_hist), "..", max(upd_hist), "n_updates", len(upd_hist))
print("update-0 rows:", len(keep), "row keys:", sorted(json.loads(keep[0]).keys()))
if len(keep) != 256:
    sys.exit("expected 256 update-0 rows, abort")
ck_run = sha(os.path.join(RUN, "checkpoint_0127.pt"))
ck_bak = sha(os.path.join(BACKUP, "checkpoint_0127.pt"))
print("checkpoint_0127 run==backup:", ck_run == ck_bak, ck_run[:16])
if ck_run != ck_bak:
    sys.exit("checkpoint_0127 differs from backup, abort")
os.makedirs(DST)
out = os.path.join(DST, "FORMULAS.jsonl")
with open(out, "wb") as fh:
    fh.writelines(keep)
rows = [json.loads(b) for b in keep]
prov = dict(
    purpose="INITIAL batch for the paired DEV verdict of S1_STAGE1_20261008",
    source=SRC, source_sha256=sha(SRC), rows=len(keep), out_sha256=sha(out),
    basis=("rerun update-0 rows: same init and seed as the original run; the original FORMULAS.jsonl was truncated "
           "by the 23:58 relaunch. Printed .log fields of updates 0,1,2,3 match the original."),
    final_checkpoint=os.path.join(RUN, "checkpoint_0127.pt"), final_checkpoint_sha256=ck_run,
    n_distinct_body_hold=len({(tuple(r["body"]), r["hold_min"]) for r in rows}))
json.dump(prov, open(os.path.join(DST, "VERDICT_PROVENANCE.json"), "w"), indent=1)
print(json.dumps(prov, indent=1))
