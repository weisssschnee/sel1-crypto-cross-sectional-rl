"""After killing a B5 trainer mid-update: drop a trailing line that is not valid JSON (a half-written
row) from BATCH_LOG / FORMULAS / PROGRESS_16 of one run, so --resume's _truncate_jsonl can parse
the files. Only the last line is ever touched; it belongs to an update after the last checkpoint,
which the resume drops anyway. One line.

    python b5_tailfix.py RUN_NAME
"""
import json, os, sys

A = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1"
run = sys.argv[1]
out = []
for fn in ("BATCH_LOG.jsonl", "FORMULAS.jsonl", "PROGRESS_16.jsonl"):
    p = os.path.join(A, run, fn)
    if not os.path.exists(p):
        out.append("%s missing" % fn)
        continue
    raw = open(p, "rb").read()
    lines = raw.split(b"\n")
    tail = lines[-1] if lines[-1].strip() else (lines[-2] if len(lines) > 1 else b"")
    try:
        json.loads(tail.decode("utf-8")) if tail.strip() else None
        out.append("%s ok" % fn)
    except (ValueError, UnicodeDecodeError):
        cut = raw.rfind(b"\n", 0, len(raw.rstrip(b"\n")))
        open(p, "wb").write(raw[:cut + 1])
        out.append("%s dropped %d bytes" % (fn, len(raw) - cut - 1))
print("TAILFIX %s: %s" % (run, "; ".join(out)))
