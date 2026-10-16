"""XS_SEL1 s3.1 zero-diff check: two runs' BATCH_LOG / FORMULAS from update U on, line by line.
Fields that measure the run rather than the result (seconds, times, n_vm, n_book, vram_gib) are dropped; everything
else must be identical as JSON. -> one line: lines compared, differing lines, first difference.
    python -I sel1_logdiff.py RUN_A RUN_B FROM_UPDATE
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sel1_common as S

TIMING = ("seconds", "times", "n_vm", "n_book", "vram_gib")


def rows(run, fn, u0):
    out = []
    for ln in open(os.path.join(S.OUT_ROOT, run, fn), encoding="utf-8"):
        r = json.loads(ln)
        if r["update"] >= u0:
            for k in TIMING:
                r.pop(k, None)
            out.append(json.dumps(r, sort_keys=True))
    return out


def main():
    a, b, u0 = sys.argv[1], sys.argv[2], int(sys.argv[3])
    res = {}
    for fn in ("BATCH_LOG.jsonl", "FORMULAS.jsonl"):
        ra, rb = rows(a, fn, u0), rows(b, fn, u0)
        diff = [i for i, (x, y) in enumerate(zip(ra, rb)) if x != y]
        res[fn] = dict(n_a=len(ra), n_b=len(rb), n_diff=len(diff) + abs(len(ra) - len(rb)),
                       first=None if not diff else (ra[diff[0]][:160], rb[diff[0]][:160]))
    ok = all(v["n_diff"] == 0 and v["n_a"] > 0 for v in res.values())
    json.dump(dict(a=a, b=b, from_update=u0, files=res, zero_diff=ok),
              open(os.path.join(S.OUT_ROOT, b, "LOGDIFF_vs_%s.json" % a), "w"), indent=1)
    print("zero_diff %s | %s" % (ok, " ".join("%s %d/%d diff %d" % (k[:5], v["n_a"], v["n_b"], v["n_diff"]) for k, v in res.items())))


if __name__ == "__main__":
    main()
