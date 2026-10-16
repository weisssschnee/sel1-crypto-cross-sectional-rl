"""Memo trainer check: B5_DBG_REAL_MEMOCHK (m2h_b5_trainer_dbg_memo resumed from a B5_DBG_REAL
checkpoint) against the rows the plain trainer logged for the same updates. Read-only. One line.
rc 0 = pass: the run finished (receipt), >= 128 rows were compared on matching tokens, no score
field differs on any of them, and the memo was active (n_scored < rows in some update).

    python b5_memochk.py FIRST_UPDATE LAST_UPDATE
"""
import json, os, sys
from collections import defaultdict

A = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1"
F = ("invalid_reason", "gross", "net", "ic", "window_gross", "window_ic", "window_nav", "gross_t",
     "corr_s", "live_frac", "turnover_per_day", "beta_mkt", "gross_bp_day", "hold_h", "basin")
u0, u1 = int(sys.argv[1]), int(sys.argv[2])


def rows(run):
    out = defaultdict(list)
    for l in open(os.path.join(A, run, "FORMULAS.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        if u0 <= r["update"] <= u1:
            out[r["update"]].append(r)
    return out


ref, chk = rows("B5_DBG_REAL"), rows("B5_DBG_REAL_MEMOCHK")
n = same_tok = bad = 0
first_ok, first_div = True, None
for u in range(u0, u1 + 1):
    a, b = ref.get(u, []), chk.get(u, [])
    if len(a) != len(b) or not a:
        first_ok = first_ok and u != u0
        first_div = first_div if first_div is not None else u
        continue
    for x, y in zip(a, b):
        n += 1
        if x["tokens"] != y["tokens"]:
            first_div = first_div if first_div is not None else u
            if u == u0:
                first_ok = False
            continue
        same_tok += 1
        if any(json.dumps(x.get(f)) != json.dumps(y.get(f)) for f in F):
            bad += 1
            if u == u0:
                first_ok = False
log = [json.loads(l) for l in open(os.path.join(A, "B5_DBG_REAL_MEMOCHK", "BATCH_LOG.jsonl"), encoding="utf-8")]
scored = [r.get("n_scored") for r in log if u0 <= r["update"] <= u1]
memo_on = any(s is not None and s < 128 for s in scored)
done = os.path.exists(os.path.join(A, "B5_DBG_REAL_MEMOCHK", "TRAIN_RECEIPT_V1.json"))
# first_update_identical is reported, not required: a token divergence would be resume-side
# (sampler / CUDA), not the memo; the memo claim is "same tokens -> same scores"
ok = done and bad == 0 and memo_on and same_tok >= 128
print("MEMOCHK u%d-%d rows %d same_tokens %d score_mismatch %d first_update_identical %s first_token_div %s "
      "n_scored %s..%s receipt %s -> %s" % (u0, u1, n, same_tok, bad, first_ok, first_div,
                                           min(scored, default=None), max(scored, default=None), done,
                                           "PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
