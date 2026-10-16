"""Where does one formula sit in a run's HOF_RANKED.json (TRAIN 50-tile ruler)? Read-only. One line.

    python b5_hofpos.py RUN_NAME FAMILY "EXPR"
"""
import json, os, sys

A = r"D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1"
run, fam, expr = sys.argv[1], sys.argv[2], sys.argv[3]
h = json.load(open(os.path.join(A, run, "HOF_RANKED.json"), encoding="utf-8"))
rows = h["families"][fam] if "families" in h else h[fam]
for i, r in enumerate(rows):
    if r.get("expr") == expr:
        print("HOFPOS %s %s rank %d/%d conc_k10_train_net %s full_gross_t50 %s n_logged %s"
              % (run, fam, i + 1, len(rows), r.get("conc_k10_train_net_bpd"), r.get("full_gross_t50"),
                 r.get("n_logged")))
        break
else:
    print("HOFPOS %s %s not found among %d; keys %s" % (run, fam, len(rows), list(h.keys())[:6]))
