import os, re
pat = re.compile(r"[Dd]ropout|\.eval\(\)|\.train\(")
roots = [r"D:/v4r_groot/Chengbo/workspaces/alphagpt-rl-dl-base/model_core",
         r"D:/riskpg_stateful_v5s_governor_nextgen_beem/runtime",
         r"D:/ChengboRemote/runtime/m1_stage1"]
n = 0
for root in roots:
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ("__pycache__", ".git", "artifacts")]
        for f in fn:
            if not f.endswith(".py"):
                continue
            p = os.path.join(dp, f)
            try:
                for i, line in enumerate(open(p, encoding="utf-8", errors="replace"), 1):
                    if pat.search(line):
                        print(f"{p}:{i}: {line.strip()[:140]}")
                        n += 1
                        if n > 40:
                            raise SystemExit
            except OSError:
                pass
print("matches", n)
