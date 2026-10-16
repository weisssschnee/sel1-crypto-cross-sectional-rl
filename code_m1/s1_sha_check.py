import hashlib, sys
for f in sys.argv[1:]:
    print(f, hashlib.sha256(open(r"D:/ChengboRemote/runtime/m1_stage1/" + f, "rb").read()).hexdigest()[:12])
