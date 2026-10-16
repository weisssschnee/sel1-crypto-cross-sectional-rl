import hashlib, sys
for f in sys.argv[1:]:
    print(f.replace("\\", "/").rsplit("/", 1)[-1], hashlib.sha256(open(f, "rb").read()).hexdigest()[:16])
