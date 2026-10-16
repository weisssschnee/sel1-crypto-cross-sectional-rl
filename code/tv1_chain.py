"""TV1 auto-chain (TV1_ORDER s6-7, TV1_A2_BASIN4_ORDER s2-3): R1 -> R2 -> R4 -> R3 -> BASIN4.

    python tv1_chain.py [--resume-partial]

Each round: m2h_plant.py training (one process, GPU alone), nvidia-smi sampled every 60 s for the
VRAM peak, then tv1_round.py on the same planted / null world. R3 runs only if R1 was found.
BASIN4 (real data, no plant) starts only if R2 was found AND R4's report exists; otherwise the
chain writes the TV1 s7 reading and stops. After BASIN4: m2h_verdict_dense.py, if present.

Never overwrites: a round with a receipt or STOPPED.json is not retrained (its report is made if
missing); a round with checkpoints but neither is refused unless --resume-partial, which then
passes --resume auto. State in TV1_20261010\\TV1_CHAIN_STATE.json; times are 77o local (UTC+8).
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem"
ART = os.path.join(ROOT, "artifacts", "m2h_v1")
TV1 = os.path.join(ART, "TV1_20261010")
STATE = os.path.join(TV1, "TV1_CHAIN_STATE.json")
PY = sys.executable
DENSE = ["--batch", "128", "--tiles", "16", "--reward", "ic", "--select", "full"]
ROUNDS = dict(
    R1=dict(name="TV1_R1_OLD_P32", trainer="old", mode="p32",
            args=["--updates", "256", "--seed", "20261009", "--max-hours", "3.5"]),
    R2=dict(name="TV1_R2_DENSE_P8", trainer="dense", mode="p8",
            args=["--updates", "256", "--seed", "20261009", "--max-hours", "7"] + DENSE),
    R4=dict(name="TV1_R4_DENSE_NULL", trainer="dense", mode="null",
            args=["--updates", "256", "--seed", "20261009", "--max-hours", "7"] + DENSE),
    R3=dict(name="TV1_R3_OLD_P8", trainer="old", mode="p8",
            args=["--updates", "256", "--seed", "20261009", "--max-hours", "3.5"]),
)
B4_ARGS = ["--updates", "960", "--seed", "20261010", "--max-hours", "24"] + DENSE
ST = {}


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def save(**kw):
    ST.update(kw)
    json.dump(ST, open(STATE, "w", encoding="utf-8"), indent=1, ensure_ascii=False)


def say(msg):
    print("[%s] %s" % (now(), msg), flush=True)


def vram_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def run_sampled(cmd, log_path):
    """Run cmd (stdout+stderr to log_path); nvidia-smi every 60 s. -> (rc, peak MiB, samples)."""
    samples, stop = [], threading.Event()

    def loop():
        while not stop.is_set():
            v = vram_used_mib()
            if v is not None:
                samples.append(v)
            stop.wait(60)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write("=== %s %s\n" % (now(), " ".join(cmd)))
        fh.flush()
        rc = subprocess.run(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT).returncode
        fh.write("=== %s rc=%d\n" % (now(), rc))
    stop.set()
    th.join(timeout=5)
    return rc, (max(samples) if samples else None), len(samples)


def finished(run_dir):
    return [f for f in ("TRAIN_RECEIPT_V1.json", "STOPPED.json") if os.path.exists(os.path.join(run_dir, f))]


def train(key, cmd, run_dir, resume_partial):
    """Train unless already ended. Returns False (and stops the chain) on a refusal or failure."""
    if finished(run_dir):
        say("%s already ended %s; not retrained" % (key, finished(run_dir)))
        return True
    if glob.glob(os.path.join(run_dir, "checkpoint_*.pt")):
        if not resume_partial:
            say("%s REFUSED: checkpoints but no receipt / STOPPED; rerun with --resume-partial" % key)
            save(**{key: dict(status="refused_partial", at=now())})
            return False
        cmd = cmd + ["--resume", "auto"]
    save(**{key: dict(status="training", started=now(), cmd=cmd)})
    say("%s start" % key)
    rc, peak, n = run_sampled(cmd, os.path.join(TV1, "chain_%s.log" % key))
    log = os.path.join(run_dir, "BATCH_LOG.jsonl")
    secs = [json.loads(l)["seconds"] for l in open(log, encoding="utf-8") if l.strip()] if os.path.exists(log) else []
    rec = dict(status="trained" if rc == 0 and finished(run_dir) else "train_failed", rc=rc,
               started=ST[key]["started"], ended=now(), ended_by=finished(run_dir), updates=len(secs),
               sec_per_update=round(sum(secs[1:]) / max(len(secs) - 1, 1), 2) if secs else None,
               vram_peak_mib=peak, vram_samples=n, cmd=cmd)
    save(**{key: rec})
    say("%s %s rc=%d updates=%d sec/upd=%s vram_peak=%s MiB" % (key, rec["status"], rc, len(secs),
                                                              rec["sec_per_update"], peak))
    return rec["status"] == "trained"


def report(key, r, run_dir):
    """tv1_round.py on the round's own world -> found (bool) or None if the report failed."""
    rp = os.path.join(run_dir, "ROUND_REPORT.json")
    if not os.path.exists(rp):
        cmd = [PY, "-I", os.path.join(HERE, "tv1_round.py"), r["name"], "--trainer", r["trainer"],
               "--mode", r["mode"]]
        rc, _, _ = run_sampled(cmd, os.path.join(TV1, "chain_%s_report.log" % key))
        if rc != 0 or not os.path.exists(rp):
            save(**{key: dict(ST.get(key, {}), report="failed", report_rc=rc)})
            say("%s report FAILED rc=%d" % (key, rc))
            return None
    rep = json.load(open(rp, encoding="utf-8"))
    save(**{key: dict(ST.get(key, {}), report="done", found=rep["found"],
                      t1_batch_corr_max=rep["t1_batch_corr_max"], frozen_corr_max=rep["frozen_corr_max"],
                      frozen_sha256=rep["frozen_sha256"])})
    say("%s found=%s t1corr_max=%s frozen_corr_max=%s" % (key, rep["found"], rep["t1_batch_corr_max"],
                                                        rep["frozen_corr_max"]))
    return bool(rep["found"])


def do_round(key, resume_partial):
    r = ROUNDS[key]
    run_dir = os.path.join(ART, r["name"])
    cmd = [PY, "-I", os.path.join(HERE, "m2h_plant.py"), "--trainer", r["trainer"], "--mode", r["mode"],
           "--", r["name"]] + r["args"]
    if not train(key, cmd, run_dir, resume_partial):
        return None
    return report(key, r, run_dir)


def reading(found):
    """TV1_ORDER s7, registered before the runs."""
    out = []
    if found.get("R2"):
        out.append("R2 found: the dense reward is fixed; BASIN4 retrains the three families on real data")
    if found.get("R1") and found.get("R3") is False and found.get("R2"):
        out.append("R1 found, R3 not, R2 found: root cause = sparse reward, confirmed")
    if found.get("R1") is False:
        out.append("R1 not found: the old trainer misses even the strong plant; sampler / gradient path broken")
    if found.get("R2") is False:
        out.append("R2 not found: changing the reward alone is not enough; more tiles or the RGP layer, to decide")
    out.append("R4 frozen top-5 TRAIN net/gross = the level picked luck reaches")
    return out


def main():
    resume_partial = "--resume-partial" in sys.argv[1:]
    os.makedirs(TV1, exist_ok=True)
    if os.path.exists(STATE):
        ST.update(json.load(open(STATE, encoding="utf-8")))
    save(chain_started=now(), pid=os.getpid(), vram_idle_mib=vram_used_mib())
    found = {}
    for key in ("R1", "R2", "R4"):
        f = do_round(key, resume_partial)
        if f is None:
            save(chain_status="stopped_at_%s" % key, chain_ended=now())
            return
        found[key] = f
    if found["R1"]:
        f = do_round("R3", resume_partial)
        if f is None:
            save(chain_status="stopped_at_R3", chain_ended=now())
            return
        found["R3"] = f
    else:
        save(R3=dict(status="skipped", why="R1 not found (TV1_ORDER s6)"))
    save(found=found, reading=reading(found))
    if not (found["R2"] and ST.get("R4", {}).get("report") == "done"):
        save(chain_status="tv1_done_basin4_not_launched", basin4="not launched: needs R2 found and R4 done",
             chain_ended=now())
        say("TV1 done, BASIN4 not launched %s" % found)
        return

    name = ST.get("B4", {}).get("name") or "BASIN4_H1_DENSE_%s" % time.strftime("%Y%m%d")
    run_dir = os.path.join(ART, name)
    sec = ST["R2"].get("sec_per_update")
    eta_h = round(960 * sec / 3600.0, 1) if sec else None
    if not finished(run_dir):
        partial = bool(glob.glob(os.path.join(run_dir, "checkpoint_*.pt")))
        bat = "b4_resume.bat" if partial and resume_partial else "b4_launch.bat"
        save(B4=dict(name=name, status="training", started=now(), bat=bat, eta_h_from_R2=eta_h))
        say("BASIN4 %s start via %s, ETA %s h (R2 %s s/upd)" % (name, bat, eta_h, sec))
        rc, peak, n = run_sampled(["cmd", "/c", os.path.join(HERE, bat), name],
                                  os.path.join(TV1, "chain_B4.log"))
        save(B4=dict(ST["B4"], rc=rc, ended=now(), ended_by=finished(run_dir), vram_peak_mib=peak,
                     status="trained" if finished(run_dir) else "train_failed"))
        if not finished(run_dir):
            save(chain_status="stopped_at_B4", chain_ended=now())
            say("BASIN4 failed rc=%d" % rc)
            return
    vd = os.path.join(HERE, "m2h_verdict_dense.py")
    if os.path.exists(vd) and not os.path.exists(os.path.join(run_dir, "VERDICT_DENSE.json")):
        rc, _, _ = run_sampled([PY, "-I", vd, name], os.path.join(TV1, "chain_B4_verdict.log"))
        save(B4=dict(ST["B4"], verdict_rc=rc))
    save(chain_status="done", chain_ended=now())
    say("chain done")


if __name__ == "__main__":
    main()
