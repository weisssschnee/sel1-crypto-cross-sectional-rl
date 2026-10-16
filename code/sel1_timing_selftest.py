"""Self-test of the SEL1 timing layer (CPU only, no training). One JSON line per part, rc 0 = all pass.

  synth   a synthetic extended month with planted trigger events and planted signal paths; every fwd /
          funding value is a small dyadic number, so the hand calculation below (scalar loop, its own
          freeze / exit / hedge bookkeeping) must equal sel1_timing_common.settle_event bit for bit:
            E1 in / out inside the window, a funding event      E2 short, exit decided inside a coin gap (Rule 0)
            E3 forced exit searched past two dead minutes        E4 coin not in the shard -> NO_COIN
            E5 trigger before the lookback -> NO_LOOKBACK         E6 no tradable exit minute -> NO_EXIT
          plus the always-in baseline, the reward / drawdown, and the same events through the trainer's
          eval_one with the signal planted as a VM terminal (--vm).
  real    (--real) on the TRAIN shards: V4 / shard coin overlap, the hour-close alignment of V4 trade_close
          against the 1-min close, and a plumbing pass on events planted on real coins (writes the event csv the
          trainer's --no-step smoke reads).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sel1_timing_common as TC

D8, D9, D11, D12 = 2.0 ** -8, 2.0 ** -9, 2.0 ** -11, 2.0 ** -12
RATE = {c: TC.specs()[c][1] for c in TC.COSTS}


def synth_shard():
    A, off, n = 8, TC.LOOKBACK, 3 * 1440
    T = off + n + TC.HEAD
    syms = ["C%02d" % i for i in range(A - 1)] + ["BTCUSDT"]
    t = np.arange(T)
    fwd = np.empty((A, T))
    for a in range(A - 1):
        fwd[a] = D8 * (((t + a) % 5) - 2)
    fwd[A - 1] = D9 * ((t % 3) - 1)
    return dict(month="2024-03", first=pd.Timestamp("2024-03-01"), symbols=syms, ix={s: i for i, s in enumerate(syms)},
                bi=A - 1, off=off, n_min=n, T=T, fwd=fwd, fi=np.zeros((A, T), bool), fr=np.zeros((A, T)),
                feats=np.full((A, len(TC.TERMS), T), 1.0, np.float32))


def w0_of(trig, sh):
    return sh["off"] + int((pd.Timestamp(trig) - sh["first"]) / pd.Timedelta(minutes=1)) + TC.HOUR_CLOSE_MIN + 60


def plant(sh):
    """Events + planted signal (A, T) + the hand-written position path of each kept event."""
    L = TC.LAG
    s = np.full(sh["fwd"].shape, np.nan)
    ev, hand = [], {}
    # E1 long C00, beta 0.5: in on w0+10 .. w0+49, out from w0+50; funding on both legs at w0+30
    t1 = "2024-03-01 02:00"
    w0 = w0_of(t1, sh)
    s[0, w0 + 10 - L:w0 + 50 - L] = 1.0
    s[0, w0 + 50 - L] = -1.0
    sh["fi"][0, w0 + 30], sh["fr"][0, w0 + 30] = True, D11
    sh["fi"][-1, w0 + 30], sh["fr"][-1, w0 + 30] = True, D12
    ev.append(dict(sym="C00", trig_hour=t1, n_rows=2, dir=1, weight=0.1, beta=0.5))
    hand["C00"] = (w0, w0 + 120, w0 + 120, [(w0 + 10, w0 + 50)], 1, 0.5)
    # E2 short C01, beta 1: in from w0 (decided at dec0), exit decided for w0+22 inside the coin gap w0+20..w0+24
    t2 = "2024-03-02 05:00"
    w0 = w0_of(t2, sh)
    s[1, w0 - L:w0 + 22 - L] = -1.0
    s[1, w0 + 22 - L] = 1.0
    sh["fwd"][1, w0 + 20:w0 + 25] = np.nan
    ev.append(dict(sym="C01", trig_hour=t2, n_rows=1, dir=-1, weight=0.1, beta=1.0))
    hand["C01"] = (w0, w0 + 60, w0 + 60, [(w0, w0 + 22)], -1, 1.0)
    # E3 long C02, beta 0.5, always in; minutes w1, w1+1 untradable -> exit settled at w1+2
    t3 = "2024-03-02 20:00"
    w0 = w0_of(t3, sh)
    s[2, w0 - L - 5:w0 + 240] = 1.0
    sh["fwd"][2, w0 + 120:w0 + 122] = np.nan
    ev.append(dict(sym="C02", trig_hour=t3, n_rows=2, dir=1, weight=0.1, beta=0.5))
    hand["C02"] = (w0, w0 + 120, w0 + 122, [(w0, w0 + 120)], 1, 0.5)
    # E4 / E5 / E6 drops
    ev.append(dict(sym="ZZZ", trig_hour=t1, n_rows=1, dir=1, weight=0.1, beta=1.0))
    ev.append(dict(sym="C03", trig_hour="2024-02-29 20:00", n_rows=1, dir=1, weight=0.1, beta=1.0))
    t6 = "2024-03-03 10:00"
    w0 = w0_of(t6, sh)
    sh["fwd"][4, w0 + 60:] = np.nan
    ev.append(dict(sym="C04", trig_hour=t6, n_rows=1, dir=1, weight=0.1, beta=1.0))
    for e in ev:
        e.update(n_weights=1, src="selftest")
    s[[3, 5, 6, 7]] = 1.0
    return pd.DataFrame(ev), s, hand


def hand_settle(sh, a, w0, w1, x, spans, d, b, rate):
    """Scalar loop: position p (coin) and h = -b p (BTC); each leg's turnover on its own frozen path."""
    nets, prev_a, prev_b = [], 0.0, 0.0
    for t in range(w0, x + 1):
        g = 1.0 if any(lo <= t < hi for lo, hi in spans) and t < w1 else 0.0
        p = d * 1.0 * g
        h = -b * p
        fa, fb = sh["fwd"][a, t], sh["fwd"][sh["bi"], t]
        ta, tb = np.isfinite(fa), np.isfinite(fb)
        cur_a = p if ta else prev_a
        cur_b = h if tb else prev_b
        turn = abs(cur_a - prev_a) + abs(cur_b - prev_b)
        gross = (p * fa if ta else 0.0) + (h * fb if tb else 0.0)
        fund = (p * sh["fr"][a, t] if sh["fi"][a, t] else 0.0) + (h * sh["fr"][sh["bi"], t] if sh["fi"][sh["bi"], t] else 0.0)
        nets.append(gross - rate * turn - fund)
        prev_a, prev_b = cur_a, cur_b
    return np.asarray(nets)


def check_synth(use_vm=False):
    sh = synth_shard()
    ev, s, hand = plant(sh)
    wins, drops = TC.windows(ev, sh)
    out = dict(part="synth", drops=drops, drops_ok=drops == {"NO_COIN": 1, "NO_LOOKBACK": 1, "NO_EXIT": 1})
    got = {w["sym"]: w for w in wins}
    out["windows_ok"] = sorted(got) == sorted(hand) and all(
        (got[k]["w0"], got[k]["w1"], got[k]["x"]) == hand[k][:3] for k in hand)
    segs = [TC.seg_of(s[w["a"]], w) for w in wins]
    sc = {r["sym"]: r for r in TC.score_events(segs, wins, sh)}
    base = {r["sym"]: r for r in TC.always_in(wins, sh)}
    bit, rows = True, []
    for k, (w0, w1, x, spans, d, b) in hand.items():
        a = sh["ix"][k]
        for name, res, sp in (("timed", sc[k], spans), ("always_in", base[k], [(w0, w1)])):
            for c in TC.COSTS:
                nets = hand_settle(sh, a, w0, w1, x, sp, d, b, RATE[c])
                lp = np.log1p(np.clip(nets, -0.9, None))
                ln = float(np.sum(lp))
                ok = ln == res[c]
                if c == "flat6":
                    cum = np.cumsum(lp)
                    dd = float((np.maximum.accumulate(np.r_[0.0, cum])[1:] - cum).max())
                    ok = ok and dd == res["dd"]
                bit = bit and ok
                rows.append(dict(ev=k, book=name, cost=c, hand=ln, code=res[c], equal=ok))
    out["bit_equal"] = bit
    out["rows"] = [r for r in rows if not r["equal"]] or "all %d equal" % len(rows)
    # reward: bp per event-day of flat6 - DD_LAMBDA * dd
    days = sum(r["days"] for r in sc.values())
    rw = 1e4 * sum(r["flat6"] - TC.DD_LAMBDA * r["dd"] for r in sc.values()) / days
    out["reward_ok"] = rw == TC.reward(list(sc.values()))
    out["reward"] = rw
    out["always_in_reward"] = TC.reward(list(base.values()))
    # lag: a signal flip at minute w0+10-LAG reaches the position at w0+10, not earlier
    w = got["C00"]
    g = TC.gate_path(TC.seg_of(s[0], w), w)
    out["lag_ok"] = bool(g[9] == 0.0 and g[10] == 1.0 and g[49] == 1.0 and g[50] == 0.0)
    # no look-ahead: changing the signal after the decision minute does not change the gate path up to it
    s2 = s.copy()
    s2[0, w["w0"] + 10 - TC.LAG + 1:] = np.where(np.isfinite(s2[0, w["w0"] + 10 - TC.LAG + 1:]), -1.0, np.nan)
    out["gate_causal_ok"] = bool(np.array_equal(TC.gate_path(TC.seg_of(s2[0], w), w)[:11], g[:11]))
    # cap
    out["cap_ok"] = bool(max(np.abs(TC.held_of(TC.gate_path(TC.seg_of(s[k_], got[k]), got[k]), got[k])[:, 0]).max()
                             for k_, k in ((0, "C00"), (1, "C01"), (2, "C02"))) <= TC.POS_CAP)
    if use_vm:
        out.update(check_vm(sh, s, wins, sc, segs))
    ok = all(out[k] for k in ("drops_ok", "windows_ok", "bit_equal", "reward_ok", "lag_ok", "gate_causal_ok",
                                "cap_ok")) and out.get("vm_ok", True)
    out["ok"] = bool(ok)
    return out


def check_vm(sh, s, wins, sc, segs):
    """The planted signal as VM terminal 0, body [0, END], through the trainer's per-event slices and eval_one."""
    import torch
    import sel1_timing_trainer as TT
    sys.path.insert(0, os.path.join(TC.C.ROOT, "src"))
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import XsTsCsStackVM, CandidateLocalInvalid
    vm = XsTsCsStackVM(len(TC.TERMS), windows=TC.C.WINDOWS)
    sh = dict(sh)
    f = sh["feats"].copy()
    f[:, 0, :] = s.astype(np.float32)
    sh["feats"] = f
    sl = TC.event_slices(wins, sh, torch.device("cpu"))
    body = [0, vm.end_token]
    sv = TC.event_signals(vm, body, sl, wins, sh)
    same = all(np.array_equal(a, b, equal_nan=True) for a, b in zip(sv, segs))
    r_h, r_s, vec, dg = TT.eval_one(vm, body, (sl, sl), sh, wins, wins, TC.DD_LAMBDA,
                                    (CandidateLocalInvalid, TC.Invalid))
    return dict(vm_signal_equal=same, vm_reward=r_h, vm_ok=bool(same and r_h == TC.reward(list(sc.values()))))


def check_events_from_rows():
    """Spell extraction: a long spell, a direct flip (two spells), a spell to the window end; BTC column skipped."""
    held = np.zeros((10, 3))
    held[1:4, 0] = 0.1
    held[4:6, 0] = -0.1
    held[7:, 1] = 0.2
    held[:, 2] = 9.0
    ts = pd.date_range("2024-03-01", periods=800, freq="h")
    ctx = dict(ts=ts, syms=["AAA", "BBB", "BTCUSDT"], beta=np.arange(3 * 800, dtype=float).reshape(3, 800))
    ev = TC.events_from_rows([dict(held=held, cols=np.array([0, 1, 2]))], [dict(m0=100)], ctx)
    want = [("AAA", str(ts[100]), 3, 1, 100.0), ("AAA", str(ts[103]), 2, -1, 103.0),
            ("BBB", str(ts[106]), 3, 1, 800 + 106.0)]
    got = [(r["sym"], r["trig_hour"], r["n_rows"], r["dir"], r["beta"]) for r in ev.to_dict("records")]
    return dict(part="events_from_rows", ok=got == want, got=got)


V4_GRID = os.path.join(TC.C.ROOT, "runtime", "xs_v4_panel_v1", "V4_TRAIN_GRID.npz")
PROBE_MONTHS = ("2024-03", "2025-08")


def check_real(out_csv, n_events=48, seed=20261015):
    z = np.load(V4_GRID, allow_pickle=False)
    v4_syms = [str(s) for s in z["symbols"]]
    hours = pd.DatetimeIndex(z["hours"].astype("datetime64[ns]"))
    if hours.max() >= pd.Timestamp("2026-07-01"):
        raise SystemExit("V4 grid reaches OOS")
    tc = np.asarray(z["trade_close"], dtype=np.float64)
    rep = dict(part="real", overlap=TC.overlap_report(v4_syms))
    # hour-label alignment: V4 trade_close[hour r] vs 1-min close at minute 60 r + k, k in {0, 59, 60}
    al = {}
    vi = {s: i for i, s in enumerate(v4_syms)}
    for m in PROBE_MONTHS:
        sh = TC.load_shard(m)
        h0 = int(hours.get_loc(pd.Timestamp(m + "-01")))
        nh = sh["n_min"] // 60
        for k in (0, 59, 60):
            rel = []
            for s in sh["symbols"]:
                if s not in vi:
                    continue
                v = tc[vi[s], h0:h0 + nh]
                c = np.full(nh, np.nan)
                idx = np.arange(nh) * 60 + k
                ok = idx < sh["n_min"]
                c[ok] = sh["close"][sh["ix"][s], idx[ok]]
                f = np.isfinite(v) & np.isfinite(c) & (v > 0)
                rel.append(np.abs(c[f] / v[f] - 1.0))
            r = np.concatenate(rel) if rel else np.zeros(0)
            al["%s_k%d" % (m, k)] = dict(n=int(r.size), exact=round(float((r == 0).mean()), 4) if r.size else None,
                                         within_1bp=round(float((r < 1e-4).mean()), 4) if r.size else None)
        TC.drop_cache()
    best = max((0, 59, 60), key=lambda k: np.mean([al["%s_k%d" % (m, k)]["within_1bp"] or 0 for m in PROBE_MONTHS]))
    rep["alignment"] = al
    rep["alignment_best_k"] = best
    rep["alignment_ok"] = best == TC.HOUR_CLOSE_MIN
    del tc
    # planted events on real coins (beta 1.0 placeholder; real events carry SEL1's trigger-hour beta)
    rng = np.random.default_rng(seed)
    m = PROBE_MONTHS[0]
    sh = TC.ext_shard(m)
    cand = [s for s in sh["symbols"] if s in vi and s != "BTCUSDT"]
    ev = []
    for _ in range(n_events):
        d = int(rng.integers(2, 26))
        ev.append(dict(sym=cand[int(rng.integers(len(cand)))],
                       trig_hour=str(pd.Timestamp(m + "-01") + pd.Timedelta(days=d, hours=int(rng.integers(24)))),
                       n_rows=int(rng.choice([4, 24, 72])), dir=int(rng.choice([-1, 1])), weight=0.1, beta=1.0,
                       n_weights=1, src="selftest_real"))
    ev = pd.DataFrame(ev, columns=list(TC.EVENT_COLS))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    ev.to_csv(out_csv, index=False)
    ev = TC.read_events(out_csv)
    wins, drops = TC.windows(ev, sh)
    sig = np.where(np.isfinite(sh["fwd"]), np.sign(np.cos(np.arange(sh["T"]) / 97.0))[None, :], np.nan)
    # settle_event asserts settle_path == settle_multi per event
    sc = TC.score_events([TC.seg_of(sig[w["a"]], w) for w in wins], wins, sh)
    base = TC.always_in(wins, sh)
    rep["plumbing"] = dict(events=len(ev), kept=len(wins), drops=drops, events_csv=out_csv,
                           timed=TC.summary(sc), always_in=TC.summary(base),
                           resume_turn_share_max=round(max(r["resume_turn_share"] for r in sc + base), 4))
    rep["ok"] = bool(rep["alignment_ok"] and len(wins) > 0)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vm", action="store_true", help="also run the planted signal through the VM + eval_one")
    ap.add_argument("--real", action="store_true", help="also run the TRAIN-shard checks")
    ap.add_argument("--out", default=os.path.join(TC.C.ROOT, "artifacts", "m2h_v1", "XS_SEL1", "TIMING"))
    a = ap.parse_args()
    parts = [check_events_from_rows(), check_synth(a.vm)]
    if a.real:
        parts.append(check_real(os.path.join(a.out, "SELFTEST_EVENTS_REAL.csv")))
    rep = dict(schema="SEL1_TIMING_SELFTEST_V1", ok=all(p["ok"] for p in parts), parts=parts,
               written=pd.Timestamp.now().isoformat())
    if a.real:
        os.makedirs(a.out, exist_ok=True)
        json.dump(rep, open(os.path.join(a.out, "SELFTEST_REPORT.json"), "w"), indent=1, default=str)
    for p in parts:
        print(json.dumps({k: v for k, v in p.items() if k != "rows"} | {"rows": p.get("rows")}, default=str)[:900])
    print("SELFTEST", "PASS" if rep["ok"] else "FAIL")
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
