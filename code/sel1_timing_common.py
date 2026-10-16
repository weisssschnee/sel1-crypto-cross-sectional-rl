"""SEL1 timing layer: 1-min entry / exit inside the SEL1 frozen book's trigger events (order XS_SEL1, 深夜
section). New file; m1_common / m1_ruler are imported, never modified. CPU-only preparation: no training until
SEL1 has a frozen candidate past its gate and the user approves.

  events   one row per holding spell of a frozen SEL1 book (events_from_rows): coin, trigger hour (the signal
           hour of the spell's first row), rows held, direction, SEL1 weight, beta at the trigger hour (SEL1's
           hedge ratio). Only spells inside 2024-01..2025-12 can be used: the 1-min shards cover TRAIN 24-25
           and only their own coins (overlap_report).
  clock    V4 hours are taken as open-labelled: the V4 close of hour r is the 1-min close of minute r + 59
           (alignment_probe checks this on the real shards). The trigger is known at the close of the signal
           hour, minute t0 = minute(trig_hour) + 59; the window is minutes [t0, t0 + 60 * n_rows).
  action   formula signal s on the shard's cross-section, VM run per event on its own slice (LOOKBACK minutes
           before the first decision minute, as m2's episode slices; events without that history are dropped;
           the whole month in one call peaked at 32 GB on CPU). Per minute: s > 0 = in, s < 0 = out,
           s == 0 or NaN = stay; the action decided at minute t is executed at t + LAG (LAG = 2 >= 1 bar).
           The window starts flat. Position p = dir * POS_CAP * g, g in {0, 1}: |p| <= POS_CAP = 1 per event
           (the SEL1 weight is recorded, not applied).
  hedge    BTC leg = -beta * p, beta of the trigger hour (as SEL1).
  exit     forced at the first minute x >= t0 + W where coin and BTC both have a forward return (searched
           EXIT_SEARCH minutes; none -> the event is dropped as NO_EXIT). Rows in between carry the position.
  settle   m1_ruler.settle_multi on [coin, BTC]: flat6 / cost0, funding events, Rule 0 freeze, fwd without
           winsor (as SEL1). settle_path replays it per minute (equal sum asserted bit for bit).
  reward   1e4 * sum_e (flat6 log NAV_e - DD_LAMBDA * maxDD_e) / sum_e days_e, bp per event-day; maxDD of the
           event's per-minute flat6 log NAV path from 0. The always-in book (s = +1) is reported beside it,
           never rewarded.
"""
from __future__ import annotations

import collections
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m1_common as C
import m1_ruler as R

LAG = C.EXECUTION_LAG_MIN
LOOKBACK = C.LOOKBACK_MINUTES
TERMS = C.SOURCE_COLUMNS                 # the 16 columns the TRAIN shards carry (no funding_level)
SHARD_DIR = os.path.join(C.ROOT, "runtime", "m1_staging", "m1_shards")
TRAIN_MONTHS = tuple("%d-%02d" % (y, m) for y in (2024, 2025) for m in range(1, 13))
TRAIN_END = pd.Timestamp("2026-01-01")
HOUR_CLOSE_MIN = 59
POS_CAP = 1.0
DD_LAMBDA = 1.0
EXIT_SEARCH = 60
MIN_LIVE = 0.1
COSTS = ("flat6", "cost0")
EVENT_COLS = ("sym", "trig_hour", "n_rows", "dir", "weight", "beta", "n_weights", "src")


class Invalid(Exception):
    pass


def specs():
    return {c: R.cost_specs(R.NAV_USD)[c] for c in COSTS}


# --------------------------------------------------------------------------- shards
_CACHE = {}


def load_shard(month, shard_dir=SHARD_DIR):
    """m2_trainer.load_shard's arrays (fwd = C.forward_returns, raw-ms funding events) plus the clock."""
    key = (shard_dir, month)
    if key in _CACHE:
        return _CACHE[key]
    path = os.path.join(shard_dir, f"m1_shard_{month}.npz")
    last = None
    for k in range(6):
        try:
            z = np.load(path, allow_pickle=False)
            break
        except PermissionError as e:
            last = e
            time.sleep(10)
    else:
        raise last
    symbols = [str(s) for s in z["symbols"]]
    feats, close = z["feats"], z["close"]
    if feats.shape[1] != len(TERMS):
        raise RuntimeError(f"{month}: shard has {feats.shape[1]} terminals, timing layer expects {len(TERMS)}")
    fwd = C.forward_returns(close, None)
    fund = z["fund"].reshape(-1, 3)
    fi = np.zeros_like(fwd, dtype=bool)
    fr = np.zeros_like(fwd)
    a_i, m_i = fund[:, 0].astype(int), fund[:, 1].astype(int)
    keep = (m_i >= 0) & (m_i < fwd.shape[1]) & (a_i >= 0) & (a_i < fwd.shape[0])
    fi[a_i[keep], m_i[keep]] = True
    fr[a_i[keep], m_i[keep]] = fund[:, 2][keep]
    if "BTCUSDT" not in symbols:
        raise RuntimeError(f"{month}: no BTCUSDT in the shard (hedge leg)")
    sh = dict(month=month, first=pd.Timestamp(month + "-01"), symbols=symbols,
              ix={s: i for i, s in enumerate(symbols)}, bi=symbols.index("BTCUSDT"),
              n_min=int(z["n_min"]), feats=feats, close=close, fwd=fwd, fi=fi, fr=fr)
    _CACHE[key] = sh
    return sh


def drop_cache(keep=()):
    for k in [k for k in _CACHE if k[1] not in keep]:
        del _CACHE[k]


# --------------------------------------------------------------------------- events
def events_from_rows(rows, ms, ctx, src=""):
    """SEL1 rows (sel1_common.score_sel(..., keep=True)) of the months ms (same order) -> event table.
    A spell is a run of constant sign in a coin column (a direct flip starts a new spell); the trigger hour
    is the signal hour of its first row, m0 + s0 - 1; beta is SEL1's hedge ratio there."""
    ts, syms = ctx["ts"], ctx["syms"]
    out = []
    for r, m in zip(rows, ms):
        held, cols = r["held"], r["cols"]
        for j in range(held.shape[1] - 1):              # last column is the BTC leg
            hj = held[:, j]
            sg = np.sign(hj)
            edges = np.flatnonzero(np.diff(np.r_[0.0, sg, 0.0]) != 0)
            for s0, s1 in zip(edges[:-1], edges[1:]):
                if sg[s0] == 0:
                    continue
                a, hr = int(cols[j]), int(m["m0"] + s0 - 1)
                out.append(dict(sym=syms[a], trig_hour=str(ts[hr]), n_rows=int(s1 - s0), dir=int(sg[s0]),
                                weight=float(abs(hj[s0])), beta=float(ctx["beta"][a, hr]),
                                n_weights=int(np.unique(np.abs(hj[s0:s1])).size), src=src))
    return pd.DataFrame(out, columns=list(EVENT_COLS))


def read_events(path):
    ev = pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)
    missing = set(EVENT_COLS) - set(ev.columns)
    if missing:
        raise SystemExit(f"{path}: event table lacks {sorted(missing)}")
    th = pd.to_datetime(ev["trig_hour"])
    if (th >= TRAIN_END).any():
        raise SystemExit(f"{path}: events at or after {TRAIN_END} (DEV / OOS) refused; TRAIN events only")
    ev = ev.assign(month=th.dt.strftime("%Y-%m"))
    return ev


# --------------------------------------------------------------------------- extended month (seams)
HEAD = HOUR_CLOSE_MIN + 1 + EXIT_SEARCH     # minutes of the next month a spell ending at the month end needs


def _shift_month(month, k):
    p = pd.Period(month, "M") + k
    return str(p)


def _put(dst, src_sh, syms, key, lo, hi, at):
    """dst[:, ..., at:at + hi - lo] = src_sh[key][rows of syms, ..., lo:hi]; coins absent from src stay."""
    for i, s in enumerate(syms):
        j = src_sh["ix"].get(s)
        if j is not None:
            dst[i, ..., at:at + hi - lo] = src_sh[key][j, ..., lo:hi]


def ext_shard(month, shard_dir=SHARD_DIR):
    """The month on its own symbols, with the previous month's last LOOKBACK minutes in front (VM history) and
    the next month's first HEAD minutes behind (spells that end at the month end), both only inside TRAIN
    2024-01..2025-12 (2025-12 gets no tail: 2026-01 is DEV). fwd is C.forward_returns on the joined close, so
    the seam minute has its return. off = index of the month's minute 0."""
    if month not in TRAIN_MONTHS:
        raise SystemExit(f"{month}: timing layer reads TRAIN 2024-01..2025-12 shards only")
    cur = load_shard(month, shard_dir)
    pm, nm = _shift_month(month, -1), _shift_month(month, 1)
    prv = load_shard(pm, shard_dir) if pm in TRAIN_MONTHS else None
    nxt = load_shard(nm, shard_dir) if nm in TRAIN_MONTHS else None
    P = LOOKBACK if prv is not None else 0
    N = HEAD if nxt is not None else 0
    syms, n = cur["symbols"], cur["n_min"]
    A, F, T = len(syms), len(TERMS), P + n + N
    feats = np.full((A, F, T), np.nan, dtype=np.float32)
    close = np.full((A, T), np.nan)
    fi = np.zeros((A, T), dtype=bool)
    fr = np.zeros((A, T))
    feats[:, :, P:P + n], close[:, P:P + n] = cur["feats"], cur["close"]
    fi[:, P:P + n], fr[:, P:P + n] = cur["fi"], cur["fr"]
    for src, lo, hi, at in ((prv, (prv or {}).get("n_min", 0) - P, (prv or {}).get("n_min", 0), 0),
                            (nxt, 0, N, P + n)):
        if src is None or hi <= lo:
            continue
        for key, dst in (("feats", feats), ("close", close), ("fi", fi), ("fr", fr)):
            _put(dst, src, syms, key, lo, hi, at)
    fwd = C.forward_returns(close, None)
    return dict(month=month, first=cur["first"], symbols=syms, ix=cur["ix"], bi=cur["bi"], off=P, n_min=n,
                T=T, feats=feats, close=close, fwd=fwd, fi=fi, fr=fr, has_prev=prv is not None,
                has_next=nxt is not None)


# --------------------------------------------------------------------------- event windows
def window_of(e, sh):
    """Event row -> dict(a, w0, w1, x, dec0, dir, beta) in the extended shard's minute index, or a drop reason.
    w0 = SEL1's entry minute (close of the hour after the trigger hour), w1 = w0 + 60 n_rows (SEL1's exit),
    x = first minute >= w1 where coin and BTC both trade (the forced exit is settled there)."""
    a = sh["ix"].get(e["sym"])
    if a is None:
        return "NO_COIN"
    mt = int(round((pd.Timestamp(e["trig_hour"]) - sh["first"]) / pd.Timedelta(minutes=1))) + HOUR_CLOSE_MIN
    tc = sh["off"] + mt                          # minute the trigger is known (close of the trigger hour)
    w0 = tc + 60
    w1 = w0 + 60 * int(e["n_rows"])
    dec0 = w0 - LAG
    if dec0 < LOOKBACK:
        return "NO_LOOKBACK"
    trad = np.isfinite(sh["fwd"][[a, sh["bi"]]]).all(axis=0)
    x = next((t for t in range(w1, min(w1 + EXIT_SEARCH + 1, sh["T"])) if trad[t]), None)
    if x is None:
        return "NO_EXIT"
    return dict(a=int(a), w0=int(w0), w1=int(w1), x=int(x), dec0=int(dec0), dir=int(e["dir"]),
                beta=float(np.nan_to_num(e["beta"])), weight=float(e["weight"]), sym=e["sym"],
                trig_hour=str(e["trig_hour"]))


def windows(ev_month, sh):
    out, drops = [], collections.Counter()
    for e in ev_month.to_dict("records"):
        w = window_of(e, sh)
        if isinstance(w, str):
            drops[w] += 1
        else:
            out.append(w)
    return out, dict(drops)


def gate_path(seg, w):
    """Timing state g on minutes w0 .. x from the formula signal on the decision minutes dec0 .. w1 - LAG - 1
    (seg), direction applied (s_eff = dir * s: 'in' when the formula agrees with the event's side). The decision
    at minute t sets g from t + LAG on: s_eff > 0 -> 1, s_eff < 0 -> 0, 0 / NaN -> unchanged; flat at w0, 0 from
    w1 on."""
    seg = np.asarray(seg, dtype=np.float64)
    if seg.shape != (w["w1"] - LAG - w["dec0"],):
        raise ValueError("signal segment length != decision minutes of the event")
    d = w["dir"] * seg
    dec = np.where(d > 0, 1.0, np.where(d < 0, 0.0, np.nan))
    g = pd.Series(dec).ffill().fillna(0.0).to_numpy()
    out = np.zeros(w["x"] - w["w0"] + 1)
    out[:w["w1"] - w["w0"]] = g
    return out


def held_of(g, w):
    p = w["dir"] * POS_CAP * g
    return np.column_stack([p, -w["beta"] * p])


def arrays_of(w, sh):
    cols = [w["a"], sh["bi"]]
    sl = slice(w["w0"], w["x"] + 1)
    return sh["fwd"][cols, sl].T, sh["fi"][cols, sl].T, sh["fr"][cols, sl].T


# --------------------------------------------------------------------------- settlement
def settle_path(held, fw, fi, fr, rate):
    """settle_multi's per-minute net for one flat rate, replayed row by row (Rule 0 freeze, turnover on the
    frozen path, gross / funding on held). Element-wise the same operations in the same order, so the sum of
    log1p(net) equals settle_multi's log_nav bit for bit (asserted in settle_event)."""
    T, A = held.shape
    trad = np.isfinite(fw)
    prev = np.zeros(A)
    net = np.empty(T)
    for t in range(T):
        cur = np.where(trad[t], held[t], prev) if t else np.where(trad[0], held[0], 0.0)
        turn = np.abs(cur - prev).sum()
        gross = np.nansum(held[t] * np.nan_to_num(fw[t]))
        fund = np.nansum(held[t] * np.where(fi[t], fr[t], 0.0))
        net[t] = gross - rate * turn - fund
        prev = cur
    return net


def settle_event(g, w, sh):
    held = held_of(g, w)
    fw, fi, fr = arrays_of(w, sh)
    assert held.shape == fw.shape == fi.shape == fr.shape, "held / fwd shape differs"
    sp = specs()
    res, parts = R.settle_multi(held, fw, fi, fr, sp)
    out = dict(sym=w["sym"], trig_hour=w["trig_hour"], days=held.shape[0] / 1440.0, turn=parts["turnover"],
               in_frac=float(g[:w["w1"] - w["w0"]].mean()) if w["w1"] > w["w0"] else 0.0,
               resume_turn_share=parts["resume_turn_share"])
    for c in COSTS:
        net = settle_path(held, fw, fi, fr, sp[c][1])
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = np.log1p(np.clip(net, -0.9, None))
        ln = float(np.sum(lp))
        if ln != res[c]["log_nav"]:
            raise AssertionError(f"settle_path != settle_multi ({c}): {ln!r} vs {res[c]['log_nav']!r}")
        out[c] = ln
        if c == "flat6":
            cum = np.cumsum(lp)
            out["dd"] = float((np.maximum.accumulate(np.r_[0.0, cum])[1:] - cum).max())
    return out


# --------------------------------------------------------------------------- formula signal
def event_slices(wins, sh, device):
    """Per event the feature tensor of minutes [dec0 - LOOKBACK, w1 - LAG): LOOKBACK of history plus the event's
    decision minutes (m2's episode slice). Built once per update, shared by every formula; each event's signal
    depends on its own slice only, never on which other events the month has."""
    import torch
    return [torch.from_numpy(np.ascontiguousarray(sh["feats"][:, :, w["dec0"] - LOOKBACK:w["w1"] - LAG])).to(
        device=device, dtype=torch.float32) for w in wins]


def event_signals(vm, body, slices, wins, sh):
    """Formula signal on each event's decision minutes dec0 .. w1 - LAG - 1: (coin row, live fraction per probe).
    Raises Invalid when, at a probe minute (first decision minute and quarter points) of any event, the formula
    is live on fewer than MIN_LIVE of the coins that trade there."""
    import torch
    out, pf = [], []
    with torch.no_grad():
        for x, w in zip(slices, wins):
            s = vm.execute(list(body), x).detach().cpu().numpy()[:, LOOKBACK:].astype(np.float64)
            n = s.shape[1]
            for k in sorted({0, n // 4, n // 2, 3 * n // 4}):
                el = np.isfinite(sh["fwd"][:, w["dec0"] + k])
                if el.sum():
                    pf.append(float((np.isfinite(s[:, k]) & el).sum() / el.sum()))
            out.append(s[w["a"]])
    if not pf or min(pf) < MIN_LIVE:
        raise Invalid("ZERO_COVERAGE_MIN_%.2f" % (min(pf) if pf else 0.0))
    return out


def seg_of(s_full_row, w):
    """A full-length signal row cut to the event's decision minutes (tests / planted signals)."""
    return np.asarray(s_full_row[w["dec0"]:w["w1"] - LAG], dtype=np.float64)


def score_events(segs, wins, sh):
    return [settle_event(gate_path(sg, w), w, sh) for sg, w in zip(segs, wins)]


def always_in(wins, sh):
    """Baseline: the SEL1 event as SEL1 takes it on the 1-min clock (in on every row of the window)."""
    return [settle_event(np.r_[np.ones(w["w1"] - w["w0"]), np.zeros(w["x"] - w["w1"] + 1)], w, sh) for w in wins]


def reward(scores, lam=DD_LAMBDA):
    """bp per event-day of flat6 log NAV minus lam * per-event max drawdown (log units)."""
    d = sum(r["days"] for r in scores)
    if not d:
        return float("nan")
    return 1e4 * sum(r["flat6"] - lam * r["dd"] for r in scores) / d


def summary(scores, lam=DD_LAMBDA):
    if not scores:
        return dict(events=0)
    d = sum(r["days"] for r in scores)
    return dict(events=len(scores), reward=round(reward(scores, lam), 4),
                flat6_bpd=round(1e4 * sum(r["flat6"] for r in scores) / d, 4),
                cost0_bpd=round(1e4 * sum(r["cost0"] for r in scores) / d, 4),
                dd_mean=round(float(np.mean([r["dd"] for r in scores])), 6),
                in_frac=round(float(np.mean([r["in_frac"] for r in scores])), 4),
                turn_per_event=round(float(np.mean([r["turn"] for r in scores])), 4))


# --------------------------------------------------------------------------- coin overlap (report)
def overlap_report(v4_syms, shard_dir=SHARD_DIR, events=None):
    """How many V4 coins the 1-min shards carry (any month of 2024-25), and, given an event table, how many of
    its events have their coin in the event month's shard."""
    per_month, union = {}, set()
    for m in TRAIN_MONTHS:
        z = np.load(os.path.join(shard_dir, f"m1_shard_{m}.npz"), allow_pickle=False)
        s = [str(x) for x in z["symbols"]]
        per_month[m] = s
        union |= set(s)
    v4 = set(v4_syms)
    rep = dict(v4_coins=len(v4), shard_coins=len(union), both=len(v4 & union),
               shard_not_v4=sorted(union - v4), per_month_shard=[len(per_month[m]) for m in TRAIN_MONTHS])
    if events is not None:
        ok = [e["sym"] in set(per_month.get(e["month"], ())) for e in events.to_dict("records")]
        rep.update(events=len(ok), events_in_shard=int(sum(ok)))
    return rep
