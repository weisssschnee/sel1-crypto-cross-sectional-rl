"""XS_SEL1 combo sweep (order XS_SEL1, supplement 10-15 morning). New file; sel1_common is imported, not modified.

The policy samples the formula only. Each formula is scored on all 60 combos (hold 4 x state 5 x direction 3, in
sel1_calib.COMBOS order) and takes the combo with the highest A-half score (--reward, t_dd in the chain), ties to the
first in that order: the plant check's rule (sel1_nullplant.cell_task), so SEL1_NULLPLANT stays valid.

Not a leg decomposition. The book is not linear in the legs: the stateful controller decides HOLD / PARTIAL / SWITCH
on the L1 distance of the WHOLE book (both = long + short legs together), and the state switch zeroes the desired
weights before that controller, so a combo's held path is not a mix of the legs' held paths. What is shared:
  - one apply_stateful_action_policy_batch call per (hold, state) over its 3 directions x A months windows
    (S.held_compact; rows are independent, zero padding to a common width leaves numpy's pairwise sums unchanged);
    a window with no selected coin holds nothing (cost0 0, as settle_window);
  - cost0 only: settle_multi's cost0 log NAV is sum log1p(clip(gross - fund)) with gross / fund on held, not on the
    Rule-0 path (cost0 charges nothing), so cost0_window drops the turnover pass and is otherwise settle_window's
    lines. Equality with S.score_sel per combo is checked by sel1_sweepcheck.
The chosen combo is then re-scored through S.score_sel (net, turnover, cost0 exact), and that score is the reward.
"""
from __future__ import annotations

import itertools

import numpy as np

import sel1_common as S

COMBOS = list(itertools.product(S.HOLDS, range(len(S.STATES)), range(len(S.DIRS))))   # = sel1_calib.COMBOS


def cost0_window(held, cols, ctx, m0, m1):
    """settle_window(...)['cost0'] without the flat6 / turnover pass (bit-identical)."""
    if cols.size == 0:
        return 0.0
    sh = ctx["shard"]
    E = m1 - m0
    held = held.copy()
    for j, a in enumerate(cols):
        L = int(ctx["Larr"][a])
        if m0 <= L < m1:
            held[L - m0:, j] = 0.0
    rr = np.arange(E)[:, None]
    chg = np.vstack([held[:1] != 0, np.diff(held, axis=0) != 0])
    last = np.maximum.accumulate(np.where(chg, rr, -1), axis=0)
    bfull = ctx["beta"][cols, m0 - 1:m1 - 1].T
    b = np.nan_to_num(np.where(last >= 0, bfull[last.clip(0), np.arange(cols.size)[None, :]], 0.0))
    hb = -(held * b).sum(axis=1, keepdims=True)
    held2 = np.concatenate([held, hb], axis=1)
    c2 = np.r_[cols, ctx["bi"]]
    fw = sh["fwd_wx"][c2, m0:m1].T
    fi, fr = sh["fi"][c2, m0:m1].T, sh["fr"][c2, m0:m1].T
    assert held2.shape[1] == fw.shape[1] == fi.shape[1] == fr.shape[1], "held / fwd column count differs"
    gross = np.nansum(held2 * np.nan_to_num(fw), axis=1)
    fund = np.nansum(held2 * np.where(fi, fr, 0.0), axis=1)
    net = gross - fund                       # settle_multi: gross - 0.0 * turn - fund, turn finite
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sum(np.log1p(np.clip(net, -0.9, None))))


def sweep_costs(sels, ctx, halves=("A",)):
    """{combo: [cost0 log NAV per month of the halves]} for all 60 combos."""
    ms = [m for m in ctx["months"] if m["half"] in halves]
    if len(ms) != len(sels):
        raise ValueError("selection / month count differs")
    out = {}
    for h in S.HOLDS:
        for st in range(len(S.STATES)):
            keys, pairs = [], []
            for dr in range(len(S.DIRS)):
                out[(h, st, dr)] = [0.0] * len(ms)
                for k, (s, m) in enumerate(zip(sels, ms)):
                    d, c = S.compact_desired(s, ctx, m["m0"], m["m1"], st, dr)
                    if c.size:
                        keys.append((dr, k))
                        pairs.append((d, c))
            for (dr, k), (held, cols) in zip(keys, S.held_compact(pairs, h) if pairs else []):
                out[(h, st, dr)][k] = cost0_window(held, cols, ctx, ms[k]["m0"], ms[k]["m1"])
    return out


def scores(costs, kind, dd_max):
    """{combo: S.reward(...)} on cost0-only rows (t / t_dd read cost0 only; mean needs days and is not offered)."""
    if kind == "mean":
        raise ValueError("--combo-sweep scores t / t_dd (mean needs the days of every month)")
    return {c: S.reward([dict(cost0=x) for x in v], kind, dd_max) for c, v in costs.items()}


def pick(sc):
    """sel1_nullplant.cell_task's rule: highest finite score, ties to the first combo in COMBOS order."""
    best = None
    for c in COMBOS:
        s = sc[c]
        if np.isfinite(s) and (best is None or s > best[0]):
            best = (s, c)
    return best


def sweep(sels, ctx, kind, dd_max):
    """-> (combo, rows_lite of the combo through S.score_sel, score of those rows, sweep score, 60 scores) or None."""
    sc = scores(sweep_costs(sels, ctx), kind, dd_max)
    best = pick(sc)
    if best is None:
        return None
    s, c = best
    rows = S.rows_lite(S.score_sel(sels, ctx, *c))
    return c, rows, S.reward(rows, kind, dd_max), s, [sc[x] for x in COMBOS]


def sweep_task(task):
    """Worker: (body, sels, kind, dd_max) -> (body, sweep(...) without the 60 scores)."""
    body, sels, kind, dd_max = task
    r = sweep(sels, S._WS["view"], kind, dd_max)
    return body, (None if r is None else r[:4])
