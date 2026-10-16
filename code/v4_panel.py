"""V4 panel (order V4_COMBO_V1_CR1 s1 / s2.1; contract XS_BASIN_TIME_CONTRACT_V4_20261013). CPU, no OOS.

TRAIN long panel 2020-11..2025-12 (TRAIN 2021-01..2025-12; 2020-11/12 lookback warm-up only) from
  BACKFILL V3  2020-11..2023-12, 258 coins incl. the ones that died before 2024
  TRAIN 24-25  12 of its 54 columns (the 11 PANEL_COLS + label), the 489 coins alive in 2026-06
  SEAM V2      2024-01..2025-12 rows of the 275 coins the 2026-06 survivor panel dropped
DEV grid = DEV2026_HOURLY_PANEL_V1 (498 coins, 2025-06-01.., all its cells kept) + the SEAM coins it
lacks (SEAM rows inside the same hours), >= 672 h lookback before 2026-01-01.

Every source is sha-checked, footer-checked (C.guard_no_oos) and read with timestamp < 2026-07-01;
no DEV row enters the TRAIN files. Label: forward_trade_return_1h is recomputed from the stitched
close (C.forward_returns, the ruler's own fwd) and compared with each source's label where both
are finite. Renamed coins are not stitched (old and new symbols stay separate rows). 2021 OI / LS
are left as delivered (about 10 % present), never filled.

Outputs in OUT_DIR, refused if any exists:
  V4_TRAIN2020NOV_2025DEC_HOURLY12_LONG.parquet, V4_TRAIN_GRID.npz, V4_DEV2026_HOURLY_PANEL.npz,
  V4_DELIST.json (coins that stop before the DEV end: last real hour L, quote volume > 0),
  V4_PANEL_CHECKS.json, V4_PANEL_MANIFEST.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem"
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(ROOT, "runtime"), os.path.join(ROOT, "src"), r"D:\ChengboRemote\runtime\m1_stage1", HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)                      # -I drops cwd and PYTHONPATH
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import m1_common as C
import m2h_common as M

RTD = os.path.join(ROOT, "runtime")
SRC = dict(BF3=(os.path.join(RTD, r"train2021_2023_panel_v1\BACKFILL2020NOV_2023DEC_HOURLY11_LONG_V3.parquet"),
                "2e9b08a8e2a0bac5"),
           TR=(os.path.join(RTD, r"train2024_2025_panel_v1\TRAIN2024JAN_2025DEC_HOURLY54_TARGET_LONG_V1.parquet"),
               None),
           SM2=(os.path.join(RTD, r"xs_seam2024_2026_panel_v1\SEAM2024JAN_2026JUN_HOURLY11_LONG_V2.parquet"),
                "9b7d14424ca63cdf"))
DEV_V1 = os.path.join(ROOT, "artifacts", "m2h_v1", "DEV2026_HOURLY_PANEL_V1.npz")
OUT_DIR = os.path.join(RTD, "xs_v4_panel_v1")
LONG = os.path.join(OUT_DIR, "V4_TRAIN2020NOV_2025DEC_HOURLY12_LONG.parquet")
GRID = os.path.join(OUT_DIR, "V4_TRAIN_GRID.npz")
DEVZ = os.path.join(OUT_DIR, "V4_DEV2026_HOURLY_PANEL.npz")
DELIST = os.path.join(OUT_DIR, "V4_DELIST.json")
CHECKS = os.path.join(OUT_DIR, "V4_PANEL_CHECKS.json")
MANIFEST = os.path.join(OUT_DIR, "V4_PANEL_MANIFEST.json")
LABEL = "forward_trade_return_1h"
PCOLS = list(M.PANEL_COLS)
COLS = PCOLS + [LABEL]
FIRST = pd.Timestamp("2020-11-01 00:00")
Y2024 = pd.Timestamp("2024-01-01 00:00")
TRAIN_END = pd.Timestamp("2025-12-31 23:00")
DEV_FIRST = pd.Timestamp("2026-01-01 00:00")
OOS = pd.Timestamp("2026-07-01 00:00")
UNIV_N, UNIV_WIN, UNIV_MIN = M.UNIV_N, M.UNIV_WIN, M.UNIV_MIN
ALIVE_SLACK_H = 24


def log(m):
    print(m, flush=True)


def sha_file(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()[:n]


def read(path, cols, lo=None, hi=OOS):
    C.guard_no_oos(path)
    flt = [("timestamp", "<", hi.to_pydatetime())]
    if lo is not None:
        flt.append(("timestamp", ">=", lo.to_pydatetime()))
    t = pq.read_table(path, columns=["symbol", "timestamp"] + cols, filters=flt).to_pandas()
    if getattr(t.timestamp.dt, "tz", None) is not None:
        t["timestamp"] = t.timestamp.dt.tz_convert("UTC").dt.tz_localize(None)
    t["timestamp"] = t.timestamp.astype("datetime64[ns]")
    t["symbol"] = t.symbol.astype(str)
    for c in cols:
        t[c] = t[c].astype(np.float64)
    if len(t) and t.timestamp.max() >= OOS:
        raise SystemExit("OOS row in %s" % path)
    return t


def same(a, b):
    """Cell-wise equality with NaN == NaN."""
    return (a == b) | (np.isnan(a) & np.isnan(b))


def univ_alive(qv, alive):
    """Point-in-time TOP100: trailing 720 h quote volume (>= 360 h), ranked only among coins with a
    bar at that hour (a coin that has stopped trading cannot hold a slot)."""
    vol = pd.DataFrame(qv.T).rolling(UNIV_WIN, min_periods=UNIV_MIN).sum().to_numpy().T
    vol = np.where(alive, vol, np.nan)
    return (pd.DataFrame(vol.T).rank(axis=1, ascending=False, method="first") <= UNIV_N).to_numpy().T


def last_real(qv, present):
    """Per coin: index of the last hour with quote volume > 0 (-1 if none)."""
    real = present & (np.nan_to_num(qv) > 0)
    T = real.shape[1]
    idx = np.where(real, np.arange(T)[None, :], -1)
    return idx.max(axis=1)


def alive_mask(present, lidx):
    """A coin is a TOP100 candidate from its first row to its last real hour L (gaps inside kept)."""
    T = present.shape[1]
    has = present.any(axis=1)
    first = np.where(has, present.argmax(axis=1), T)
    h = np.arange(T)[None, :]
    return (h >= first[:, None]) & (h <= lidx[:, None])


def delist_table(syms, ts, present, lidx, end_idx):
    """Coins whose last real hour L (quote volume > 0) is > 24 h before the grid end: the weight
    layer liquidates at L. Covers truncated coins and any zero-volume zombie tail left in a source."""
    last_row = np.where(present.any(axis=1), present.shape[1] - 1 - present[:, ::-1].argmax(axis=1), -1)
    rows = []
    for a in np.flatnonzero((lidx >= 0) & (lidx < end_idx - ALIVE_SLACK_H)):
        rows.append(dict(symbol=syms[a], L=str(ts[lidx[a]]), L_idx=int(lidx[a]), last_row=str(ts[last_row[a]])))
    return rows


def grid_of(t, syms, ts):
    s, h, g = M.long_to_grids(t, COLS, syms, ts)
    si = pd.Index(syms).get_indexer(t.symbol)
    ti = ((t.timestamp - ts[0]) / pd.Timedelta(hours=1)).to_numpy().astype(int)
    pres = np.zeros((len(syms), len(ts)), dtype=bool)
    pres[si, ti] = True
    return g, pres


def coverage(g, pres, ts, univ=None):
    yrs = np.asarray(ts.year)
    out = {}
    for y in sorted(set(yrs)):
        m = yrs == y
        p = pres[:, m]
        n = max(int(p.sum()), 1)
        out[str(y)] = dict(coins=int(p.any(axis=1).sum()), rows=int(p.sum()),
                           **{c: round(float((np.isfinite(g[c][:, m]) & p).sum() / n), 4) for c in COLS})
        if univ is not None:
            u = univ[:, m]
            out[str(y)]["top100_mean"] = round(float(u.sum(axis=0).mean()), 1)
            out[str(y)]["top100_min"] = int(u.sum(axis=0).min())
    return out


def main():
    t0 = time.time()
    for p in (LONG, GRID, DEVZ, DELIST, CHECKS, MANIFEST):
        if os.path.exists(p):
            raise SystemExit("%s exists; not overwritten" % p)
    os.makedirs(OUT_DIR, exist_ok=True)
    shas = {k: sha_file(p) for k, (p, _) in SRC.items()}
    for k, (p, want) in SRC.items():
        if want and shas[k] != want:
            raise SystemExit("%s sha16 %s != %s" % (k, shas[k], want))
    shas["DEV_V1"] = sha_file(DEV_V1)
    log("sha %s %.0fs" % (shas, time.time() - t0))
    bf = read(SRC["BF3"][0], COLS, hi=Y2024)
    tr = read(SRC["TR"][0], COLS, hi=DEV_FIRST)
    sm = read(SRC["SM2"][0], COLS, hi=OOS)
    sm_tr, sm_dev = sm[sm.timestamp < DEV_FIRST], sm[sm.timestamp >= DEV_FIRST - pd.Timedelta(days=214)]
    ck = {}
    dup = {k: int(v.duplicated(["symbol", "timestamp"]).sum()) for k, v in (("BF3", bf), ("TR", tr), ("SM2", sm))}
    ck["sources"] = dict(rows=dict(BF3=len(bf), TR=len(tr), SM2_train=len(sm_tr)), dup=dup,
                         tr_x_sm2_symbols=len(set(tr.symbol) & set(sm.symbol)),
                         bf3_max=str(bf.timestamp.max()), tr_min=str(tr.timestamp.min()),
                         sm2_train_min=str(sm_tr.timestamp.min()))
    long = pd.concat([bf, tr, sm_tr], ignore_index=True)
    ck["dup_stitched"] = int(long.duplicated(["symbol", "timestamp"]).sum())
    if ck["dup_stitched"] or any(dup.values()) or ck["sources"]["tr_x_sm2_symbols"]:
        raise SystemExit("duplicate rows %s" % ck)
    syms = sorted(long.symbol.unique())
    ts = pd.date_range(FIRST, TRAIN_END, freq="h")
    g, pres = grid_of(long, syms, ts)
    del long, bf, sm_tr
    log("TRAIN grid %d x %d %.0fs" % (len(syms), len(ts), time.time() - t0))
    # label: the ruler's own fwd from the stitched close, checked against each source's label
    fwd = C.forward_returns(g["trade_close"], None)
    lab = g[LABEL]
    both = np.isfinite(fwd) & np.isfinite(lab)
    dif = np.abs(fwd - lab)[both]
    ck["label"] = dict(both=int(both.sum()), maxdiff=float(dif.max()) if dif.size else 0.0,
                       n_gt_1e9=int((dif > 1e-9).sum()), src_only=int((np.isfinite(lab) & ~np.isfinite(fwd)).sum()),
                       fwd_only=int((np.isfinite(fwd) & ~np.isfinite(lab)).sum()))
    g[LABEL] = fwd
    # last row of every coin: fwd NaN
    last_row = present_last = pres.shape[1] - 1 - pres[:, ::-1].argmax(axis=1)
    ck["last_row_fwd_nan"] = dict(coins=len(syms), finite=int(np.isfinite(fwd[np.arange(len(syms)), last_row]).sum()))
    # overlap with the old TRAIN (2024-25 survivors): every cell identical, fwd included
    os_, ots, og = M.load_train_grids(SRC["TR"][0])
    ri = pd.Index(syms).get_indexer(os_)
    ci = ts.get_indexer(ots)
    if (ri < 0).any() or (ci < 0).any():
        raise SystemExit("old TRAIN cell outside the V4 grid")
    ofwd = C.forward_returns(og["trade_close"], None)
    mism = {c: int((~same(g[c][np.ix_(ri, ci)], og[c])).sum()) for c in PCOLS}
    mism["fwd"] = int((~same(fwd[np.ix_(ri, ci)], ofwd)).sum())
    ck["overlap_old_train"] = dict(coins=len(os_), hours=len(ots), mismatches=mism)
    del og, ofwd
    lidx = last_real(g["trade_quote_volume"], pres)
    alive = alive_mask(pres, lidx)
    univ = univ_alive(g["trade_quote_volume"], alive)
    dl = delist_table(syms, ts, pres, lidx, len(ts) - 1)
    ck["coverage_by_year"] = coverage(g, pres, ts, univ)
    log("TRAIN checks %.0fs" % (time.time() - t0))
    # ---- DEV grid: DEV V1 cells kept, SEAM coins it lacks added on the same hours
    z = np.load(DEV_V1, allow_pickle=False)
    dh = pd.DatetimeIndex(z["hours"].astype("datetime64[ns]"))
    if dh.max() >= OOS:
        raise SystemExit("DEV V1 reaches OOS")
    dsyms = [str(x) for x in z["symbols"]]
    smd = sm_dev[(sm_dev.timestamp >= dh[0]) & (sm_dev.timestamp <= dh[-1])]
    add = sorted(set(smd.symbol) - set(dsyms))
    vsyms = dsyms + add
    sg, spres = grid_of(smd, vsyms, dh)
    dg = {}
    for c in PCOLS:
        a = np.full((len(vsyms), len(dh)), np.nan)
        a[:len(dsyms)] = z[c]
        a[len(dsyms):] = sg[c][len(dsyms):]
        dg[c] = a
    ov = [i for i, s in enumerate(dsyms) if s in set(smd.symbol)]
    both = {c: np.isfinite(dg[c][ov]) & np.isfinite(sg[c][ov]) for c in PCOLS}
    ck["dev"] = dict(v1_coins=len(dsyms), seam_added=len(add), v1_and_seam=len(ov),
                     v1_and_seam_mismatch={c: int((both[c] & (dg[c][ov] != sg[c][ov])).sum()) for c in PCOLS},
                     hours=[str(dh[0]), str(dh[-1])], lookback_h_before_2026=int(dh.get_loc(DEV_FIRST)),
                     v1_and_seam_symbols=[dsyms[i] for i in ov])
    mm = {}
    for c in PCOLS:
        a_, b_ = dg[c][ov], sg[c][ov]
        bad = both[c] & (a_ != b_)
        if bad.any():
            ad = np.abs(a_[bad] - b_[bad])
            hrs = np.flatnonzero(bad.any(axis=0))
            mm[c] = dict(frac=round(float(bad.sum() / max(int(both[c].sum()), 1)), 4), abs_med=float(np.median(ad)),
                         rel_med=float(np.median(ad / np.maximum(np.abs(b_[bad]), 1e-12))),
                         coins=int(bad.any(axis=1).sum()), first=str(dh[hrs[0]]), last=str(dh[hrs[-1]]),
                         hours_2026=int((dh[hrs] >= DEV_FIRST).sum()))
    ck["dev"]["v1_and_seam_mismatch_detail"] = mm
    dpres = np.isfinite(dg["trade_close"])
    dfwd = C.forward_returns(dg["trade_close"], None)
    ck["dev"]["v1_fwd_mismatch"] = int((~same(dfwd[:len(dsyms)], z["fwd"])).sum())
    dlidx = last_real(dg["trade_quote_volume"], dpres)
    dalive = alive_mask(dpres, dlidx)
    duniv = univ_alive(dg["trade_quote_volume"], dalive)
    ddl = delist_table(vsyms, dh, dpres, dlidx, len(dh) - 1)
    d26 = np.asarray(dh >= DEV_FIRST)
    ck["dev"]["coverage_2026"] = dict(coins=int(dpres[:, d26].any(axis=1).sum()),
                                      top100_mean=round(float(duniv[:, d26].sum(axis=0).mean()), 1),
                                      top100_from_seam=round(float(duniv[len(dsyms):, d26].sum(axis=0).mean()), 2),
                                      delisted_2026=int(sum(1 for r in ddl if r["L"] >= str(DEV_FIRST))))
    ck["dev"]["last_row_fwd_nan"] = int(np.isfinite(dfwd[np.arange(len(vsyms)),
                                                         len(dh) - 1 - dpres[:, ::-1].argmax(axis=1)]).sum()) == 0
    # ---- gates (structural; label agreement and coverage are reported)
    ok = dict(no_dup=ck["dup_stitched"] == 0, overlap_identical=sum(mism.values()) == 0,
              last_row_fwd_nan=ck["last_row_fwd_nan"]["finite"] == 0, dev_last_row_fwd_nan=ck["dev"]["last_row_fwd_nan"],
              # settlement columns must agree; OI / LS / premium differ for the 8 coins both DEV V1 and
              # SEAM carry (metrics convention of the DEV cache), V1 cells are kept, reported only
              dev_overlap_settle_identical=sum(ck["dev"]["v1_and_seam_mismatch"][c] for c in
                                               ("trade_close", "trade_quote_volume", "funding_rate")) == 0
              and ck["dev"]["v1_fwd_mismatch"] == 0,
              dev_lookback=ck["dev"]["lookback_h_before_2026"] >= 672, no_oos=bool(dh.max() < OOS))
    ck["gates"] = ok
    ck["all_pass"] = bool(all(ok.values()))
    # ---- write
    ii, jj = np.nonzero(pres)
    out = pd.DataFrame({"symbol": np.asarray(syms)[ii], "timestamp": ts[jj]})
    for c in COLS:
        out[c] = g[c][ii, jj]
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), LONG, row_group_size=1 << 20)
    del out
    np.savez(GRID, symbols=np.asarray(syms), hours=ts.values, present=pres, alive=alive, univ=univ,
             last_real_idx=lidx, **{c: g[c] for c in COLS})
    np.savez(DEVZ, symbols=np.asarray(vsyms), hours=dh.values, fwd=dfwd, present=dpres, alive=dalive,
             univ=duniv, last_real_idx=dlidx, **dg)
    json.dump(dict(train=dl, dev=ddl), open(DELIST, "w", encoding="utf-8"), indent=1)
    json.dump(ck, open(CHECKS, "w", encoding="utf-8"), indent=1, default=float)
    man = dict(schema="V4_PANEL_MANIFEST_V1", contract="XS_BASIN_TIME_CONTRACT_V4_20261013",
               code_sha16=sha_file(os.path.abspath(__file__)), sources={k: dict(path=p, sha16=shas[k])
                                                                     for k, (p, _) in SRC.items()},
               dev_v1=dict(path=DEV_V1, sha16=shas["DEV_V1"]),
               train=dict(coins=len(syms), hours=len(ts), first=str(ts[0]), last=str(ts[-1]), rows=int(pres.sum()),
                          delisted_inside=len(dl), long_sha16=sha_file(LONG), grid_sha16=sha_file(GRID)),
               dev=dict(coins=len(vsyms), hours=len(dh), first=str(dh[0]), last=str(dh[-1]),
                        delisted_inside=len(ddl), sha16=sha_file(DEVZ)),
               universe="point-in-time TOP100 by trailing %d h quote volume (>= %d h), ranked among coins alive "
                        "at that hour (first row .. last hour with quote volume > 0)" % (UNIV_WIN, UNIV_MIN),
               all_pass=ck["all_pass"], seconds=round(time.time() - t0, 1),
               written=time.strftime("%Y-%m-%dT%H:%M:%S%z"), dev_read_for_stats=False, oos_read=False)
    json.dump(man, open(MANIFEST, "w", encoding="utf-8"), indent=1)
    log("V4 panel all_pass %s gates %s train %dx%d delist %d dev %d coins (+%d seam) %.0fs"
        % (ck["all_pass"], ok, len(syms), len(ts), len(dl), len(vsyms), len(add), time.time() - t0))
    sys.exit(0 if ck["all_pass"] else 13)


if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    main()
