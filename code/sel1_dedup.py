"""XS_SEL1 behaviour identity + family z with the rounding fix (supplement 10-15 afternoon). New file; sel1_common and
m2h_dense_common are imported, never modified.

  behaviour_hash(sels)  sels = S.select_months(..., ("A",)) / S.vm_select*(...): per A month the (hi, lo, n) arrays.
                        A row trades only when n >= S.MIN_COINS (compact_desired's rule), so a row below that is
                        written as all -1; the K long / K short coins of a trading row are sorted (sets, not ranks).
                        Two formulas with the same hash hold the same book under every one of the 60 combos (the
                        state switch reads the BTC state, not the formula), so they are one behaviour.
  zfam(r, valid, groups) D.zfam, except a family whose valid std < STD_EPS gets advantage 0 for all its valid rows
                        (numpy's std of identical values can come out ~1e-16, and /1e-16 then clips to +-3 noise).
"""
from __future__ import annotations

import hashlib

import numpy as np

import sel1_common as S
import m2h_dense_common as D

STD_EPS = 1e-12


def behaviour_hash(sels):
    h = hashlib.blake2b(digest_size=16)
    for hi, lo, n in sels:
        on = np.asarray(n) >= S.MIN_COINS
        a = np.where(on[:, None], np.sort(np.asarray(hi, np.int16), axis=1), -1).astype(np.int16)
        b = np.where(on[:, None], np.sort(np.asarray(lo, np.int16), axis=1), -1).astype(np.int16)
        h.update(np.int64(a.shape[0]).tobytes())
        h.update(np.ascontiguousarray(a).tobytes())
        h.update(np.ascontiguousarray(b).tobytes())
    return h.hexdigest()


def zfam(r, valid, groups, clip=D.ZCLIP):
    r = np.asarray(r, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    out = np.full(len(r), np.nan)
    for sl in D.group_slices(groups):
        m = valid[sl]
        if not m.any():
            continue
        v = r[sl][m]
        sd = float(v.std())
        z = np.zeros_like(v) if sd < STD_EPS else (v - v.mean()) / sd
        seg = np.full(m.shape, np.nan)
        seg[m] = np.clip(z, -clip, clip)
        out[sl] = seg
    return out
