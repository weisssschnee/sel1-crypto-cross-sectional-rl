"""XS_SEL1 two-plant world for SEL1_ARMS (supplement 10-15 evening). New file; sel1_common is imported, not modified.

  world   NULL + P1 + P2: the pcnull world of the chain (P1 = the plant formula, alpha1 = the chain's alpha) and a second
          plant on top, fwd[:, t] += alpha2 * z2[:, t-1], z2 = P2's TOP100 rank z as the NULL world sees it
          (S.plant_signal_shifted's lines with P2's body). Shard = no_winsor_shard(base, fwd = base.fwd + a1 add1 + a2 add2).
  spec    a json file (SEL1_ARMS_RULE.json's chosen p2): {"body": [...], "alpha": a2}.
    build(mode, device, vm, alpha, spec_path, log) -> ctx (as S.build; ctx["plant2"] = spec, ctx["zp2"] = z2)
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

import sel1_common as S
import m2h_dense_common as D
import v4_world as X


def shifted_rankz(vm, feats, univ, device, offsets, body):
    """S.plant_signal_shifted with an arbitrary body (same lines)."""
    import torch
    x = torch.from_numpy(np.ascontiguousarray(feats)).to(device=device, dtype=torch.float64)
    raw = vm.execute(list(body), x).detach().cpu().numpy()
    del x
    if offsets is not None:
        raw = np.stack([np.roll(raw[a], int(o)) for a, o in enumerate(offsets)])
    v = np.where(univ & np.isfinite(raw), raw, np.nan)
    rk = pd.DataFrame(v.T).rank(axis=1).to_numpy()
    n = np.isfinite(rk).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rk = rk - np.nanmean(rk, axis=1, keepdims=True)
        sd = np.nanstd(rk, axis=1, keepdims=True)
    s = np.where((n[:, None] >= D.PLANT_MIN_COINS) & (sd > 0), rk / np.where(sd > 0, sd, 1.0), np.nan)
    return s.T


def load_feats():
    w = X.load(X.GRID)
    X.world(w, feats=True)
    return w.pop("feats"), w["univ"]


def with_plant2(ctx, z2, alpha2):
    """ctx (built with keep_base=True, pcnull) -> a copy whose shard carries both plants."""
    w = dict(ctx)
    a1 = float(ctx["alpha"])
    fwd = ctx["base"]["fwd"] + a1 * S.plant_add(ctx["zp"]) + float(alpha2) * S.plant_add(z2)
    w["shard"] = S.no_winsor_shard(ctx["base"], ctx["dead"], fwd=fwd)
    w["zp2"] = z2
    return w


def build(mode, device, vm, alpha, spec_path, log=print):
    if mode != "pcnull":
        raise SystemExit("--plant2 is for --mode pcnull (the SEL1_ARMS world)")
    spec = json.load(open(spec_path, encoding="utf-8"))
    spec = spec.get("p2", spec)
    ctx = S.build(mode, device, vm, alpha=alpha, log=log, keep_base=True)
    feats, univ = load_feats()
    z2 = shifted_rankz(vm, feats, univ, device, ctx["offsets"], tuple(spec["body"]))
    del feats
    w = with_plant2(ctx, z2, spec["alpha"])
    w.pop("base", None)
    w["plant2"] = dict(body=list(spec["body"]), alpha=float(spec["alpha"]))
    log("SEL1 world + P2 %s alpha2 %.4e" % (vm.describe(list(spec["body"]), list(S.M.TERMINALS)), float(spec["alpha"])))
    return w
