"""BASIN3_H1_20261009 pre-launch self-checks (handoff section IV).

1  self-computed forward return is bit-equal to the panel's forward_trade_return_1h
2  a constant signal earns exactly 0
3  T1 reproduction: DELTA(log_close, 168), hold 168, 50 TRAIN tiles == h1 T1_mom_168h|TOP100|lag1
4  T2 reproduction: NEG(TS_MEAN(funding_level, 168)), hold 168 == h1 T2_neg_funding_7d|TOP100|lag1
5  1000 formulas per family reference no terminal outside their own family
Each check prints ONE line: "CHK<n> PASS|FAIL ...". Any FAIL means: stop and report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem"
PANEL = os.path.join(ROOT, r"runtime\train2024_2025_panel_v1\TRAIN2024JAN_2025DEC_HOURLY54_TARGET_LONG_V1.parquet")
H1 = r"D:\ChengboRemote\runtime\strategy_survey_v1\h1_handwritten_baselines_TRAIN.json"

for p in (r"D:\ChengboRemote\runtime\m1_stage1", r"D:\ChengboRemote\runtime\basin3_h1",
          os.path.join(ROOT, "src"), os.path.join(ROOT, "runtime")):
    sys.path.insert(0, p)

import m1_common as C
import m1_ruler as R
import m2h_common as M
from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import (
    XsTsCsStackVM, OPERATOR_NAMES, WINDOWED_OPERATORS)
from okx_alpha_discovery.adapters.xs_tscs_sampler_v2 import build_legality_table, sample_batch
from okx_alpha_discovery.evaluation import stateful_action_policy as policy_mod

T1_TARGET = 7.8633399064      # h1 T1_mom_168h|TOP100|lag1, hold 168, mean over 50 windows
T2_TARGET = -7.4340437864     # h1 T2_neg_funding_7d|TOP100|lag1, hold 168
TOL = 1e-6
H1_LABEL = dict(bit_equal=5218776, both=5218776, nan_mismatch=8)


def bp_per_day(log_nav):
    return 1e4 * float(log_nav) / (M.EPISODE / 24.0)


def load_world(device, dtype):
    syms, ts, grids = M.load_train_grids(PANEL)
    world = M.build_world(grids)
    feats_pad = M.pad_feats(world["feats"])
    starts = M.tile_starts(world["feats"].shape[2])
    return syms, grids, world, feats_pad, starts


def make_vm():
    import okx_alpha_discovery.adapters.xs_tscs_vm_v2 as vm_mod
    vm_mod.BLOCK_ELEMS = 1 << 25
    vm_mod.EMA_KERNEL_CACHE_MAX = 1
    return XsTsCsStackVM(len(M.TERMINALS), windows=M.WINDOWS)


def program(vm, term, op_name, window, negate=False):
    """RPN body: terminal, window selector, operator [, NEG], END."""
    toks = [int(term), vm.window_base + list(M.WINDOWS).index(window),
            vm.operator_base + list(OPERATOR_NAMES).index(op_name)]
    if negate:
        toks.append(vm.operator_base + list(OPERATOR_NAMES).index("NEG"))
    toks.append(vm.end_token)
    return toks


class _Stub(torch.nn.Module):
    """Uniform policy: isolates the grammar (check 5) from the trained policy."""

    def __init__(self, v, device):
        super().__init__()
        self.v = v
        self.device = device

    def forward(self, inp):
        return torch.zeros(inp.shape[0], self.v, device=self.device), None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    ap.add_argument("--tiles", type=int, default=0, help="0 = all 50 TRAIN tiles")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    want = set(args.only.split(",")) if args.only else {"1", "2", "3", "4", "5"}
    t_all = time.time()

    syms, grids, world, feats_pad, starts = load_world(args.device, dtype)
    univ, shard = world["univ"], world["shard"]
    if args.tiles:
        starts = starts[:args.tiles]

    # ---- 1 forward returns -------------------------------------------------
    if "1" in want:
        fwd = C.forward_returns(grids["trade_close"], None)
        pf = grids["forward_trade_return_1h"]
        both = np.isfinite(fwd) & np.isfinite(pf)
        eq = int((fwd[both] == pf[both]).sum())
        mism = int((np.isfinite(fwd) != np.isfinite(pf)).sum())
        ok = eq == H1_LABEL["bit_equal"] and mism == H1_LABEL["nan_mismatch"]
        print("CHK1 %s bit_equal=%d/%d nan_mismatch=%d (h1: %d/%d, %d)"
              % ("PASS" if ok else "FAIL", eq, int(both.sum()), mism,
                 H1_LABEL["bit_equal"], H1_LABEL["both"], H1_LABEL["nan_mismatch"]), flush=True)

    # ---- 2 constant signal -------------------------------------------------
    if "2" in want:
        s0 = starts[0]
        E, A = M.EPISODE, world["feats"].shape[0]
        worst = 0.0
        for const in (1.0, -3.5, 0.0):
            sigT = np.full((E, A), const)
            sc = R.score_window(sigT, shard, s0, E, M.HOLDS, policy_mod,
                                costs=M.COSTS, variants=("win",), band=R.BAND, nav_usd=R.NAV_USD)
            for h in M.HOLDS:
                lg, p = sc[h]["win"]
                worst = max(worst, abs(lg["flat6"]), abs(lg["cost0"]), p["turnover"])
        print("CHK2 %s max|net|=%.3e over holds %s (constant signal must trade nothing)"
              % ("PASS" if worst == 0.0 else "FAIL", worst, list(M.HOLDS)), flush=True)

    # ---- 3 / 4 T1 and T2 reproduction --------------------------------------
    if "3" in want or "4" in want:
        vm = make_vm()
        device = torch.device(args.device)
        slices = M.tile_slices(feats_pad, starts, device, dtype=dtype)
        cases = []
        if "3" in want:
            cases.append(("T1", program(vm, 0, "DELTA", 168), T1_TARGET))
        if "4" in want:
            cases.append(("T2", program(vm, 4, "TS_MEAN", 168, negate=True), T2_TARGET))
        for name, body, target in cases:
            t0 = time.time()
            res = M.score_formula(vm, body, slices, starts, univ, shard, [168], policy_mod,
                                  costs=M.COSTS, band=R.BAND, nav_usd=R.NAV_USD)
            arr = np.asarray(res["per_hold"][168]["flat6"], dtype=np.float64)
            mine = float(np.mean([bp_per_day(x) for x in arr]))
            d = abs(mine - target)
            # h1's own numbers, for a per-tile diff count when the mean disagrees
            h1 = json.load(open(H1, encoding="utf-8"))
            key = {"T1": "T1_mom_168h|TOP100|lag1", "T2": "T2_neg_funding_7d|TOP100|lag1"}[name]
            hv = np.asarray(h1["results"][key]["168"]["net"], dtype=np.float64)[:len(arr)]
            mine_bp = np.asarray([bp_per_day(x) for x in arr])
            n_diff = int((np.abs(mine_bp - hv) > 1e-6).sum())
            if n_diff:
                j = int(np.argmax(np.abs(mine_bp - hv)))
                print("     worst tile %d mine=%.6f h1=%.6f" % (j, mine_bp[j], hv[j]), flush=True)
            print("CHK%s %s mine=%.10f h1=%.10f d=%.3e tiles=%d n_diff=%d live=%.3f %.0fs"
                  % (name, "PASS" if d <= TOL else "FAIL", mine, target, d, len(arr),
                     n_diff, res["live_frac"], time.time() - t0), flush=True)
            print("     expr=%s" % vm.describe(body, list(M.TERMINALS)), flush=True)
            if dtype == torch.float64:
                sl32 = M.tile_slices(feats_pad, starts, device, dtype=torch.float32)
                r32 = M.score_formula(vm, body, sl32, starts, univ, shard, [168], policy_mod,
                                      costs=M.COSTS, band=R.BAND, nav_usd=R.NAV_USD)
                m32 = float(np.mean([bp_per_day(x) for x in r32["per_hold"][168]["flat6"]]))
                print("     train-dtype float32 mean=%.6f d_vs_h1=%.3e (info, not the gate)"
                      % (m32, abs(m32 - target)), flush=True)
                del sl32

    # ---- 5 basin grammar, through the trainer's own sampler and a fresh trainer model ----
    if "5" in want:
        import m2h_trainer as TR
        from train_basin_riskpg_v1 import make_basin_model, apply_output_bias
        import model_core.alphagpt as alphagpt_mod
        from model_core.config import ModelConfig
        from model_core.vocab import FormulaVocab
        vm = make_vm()
        device = torch.device(args.device)
        ModelConfig.DEVICE = device
        torch.manual_seed(20261009)
        op_names = (list(OPERATOR_NAMES) + [f"W{w}" for w in M.WINDOWS] + ["END"] + M.HOLD_NAMES
                    + list(M.BASIN_TOKEN_NAMES))
        model, _ = make_basin_model(alphagpt_mod=alphagpt_mod, ModelConfig=ModelConfig,
                                    FormulaVocab=FormulaVocab, feature_names=list(M.TERMINALS),
                                    operator_token_names=op_names, max_len=M.MAX_PROGRAM_LEN,
                                    device=device)
        apply_output_bias(model, len(M.TERMINALS), len(OPERATOR_NAMES), len(M.WINDOWS),
                          TR.OP_BIAS, TR.END_BIAS)
        bad_total = 0
        for b in M.BASIN_NAMES:
            terms = set(M.basin_terms(b))
            tab = build_legality_table(vm, max_len=M.MAX_PROGRAM_LEN,
                                       available_terminals=M.basin_terms(b),
                                       available_windows=M.WINDOWS).to(device)
            with torch.no_grad():
                sb = TR.sample_batch_with_hold_basins(model, vm, [tab], [1000], device=device,
                                                      basins=(b,))
            bad = 0
            first_bad = 0
            for toks in sb.formulas:
                toks = list(TR.strip_hold(vm, list(toks)))
                for t in toks:
                    if t < vm.feature_count and t not in terms:
                        bad += 1
                if not (toks and toks[0] < vm.feature_count and toks[0] in terms):
                    first_bad += 1
            bad_total += bad + first_bad
            print("CHK5 %s %s cross_family_refs=%d first_token_outside=%d of 1000"
                  % (b, "PASS" if (bad == 0 and first_bad == 0) else "FAIL", bad, first_bad),
                  flush=True)
        print("CHK5 %s total_violations=%d" % ("PASS" if bad_total == 0 else "FAIL", bad_total),
              flush=True)

    print("selftest %.0fs device=%s dtype=%s" % (time.time() - t_all, args.device, args.dtype),
          flush=True)


if __name__ == "__main__":
    main()
