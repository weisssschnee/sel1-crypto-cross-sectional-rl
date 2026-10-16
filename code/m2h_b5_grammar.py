"""BASIN5 step 2 grammar: no bare levels (BASIN5_ORDER s2). New file; m2h_dense_common, the
ruler and the VM are not modified.

Rule. The level terminals (log_close, log_high, log_low, log_quote_volume, funding_level,
log_oi_value) may reach a CS operator (CS_DEMEAN / CS_ZSCORE / CS_RANK) or the formula root only
through a de-levelling time-series step:
  DELTA (the order's TS_DELTA), TS_RANK, TS_ZSCORE,
  SUB / DIV of a subtree with its own EMA, either order (x - EMA(x, w), EMA(x, w) / x, ...).
Every other operator keeps the level if ANY argument is a level (ADD/SUB/MUL/DIV, NEG, ABS, SIGN,
LAG, TS_MEAN/MAX/MIN/STD/SLOPE/CORR, EMA, GATE including its condition, PER_ASSET_THRESH). So a
level terminal anywhere under a CS operator or the root, without one of those steps on its path,
makes the formula invalid. TS_SLOPE / TS_STD / TS_CORR are not counted as de-levelling: the order
names the three operators and the own-EMA ratio / difference only (literal reading).

    check(vm, body)  -> None if legal, else a reason string starting with LEVEL_
    selftest(vm)     -> dict(ok=..., cases=[...]); the order's five cases plus extras
"""
from __future__ import annotations

import os
import sys
import time

_ROOT = r"D:\riskpg_stateful_v5s_governor_nextgen_beem"
for _p in (os.path.dirname(os.path.abspath(__file__)), r"D:\ChengboRemote\runtime\m1_stage1",
           os.path.join(_ROOT, "src"), os.path.join(_ROOT, "runtime")):
    if _p not in sys.path:               # -I drops the script dir; same paths the trainer adds
        sys.path.append(_p)

import m2h_common as M

LEVEL_TERMS = frozenset(("log_close", "log_high", "log_low", "log_quote_volume",
                         "funding_level", "log_oi_value"))
DELEVEL_OPS = frozenset(("DELTA", "TS_RANK", "TS_ZSCORE"))
OWN_EMA_OPS = frozenset(("SUB", "DIV"))
CS_OPS = frozenset(("CS_DEMEAN", "CS_ZSCORE", "CS_RANK"))


def check(vm, body):
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import OPERATOR_ARITIES, WINDOWED_OPERATORS
    arity = dict(OPERATOR_ARITIES)
    stack = []          # per value: (is_level, token key of the subtree, key of x if this is EMA(x))
    try:
        for t in body:
            t = int(t)
            if vm.is_end(t):
                break
            if vm.is_terminal(t):
                stack.append((M.TERMINALS[t] in LEVEL_TERMS, (t,), None))
            elif vm.is_window(t):
                stack.append((False, (t,), None))
            else:
                name = vm.operator_name(t)
                w = stack.pop() if name in WINDOWED_OPERATORS else None
                args = [stack.pop() for _ in range(arity[name])][::-1]
                key = sum((a[1] for a in args), ()) + (w[1] if w else ()) + (t,)
                lv = any(a[0] for a in args)
                if name in CS_OPS and lv:
                    return "LEVEL_INTO_" + name
                if name in DELEVEL_OPS:
                    lv = False
                elif name in OWN_EMA_OPS and (args[1][2] == args[0][1] or args[0][2] == args[1][1]):
                    lv = False
                stack.append((lv, key, args[0][1] if name == "EMA" else None))
    except (IndexError, KeyError):
        return "LEVEL_MALFORMED"
    if len(stack) != 1:
        return "LEVEL_MALFORMED"
    return "LEVEL_AT_ROOT" if stack[0][0] else None


def selftest(vm):
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import OPERATOR_NAMES
    import m2h_dense_common as D
    T = lambda n: M.TERMINALS.index(n)
    O = lambda n: vm.operator_base + list(OPERATOR_NAMES).index(n)
    W = lambda h: vm.window_base + list(M.WINDOWS).index(h)
    E = vm.end_token
    c, h, lo, qv, fl = T("log_close"), T("log_high"), T("log_low"), T("log_quote_volume"), T("funding_level")
    cases = [
        # the order's five
        ("order", True, [c, W(168), O("DELTA"), E]),
        ("order", True, list(D.plant_body(vm))),
        ("order", False, [lo, E]),
        ("order", False, [qv, O("CS_RANK"), E]),
        ("order", False, [h, O("SIGN"), E]),
        # extras: own-EMA ratio / difference, CS after a de-level step, levels kept by other ops
        ("extra", True, [c, c, W(168), O("EMA"), O("SUB"), E]),
        ("extra", True, [qv, W(24), O("EMA"), qv, O("DIV"), E]),
        ("extra", True, [qv, W(168), O("TS_ZSCORE"), O("CS_RANK"), E]),
        ("extra", True, [c, W(24), O("TS_RANK"), h, W(24), O("DELTA"), O("ADD"), O("CS_ZSCORE"), E]),
        ("extra", False, [c, W(168), O("TS_MEAN"), O("CS_RANK"), E]),
        ("extra", False, [c, h, W(168), O("EMA"), O("SUB"), E]),
        ("extra", False, [fl, W(168), O("TS_MEAN"), O("NEG"), E]),
        ("extra", False, [c, W(24), O("DELTA"), O("SIGN"), lo, c, W(24), O("DELTA"), O("GATE"), E]),
        # RGP_P8_V2's planted relative (REWARD_PICK target, corr 0.931): CS_DEMEAN on a level
        ("target", False, [qv, c, O("ADD"), O("ABS"), O("CS_DEMEAN"), W(336), O("TS_SLOPE"),
                           O("CS_RANK"), E]),
    ]
    out, ok = [], True
    for kind, legal, body in cases:
        r = check(vm, body)
        good = (r is None) == legal
        ok = ok and good
        out.append(dict(kind=kind, expr=vm.describe(body, list(M.TERMINALS)), want_legal=legal,
                        reason=r, pass_=good))
    return dict(ok=ok, cases=out)


if __name__ == "__main__":
    import json
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import XsTsCsStackVM
    res = selftest(XsTsCsStackVM(len(M.TERMINALS), windows=M.WINDOWS))
    bad = [x["expr"] for x in res["cases"] if not x["pass_"]]
    print("GRAMMAR %s %d/%d %s" % ("PASS" if res["ok"] else "FAIL", len(res["cases"]) - len(bad),
                                   len(res["cases"]), " | ".join(bad)))
    if len(sys.argv) > 1:                # optional JSON receipt path
        res["written"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        json.dump(res, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    sys.exit(0 if res["ok"] else 1)
