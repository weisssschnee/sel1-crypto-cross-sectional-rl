"""BASIN5 s6: one plain-language sentence per formula (ROUND_REPORT "收敛到了什么", FROZEN_B5_TOP5).

    explain(vm, body) -> str

Rule-based gloss of the RPN tree. The book ranks the root value per hour (ruler / conc book), so
root CS_RANK / CS_ZSCORE / CS_DEMEAN (monotone per row) do not change the book and are dropped
from the sentence; a root NEG flips long and short.
"""
from __future__ import annotations

import m2h_common as M

TERM = dict(log_close="收盘价", log_high="最高价", log_low="最低价", log_quote_volume="成交额",
            funding_level="资金费率", log_oi_value="持仓额", global_ls_account_ratio="散户多空账户比",
            taker_buy_share="主动买占比", premium_close_bps="合约溢价", mark_index_basis_bps="标记-指数基差",
            premium_high="小时内最高溢价")
TS = dict(DELTA="{x}过去{w}小时的变化", TS_RANK="{x}在自身过去{w}小时里的分位",
          TS_ZSCORE="{x}相对自身{w}小时均值的标准分", TS_MEAN="{x}的{w}小时均值",
          EMA="{x}的{w}小时均线", TS_STD="{x}的{w}小时波动", TS_MAX="{x}的{w}小时最高值",
          TS_MIN="{x}的{w}小时最低值", TS_SLOPE="{x}的{w}小时趋势斜率", LAG="{w}小时前的{x}")
CS = dict(CS_RANK="{x}的截面排名", CS_ZSCORE="{x}的截面标准分", CS_DEMEAN="{x}减去全池均值")
CS_MONO = frozenset(CS)
MAX_CHARS = 160


def tree(vm, body):
    from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import OPERATOR_ARITIES, WINDOWED_OPERATORS
    arity = dict(OPERATOR_ARITIES)
    st = []
    for t in body:
        t = int(t)
        if vm.is_end(t):
            break
        if vm.is_terminal(t):
            st.append(("T", M.TERMINALS[t]))
        elif vm.is_window(t):
            st.append(("W", vm.window_value(t)))
        else:
            name = vm.operator_name(t)
            w = st.pop()[1] if name in WINDOWED_OPERATORS else None
            args = [st.pop() for _ in range(arity[name])][::-1]
            st.append(("O", name, w, args))
    return st[-1]


def _same(a, b):
    return repr(a) == repr(b)


def gloss(n):
    if n[0] == "T":
        return TERM.get(n[1], n[1])
    _, name, w, a = n
    if name in ("SUB", "DIV") and len(a) == 2:
        for x, e, sgn in ((a[0], a[1], 1), (a[1], a[0], -1)):
            if e[0] == "O" and e[1] == "EMA" and _same(e[3][0], x):
                if name == "SUB":
                    return "%s偏离其%d小时均线的幅度%s" % (gloss(x), e[2], "" if sgn > 0 else "（取负）")
                return ("%s与其%d小时均线之比" if sgn > 0 else "%d小时均线与%s之比") % (
                    (gloss(x), e[2]) if sgn > 0 else (e[2], gloss(x)))
    if name in TS:
        return TS[name].format(x=gloss(a[0]), w=w)
    if name == "TS_CORR":
        return "%s与%s的%d小时相关" % (gloss(a[0]), gloss(a[1]), w)
    if name in CS:
        return CS[name].format(x=gloss(a[0]))
    if name == "NEG":
        return "负的" + gloss(a[0])
    if name == "ABS":
        return "|%s|" % gloss(a[0])
    if name == "SIGN":
        return "%s的正负号" % gloss(a[0])
    if name in ("ADD", "SUB", "MUL", "DIV"):
        op = dict(ADD="+", SUB="−", MUL="×", DIV="÷")[name]
        return "（%s %s %s）" % (gloss(a[0]), op, gloss(a[1]))
    if name == "GATE":
        return "%s>0 时取%s、否则取%s" % (gloss(a[0]), gloss(a[1]), gloss(a[2]))
    if name == "PER_ASSET_THRESH":
        return "%s高于%s记 +1、低于%s记 −1" % (gloss(a[0]), gloss(a[1]), gloss(a[2]))
    return name


def explain(vm, body):
    n = tree(vm, body)
    flip = False
    while n[0] == "O" and (n[1] in CS_MONO or n[1] == "NEG"):
        if n[1] == "NEG":
            flip = not flip
        n = n[3][0]
    g = gloss(n)
    if len(g) > MAX_CHARS:
        g = g[:MAX_CHARS] + "…"
    hi, lo = ("最低", "最高") if flip else ("最高", "最低")
    return "每小时按「%s」排序，做多%s的、做空%s的。" % (g, hi, lo)
