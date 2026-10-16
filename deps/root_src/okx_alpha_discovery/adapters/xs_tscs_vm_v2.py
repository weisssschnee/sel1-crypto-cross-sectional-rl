"""Causal cross-sectional VM whose time windows are searched, not hardcoded.

Three token classes share one integer space:

    [0, F)                    terminals: raw panel fields
    [F, F + nops)             operators
    [F + nops, F + nops + nw) window selectors

A windowed operator pops one window selector plus its value arguments, so the
policy learns *how far back to look* instead of inheriting a baked 1/6/24
ladder.  Windows are log spaced, and EMA takes a half-life rather than a box
window, which removes the hard cutoff entirely.

Contract, enforced by tests:
  * output at t depends only on inputs at or before t;
  * a time-series operator never reads another asset;
  * a cross-section operator reduces over assets at each t independently;
  * insufficient finite history yields missing, never zero;
  * every window selector pushed must be consumed.
"""

from __future__ import annotations

from typing import Any, Sequence

MIN_FINITE_FRACTION = 0.3
_CS_MIN_ASSETS = 2


class CandidateLocalInvalid(RuntimeError):
    """The token string is not a well-formed program under this contract."""


def _torch():
    import torch

    return torch


# --- window ladder ---------------------------------------------------------
# Log spaced over one hour .. one week so the policy can pick its resolution
# without the vocabulary growing multiplicatively with the operator set.
WINDOWS: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 36, 48, 72, 96, 120, 168)

# --- operator table --------------------------------------------------------
# (name, value_arity, windowed, dimension_class, min_window)
#
# ``min_window`` is the smallest window at which the operator is defined at all.
# TS_STD/TS_SLOPE/TS_CORR need two points to produce a number and TS_ZSCORE
# needs a non-zero spread, so at w=1 they emit all-NaN, the signal becomes
# empty, every weight is zeroed and the program trades nothing.  A no-trade
# program scores exactly 0.0, which beats most real programs once costs are
# charged -- so leaving these combinations legal hands the policy a reward
# hack.  The legality table refuses them instead of letting the reward find
# them.
OPERATOR_TABLE: tuple[tuple[str, int, bool, str, int], ...] = (
    ("ADD", 2, False, "same", 1),
    ("SUB", 2, False, "same", 1),
    ("MUL", 2, False, "add", 1),
    ("DIV", 2, False, "sub", 1),
    ("NEG", 1, False, "keep", 1),
    ("ABS", 1, False, "keep", 1),
    ("SIGN", 1, False, "dimensionless", 1),
    ("GATE", 3, False, "same_tail", 1),
    ("PER_ASSET_THRESH", 3, False, "dimensionless", 1),
    ("LAG", 1, True, "keep", 1),
    ("DELTA", 1, True, "keep", 1),
    ("TS_MEAN", 1, True, "keep", 1),
    ("TS_STD", 1, True, "keep", 2),
    ("TS_MAX", 1, True, "keep", 1),
    ("TS_MIN", 1, True, "keep", 1),
    ("TS_ZSCORE", 1, True, "dimensionless", 2),
    ("TS_RANK", 1, True, "dimensionless", 2),
    ("TS_SLOPE", 1, True, "keep", 2),
    ("EMA", 1, True, "keep", 1),
    ("TS_CORR", 2, True, "dimensionless", 2),
    ("CS_DEMEAN", 1, False, "keep", 1),
    ("CS_ZSCORE", 1, False, "dimensionless", 1),
    ("CS_RANK", 1, False, "dimensionless", 1),
)

OPERATOR_NAMES: tuple[str, ...] = tuple(n for n, _, _, _, _ in OPERATOR_TABLE)
OPERATOR_ARITIES: tuple[tuple[str, int], ...] = tuple(
    (n, a) for n, a, _, _, _ in OPERATOR_TABLE
)
WINDOWED_OPERATORS: tuple[str, ...] = tuple(n for n, _, w, _, _ in OPERATOR_TABLE if w)
PLAIN_OPERATORS: tuple[str, ...] = tuple(n for n, _, w, _, _ in OPERATOR_TABLE if not w)
TIME_SERIES_OPERATORS: tuple[str, ...] = WINDOWED_OPERATORS
CROSS_SECTION_OPERATORS: tuple[str, ...] = ("CS_DEMEAN", "CS_ZSCORE", "CS_RANK")
OPERATOR_DIMENSION_CLASSES: dict[str, str] = {
    n: c for n, _, _, c, _ in OPERATOR_TABLE
}
OPERATOR_MIN_WINDOW: dict[str, int] = {
    n: mw for n, _, _, _, mw in OPERATOR_TABLE
}

# Missing-value semantics, declared per operator so callers and tests read one
# authority instead of guessing.
#   pointwise  output at t reads x[t], so a hole at t is a hole at t
#   lagged     output at t reads x[t-w], so a hole at t surfaces at t+w
#   aggregate  reduces a trailing window, tolerating interior holes while at
#              least min_periods(window) finite points remain
POINTWISE_MISSING_OPERATORS: tuple[str, ...] = (
    "NEG", "ABS", "SIGN", "DELTA", "TS_ZSCORE", "TS_RANK", "PER_ASSET_THRESH",
    "CS_DEMEAN", "CS_ZSCORE", "CS_RANK",
)
LAGGED_MISSING_OPERATORS: tuple[str, ...] = ("LAG",)
AGGREGATE_MISSING_OPERATORS: tuple[str, ...] = (
    "TS_MEAN", "TS_STD", "TS_MAX", "TS_MIN", "TS_SLOPE", "EMA", "TS_CORR",
)


def min_periods(window: int) -> int:
    """Finite observations a windowed reduction needs before it emits a value."""
    return max(2, int(window * MIN_FINITE_FRACTION))


class XsTsCsVMContract:
    contract_id: str = "XS_CAUSAL_TSCS_VM_V2_SEARCHED_WINDOWS"
    layout: str = "TERMINAL_TENSOR_IS_ASSETS_BY_TIMES_TIME_ASCENDING"
    causality: str = "OUTPUT_AT_T_DEPENDS_ONLY_ON_INPUTS_AT_OR_BEFORE_T"
    missing_semantics: str = "PROPAGATE_MISSING; ROLLING_STATS_OVER_FINITE_WINDOW_ONLY"
    window_semantics: str = "WINDOW_IS_A_SEARCHED_TOKEN; EVERY_PUSHED_WINDOW_MUST_BE_CONSUMED"
    boundary_semantics: str = "INSUFFICIENT_FINITE_HISTORY_YIELDS_MISSING_NOT_ZERO"
    cross_section_semantics: str = "REDUCE_OVER_ASSETS_AT_EACH_T_INDEPENDENTLY"
    windows: tuple[int, ...] = WINDOWS


# --- windowed primitives ---------------------------------------------------
# Every trailing-window operator goes through one helper, so the causal
# boundary is written once instead of once per operator.


# Optional blocked evaluation of the trailing-window operators. 0 keeps the original path, which builds each
# (assets, times, window) work array for the whole panel. A positive value builds them for blocks of about
# this many elements instead, so they stay cache-resident. Every output element is still one row-wise
# reduction over the same window values in the same order, so the result is bit-identical;
# tests/test_xs_tscs_vm_v2_fast_paths.py asserts that on nan-laced data and every window.
BLOCK_ELEMS = 0


def _padded(x, window: int):
    torch = _torch()
    if window < 1:
        raise CandidateLocalInvalid("NON_POSITIVE_WINDOW")
    pad = torch.full(x.shape[:-1] + (window - 1,), float("nan"), dtype=x.dtype, device=x.device)
    return torch.cat((pad, x), dim=-1)


def _trailing_windows(x, window: int):
    """(A, T) -> (A, T, window); slot [..., t, k] holds x[t - window + 1 + k].

    The front is padded with missing so position t never sees the future, and
    ``window`` is a plain int, which keeps this a view-plus-pad rather than a
    gather.
    """
    return _padded(x, window).unfold(-1, window, 1)


def _apply_blocks(x, window: int, body, y=None):
    """body(w, x_block), or body(wx, wy) when y is given, over the whole panel or block by block in time."""
    torch = _torch()
    if not BLOCK_ELEMS:
        wx = _trailing_windows(x, window)
        return body(wx, x) if y is None else body(wx, _trailing_windows(y, window))
    px = _padded(x, window)
    py = None if y is None else _padded(y, window)
    T = x.shape[-1]
    step = max(1, BLOCK_ELEMS // (max(1, x.shape[0]) * window))
    out = []
    for t0 in range(0, T, step):
        t1 = min(T, t0 + step)
        # .contiguous(): on the unfold view, elementwise ops pick their output layout by comparing dimension
        # sizes, and once a block holds fewer time steps than the window they produce (assets, window, time)
        # instead of (assets, time, window). sum(dim=-1) then runs a different kernel and the last bit changes.
        # The whole-panel path always has more time steps than the window, so it always gets the contiguous layout.
        wx = px[..., t0:t1 + window - 1].unfold(-1, window, 1).contiguous()
        if y is None:
            out.append(body(wx, x[..., t0:t1]))
        else:
            out.append(body(wx, py[..., t0:t1 + window - 1].unfold(-1, window, 1).contiguous()))
    return torch.cat(out, dim=-1)


def _finite_parts(w):
    torch = _torch()
    finite = torch.isfinite(w)
    count = finite.to(w.dtype).sum(dim=-1)
    clean = torch.where(finite, w, torch.zeros_like(w))
    return finite, count, clean


def _gate_min_periods(value, count, window: int):
    torch = _torch()
    need = float(min_periods(window))
    return torch.where(count >= need, value, torch.full_like(value, float("nan")))


def op_lag(x, window: int):
    """x[t - window]; a hole at t surfaces at t + window."""
    torch = _torch()
    pad = torch.full(x.shape[:-1] + (window,), float("nan"), dtype=x.dtype, device=x.device)
    return torch.cat((pad, x), dim=-1)[..., : x.shape[-1]]


def op_delta(x, window: int):
    return x - op_lag(x, window)


def op_ts_mean(x, window: int):
    def body(w, xb):
        _, count, clean = _finite_parts(w)
        return _gate_min_periods(clean.sum(dim=-1) / count.clamp_min(1.0), count, window)

    return _apply_blocks(x, window, body)


def _moments_of(w):
    _, count, clean = _finite_parts(w)
    n = count.clamp_min(1.0)
    mean = clean.sum(dim=-1) / n
    sq = (clean * clean).sum(dim=-1) / n
    var = (sq - mean * mean).clamp_min(0.0)
    return mean, var, count


def _ts_moments(x, window: int):
    return _moments_of(_trailing_windows(x, window))


def op_ts_std(x, window: int):
    torch = _torch()

    def body(w, xb):
        mean, var, count = _moments_of(w)
        n = count.clamp_min(2.0)
        unbiased = var * n / (n - 1.0)
        return _gate_min_periods(torch.sqrt(unbiased), count, window)

    return _apply_blocks(x, window, body)


def op_ts_zscore(x, window: int):
    torch = _torch()

    def body(w, xb):
        mean, var, count = _moments_of(w)
        n = count.clamp_min(2.0)
        std = torch.sqrt((var * n / (n - 1.0)).clamp_min(0.0))
        z = (xb - mean) / torch.where(std > 0, std, torch.full_like(std, float("nan")))
        return _gate_min_periods(z, count, window)

    return _apply_blocks(x, window, body)


def op_ts_max(x, window: int):
    torch = _torch()

    def body(w, xb):
        finite, count, _ = _finite_parts(w)
        filled = torch.where(finite, w, torch.full_like(w, float("-inf")))
        return _gate_min_periods(filled.max(dim=-1).values, count, window)

    return _apply_blocks(x, window, body)


def op_ts_min(x, window: int):
    torch = _torch()

    def body(w, xb):
        finite, count, _ = _finite_parts(w)
        filled = torch.where(finite, w, torch.full_like(w, float("inf")))
        return _gate_min_periods(filled.min(dim=-1).values, count, window)

    return _apply_blocks(x, window, body)


def op_ts_rank(x, window: int):
    """Where x[t] sits inside its own trailing window, mapped to [-1, 1]."""
    torch = _torch()

    def body(w, xb):
        finite, count, _ = _finite_parts(w)
        below = (w <= xb.unsqueeze(-1)) & finite
        frac = below.to(xb.dtype).sum(dim=-1) / count.clamp_min(1.0)
        ranked = 2.0 * frac - 1.0
        ranked = torch.where(torch.isfinite(xb), ranked, torch.full_like(ranked, float("nan")))
        return _gate_min_periods(ranked, count, window)

    return _apply_blocks(x, window, body)


def op_ts_slope(x, window: int):
    """OLS slope of x on time inside the trailing window, per hour."""
    torch = _torch()

    def body(w, xb):
        finite, count, clean = _finite_parts(w)
        n = count.clamp_min(1.0)
        t = torch.arange(window, dtype=xb.dtype, device=xb.device)
        t = t.expand_as(w)
        t_clean = torch.where(finite, t, torch.zeros_like(t))
        mean_t = t_clean.sum(dim=-1) / n
        mean_x = clean.sum(dim=-1) / n
        cov = (t_clean * clean).sum(dim=-1) / n - mean_t * mean_x
        var_t = (t_clean * t_clean).sum(dim=-1) / n - mean_t * mean_t
        slope = cov / torch.where(var_t > 0, var_t, torch.full_like(var_t, float("nan")))
        return _gate_min_periods(slope, count, window)

    return _apply_blocks(x, window, body)


_EMA_KERNEL_CACHE: dict = {}
# 2026-10-01 B2 finding: at T=17520 one float64 kernel is 2.45 GB (the scorer's
# 5135-h window was 211 MB), so an unevicted cache of up to 15 windows put tens
# of GB in every worker and crashed torch's c10.dll under memory pressure. The
# cap evicts oldest-first; the default 64 keeps the previous behaviour for every
# caller that stays under it (identical numerics -- kernels are only
# re-materialized, never recomputed differently).
EMA_KERNEL_CACHE_MAX = 64


def _ema_kernel(length: int, decay: float, dtype, device):
    """Causal decay matrix K[k, t] = decay ** (t - k), zero where t < k.

    Turning the EMA recurrence into one matmul matters: as a Python loop over
    timesteps it was the slowest operator on GPU by 5x, since each step paid a
    kernel launch to do almost no work.  Built from a difference of exponents
    so no intermediate ever holds decay ** -k, which would underflow.
    """
    torch = _torch()
    key = (length, round(decay, 12), str(dtype), str(device))
    cached = _EMA_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached
    idx = torch.arange(length, dtype=dtype, device=device)
    # Used as ``x @ K``, so K's ROW index is the source time s and its COLUMN
    # index is the output time t.  lag[s, t] = t - s must therefore be negative
    # (and the entry zero) whenever t < s, or output t would read the future.
    lag = idx.unsqueeze(1).neg() + idx.unsqueeze(0)
    import math

    kernel = torch.where(
        lag >= 0,
        torch.exp(lag.clamp_min(0.0) * math.log(decay)),
        torch.zeros_like(lag),
    )
    if bool((kernel.tril(diagonal=-1) != 0).any()):
        raise CandidateLocalInvalid("EMA_KERNEL_NOT_CAUSAL")
    while len(_EMA_KERNEL_CACHE) >= EMA_KERNEL_CACHE_MAX:
        _EMA_KERNEL_CACHE.pop(next(iter(_EMA_KERNEL_CACHE)))
    _EMA_KERNEL_CACHE[key] = kernel
    return kernel


def op_ema(x, window: int):
    """Missing-aware EMA whose half-life is the selected window.

    Scale-free and boundary-free: there is no point at which history is
    discarded, only down-weighted, which is what makes it a better answer than
    a box window when the right horizon is unknown.
    """
    torch = _torch()
    decay = 0.5 ** (1.0 / float(window))
    finite = torch.isfinite(x)
    clean = torch.where(finite, x, torch.zeros_like(x))
    seen = finite.to(x.dtype)
    kernel = _ema_kernel(x.shape[-1], decay, x.dtype, x.device)
    num = clean @ kernel
    den = seen @ kernel
    value = num / torch.where(den > 0, den, torch.full_like(den, float("nan")))
    live = seen.cumsum(dim=-1)
    need = float(min_periods(window))
    return torch.where(live >= need, value, torch.full_like(value, float("nan")))


def op_ts_corr(x, y, window: int):
    """Pearson correlation of two series inside a shared trailing window."""
    torch = _torch()

    def body(wx, wy):
        both = torch.isfinite(wx) & torch.isfinite(wy)
        count = both.to(x.dtype).sum(dim=-1)
        n = count.clamp_min(1.0)
        cx = torch.where(both, wx, torch.zeros_like(wx))
        cy = torch.where(both, wy, torch.zeros_like(wy))
        mx = cx.sum(dim=-1) / n
        my = cy.sum(dim=-1) / n
        cov = (cx * cy).sum(dim=-1) / n - mx * my
        vx = (cx * cx).sum(dim=-1) / n - mx * mx
        vy = (cy * cy).sum(dim=-1) / n - my * my
        denom = torch.sqrt((vx * vy).clamp_min(0.0))
        corr = cov / torch.where(denom > 0, denom, torch.full_like(denom, float("nan")))
        return _gate_min_periods(corr.clamp(-1.0, 1.0), count, window)

    return _apply_blocks(x, window, body, y)


# --- pointwise and cross-sectional operators --------------------------------


def _cs_thin_to_missing(value, count):
    torch = _torch()
    return torch.where(count >= _CS_MIN_ASSETS, value, torch.full_like(value, float("nan")))


def _cs_counts(x):
    torch = _torch()
    finite = torch.isfinite(x)
    return finite, finite.to(x.dtype).sum(dim=0, keepdim=True)


def op_cs_demean(x):
    torch = _torch()
    finite, count = _cs_counts(x)
    clean = torch.where(finite, x, torch.zeros_like(x))
    mean = clean.sum(dim=0, keepdim=True) / count.clamp_min(1.0)
    return _cs_thin_to_missing(x - mean, count)


def op_cs_zscore(x):
    torch = _torch()
    finite, count = _cs_counts(x)
    clean = torch.where(finite, x, torch.zeros_like(x))
    n = count.clamp_min(1.0)
    mean = clean.sum(dim=0, keepdim=True) / n
    sq = (clean * clean).sum(dim=0, keepdim=True) / n
    nn = count.clamp_min(2.0)
    var = ((sq - mean * mean).clamp_min(0.0)) * nn / (nn - 1.0)
    std = torch.sqrt(var)
    z = (x - mean) / torch.where(std > 0, std, torch.full_like(std, float("nan")))
    return _cs_thin_to_missing(z, count)


def op_cs_rank(x):
    """Cross-sectional rank at each t, mapped to [-1, 1].

    Ranks come from a double argsort (A log A) rather than an all-pairs
    comparison (A squared); missing assets are sorted to the end and then
    masked back out.
    """
    torch = _torch()
    finite, count = _cs_counts(x)
    big = torch.where(finite, x, torch.full_like(x, float("inf")))
    order = big.argsort(dim=0)
    position = order.argsort(dim=0).to(x.dtype)
    frac = (position + 1.0) / count.clamp_min(1.0)
    frac = torch.where(finite, frac, torch.full_like(frac, float("nan")))
    return _cs_thin_to_missing(2.0 * frac - 1.0, count)


def op_add(a, b):
    return a + b


def op_sub(a, b):
    return a - b


def op_mul(a, b):
    return a * b


def op_div(a, b):
    torch = _torch()
    safe = torch.where(b.abs() > 0, b, torch.full_like(b, float("nan")))
    return a / safe


def op_neg(x):
    return -x


def op_abs(x):
    return x.abs()


def op_sign(x):
    torch = _torch()
    return torch.where(torch.isfinite(x), torch.sign(x), torch.full_like(x, float("nan")))


def op_gate(c, a, b):
    """Pointwise select: c > 0 picks a, otherwise b; missing c stays missing."""
    torch = _torch()
    picked = torch.where(c > 0, a, b)
    return torch.where(torch.isfinite(c), picked, torch.full_like(picked, float("nan")))


def op_per_asset_thresh(signal, long_th, short_th):
    """Per-asset independent {-1, 0, +1} position from signal and two
    thresholds.

    Inputs are (A, T) tensors. long_th and short_th are themselves per-asset
    (broadcast across time). The output is +1 where signal > long_th, -1
    where signal < short_th, and 0 elsewhere; missing at any of the three
    inputs stays missing. The two thresholds are per-asset constants that
    the program itself supplies as terminal values, so a single formula
    can declare its own per-asset entry / exit bands and the trainer no
    longer has to collapse per-asset information into a single
    cross-section rank before deciding a position.

    For NaN long_th or short_th on an asset the position is NaN there.
    """
    torch = _torch()
    both = torch.isfinite(long_th) & torch.isfinite(short_th)
    sig_ok = torch.isfinite(signal)
    long_ok = sig_ok & both & (signal > long_th)
    short_ok = sig_ok & both & (signal < short_th)
    out = torch.where(long_ok, torch.ones_like(signal),
                      torch.where(short_ok, -torch.ones_like(signal),
                                  torch.zeros_like(signal)))
    return torch.where(sig_ok & both, out, torch.full_like(out, float("nan")))


_PLAIN_IMPL = {
    "ADD": op_add, "SUB": op_sub, "MUL": op_mul, "DIV": op_div,
    "NEG": op_neg, "ABS": op_abs, "SIGN": op_sign, "GATE": op_gate,
    "PER_ASSET_THRESH": op_per_asset_thresh,
    "CS_DEMEAN": op_cs_demean, "CS_ZSCORE": op_cs_zscore, "CS_RANK": op_cs_rank,
}
_WINDOWED_IMPL = {
    "LAG": op_lag, "DELTA": op_delta, "TS_MEAN": op_ts_mean, "TS_STD": op_ts_std,
    "TS_MAX": op_ts_max, "TS_MIN": op_ts_min, "TS_ZSCORE": op_ts_zscore,
    "TS_RANK": op_ts_rank, "TS_SLOPE": op_ts_slope, "EMA": op_ema,
    "TS_CORR": op_ts_corr,
}


MAX_STACK_DEPTH = 256
UNREACHABLE = 1 << 30


def solve_min_finish(
    max_depth: int,
    *,
    terminal_available: bool = True,
    window_available: bool = True,
    available_ops: Sequence[str] | None = None,
) -> list[list[int]]:
    """Cheapest token count, END included, that closes each stack state.

    A state is ``(value_depth, window_on_top)``.  Solved as a shortest path over
    states instead of written as a formula, because a per-basin vocabulary
    exposes only some operators: withhold every unary operator and folding a
    deep stack genuinely costs more, so a cost derived from the full operator set
    would let the sampler walk into states it cannot close.  ``UNREACHABLE``
    marks a state with no completion at all; callers treat it as illegal rather
    than as a large number.
    """
    arity = dict(OPERATOR_ARITIES)
    names = tuple(n for n, _ in OPERATOR_ARITIES) if available_ops is None else tuple(available_ops)
    unknown = set(names) - set(arity)
    if unknown:
        raise ValueError(f"unknown operators: {sorted(unknown)}")
    ops = [(arity[n], n in WINDOWED_OPERATORS) for n in names]
    cost = [[UNREACHABLE, UNREACHABLE] for _ in range(max_depth + 2)]
    cost[1][0] = 1  # depth 1, nothing pending: spend END and stop

    changed = True
    while changed:  # edge costs are all 1 and the graph is tiny
        changed = False
        for depth in range(max_depth + 1):
            for window in (0, 1):
                best = cost[depth][window]
                if window == 0:
                    if terminal_available and depth + 1 <= max_depth:
                        best = min(best, 1 + cost[depth + 1][0])
                    if window_available and depth >= 1:
                        best = min(best, 1 + cost[depth][1])
                for value_arity, windowed in ops:
                    if windowed != bool(window) or depth < value_arity:
                        continue
                    nxt = depth - value_arity + 1
                    if nxt <= max_depth:
                        best = min(best, 1 + cost[nxt][0])
                if best < cost[depth][window]:
                    cost[depth][window] = best
                    changed = True
    return cost


# --- the machine -----------------------------------------------------------


class XsTsCsStackVM:
    """RPN evaluator over an (assets, features, times) panel.

    Token space, in one contiguous range so the policy head stays a single
    categorical:

        [0, F)                      terminal: panel field
        [F, F + nops)               operator
        [F + nops, F + nops + nw)   window selector

    A windowed operator pops its window selector first, then its values.  A
    window selector on top of the stack at the end of the program, or used as a
    value, is a malformed program: ``execute`` raises instead of guessing.
    """

    contract = XsTsCsVMContract

    def __init__(self, feature_count: int, windows: Sequence[int] = WINDOWS):
        if feature_count < 1:
            raise ValueError("feature_count must be positive")
        self.feature_count = int(feature_count)
        self.windows = tuple(int(w) for w in windows)
        if any(w < 1 for w in self.windows):
            raise ValueError("windows must be positive")
        self.operator_base = self.feature_count
        self.window_base = self.feature_count + len(OPERATOR_NAMES)
        self.end_token = self.window_base + len(self.windows)
        self.token_count = self.end_token + 1
        self._min_finish_cache = solve_min_finish(MAX_STACK_DEPTH)

    # token classification, also used by the sampler's legality mask
    def is_terminal(self, token: int) -> bool:
        return 0 <= token < self.feature_count

    def is_operator(self, token: int) -> bool:
        return self.operator_base <= token < self.window_base

    def is_window(self, token: int) -> bool:
        return self.window_base <= token < self.end_token

    def is_end(self, token: int) -> bool:
        return token == self.end_token

    def operator_name(self, token: int) -> str:
        return OPERATOR_NAMES[token - self.operator_base]

    def window_value(self, token: int) -> int:
        return self.windows[token - self.window_base]

    def describe(self, tokens: Sequence[int], field_names: Sequence[str] | None = None) -> str:
        """Render a token string as readable infix-ish text for receipts."""
        out: list[Any] = []
        for raw in tokens:
            token = int(raw)
            if self.is_end(token):
                break
            if self.is_terminal(token):
                out.append(field_names[token] if field_names else f"f{token}")
            elif self.is_window(token):
                out.append(self.window_value(token))
            else:
                name = self.operator_name(token)
                value_arity = dict(OPERATOR_ARITIES)[name]
                window = None
                if name in WINDOWED_OPERATORS:
                    window = out.pop() if out and isinstance(out[-1], int) else "?"
                args = [out.pop() for _ in range(min(value_arity, len(out)))][::-1]
                inner = ", ".join(str(a) for a in args)
                if window is not None:
                    inner = f"{inner}, w={window}"
                out.append(f"{name}({inner})")
        return str(out[-1]) if out else ""

    def _apply_operator(self, name: str, stack: list) -> None:
        value_arity = dict(OPERATOR_ARITIES)[name]
        windowed = name in WINDOWED_OPERATORS
        window = None
        if windowed:
            if not stack or not isinstance(stack[-1], int):
                raise CandidateLocalInvalid("MISSING_WINDOW_SELECTOR")
            window = stack.pop()
            # Enforced here rather than in the sampler's legality table on
            # purpose.  The table's budget check is a shortest-path solve over
            # (value_depth, window_on_top); making the window value part of that
            # state would have to re-derive the completion costs per window, and
            # an optimistic cost there silently admits programs that cannot be
            # closed.  A wasted sample is the cheaper error: roughly 3% of
            # windowed draws land here, they are refused as mechanical skips and
            # earn no reward, so the policy is never paid for the degeneracy.
            if window < OPERATOR_MIN_WINDOW[name]:
                raise CandidateLocalInvalid("WINDOW_BELOW_OPERATOR_MINIMUM")
        if len(stack) < value_arity:
            raise CandidateLocalInvalid("STACK_UNDERFLOW")
        args = [stack.pop() for _ in range(value_arity)][::-1]
        if any(isinstance(a, int) for a in args):
            raise CandidateLocalInvalid("WINDOW_USED_AS_VALUE")
        if windowed:
            stack.append(_WINDOWED_IMPL[name](*args, window))
        else:
            stack.append(_PLAIN_IMPL[name](*args))

    def execute(self, tokens: Sequence[int], features):
        """Evaluate one program; returns an (assets, times) tensor."""
        stack: list = []
        for raw_token in tokens:
            try:
                token = int(raw_token)
            except (TypeError, ValueError) as exc:
                raise CandidateLocalInvalid("NON_INTEGER_TOKEN") from exc
            if token < 0:
                raise CandidateLocalInvalid("NEGATIVE_TOKEN")
            if self.is_end(token):
                break
            if self.is_terminal(token):
                stack.append(features[:, token, :])
            elif self.is_window(token):
                stack.append(self.window_value(token))
            elif self.is_operator(token):
                self._apply_operator(self.operator_name(token), stack)
            else:
                raise CandidateLocalInvalid("UNKNOWN_TOKEN")
        if len(stack) != 1:
            raise CandidateLocalInvalid("FINAL_STACK_DEPTH")
        if isinstance(stack[0], int):
            raise CandidateLocalInvalid("PROGRAM_IS_A_WINDOW")
        return stack[0]

    # --- sampler support ---------------------------------------------------
    def min_tokens_to_finish(self, value_depth: int, window_on_top: bool) -> int:
        """Fewest tokens, END included, that can still close this partial program.

        Delegates to the shortest-path solver rather than a closed form.  A
        closed form has to assume how much folding power the operator set has,
        and an earlier one here assumed binary: it costed a depth-3 stack at 3
        tokens when GATE folds it in 1, so the budget check rejected deep
        programs that were in fact closable.  The solver reads the real arities,
        including whatever subset a basin exposes.
        """
        cost = self._min_finish_cache
        if value_depth >= len(cost):
            return UNREACHABLE
        return cost[value_depth][1 if window_on_top else 0]

    def legal_actions(self, value_depth: int, window_on_top: bool, remaining: int):
        """Tokens that can extend a partial program without dooming it.

        ``remaining`` counts every slot still available, including the END token
        the program must eventually spend.  An action is legal only when the
        cheapest completion of the state it leads to still fits, which is what
        keeps the length cap from silently banning deep formulas: a program is
        cut short only when it genuinely cannot be closed, never because the
        budget was spent on padding.
        """
        def fits(next_depth: int, next_window: bool) -> bool:
            return 1 + self.min_tokens_to_finish(next_depth, next_window) <= remaining

        legal_terminals = not window_on_top and fits(value_depth + 1, False)
        legal_windows = not window_on_top and value_depth >= 1 and fits(value_depth, True)
        legal_end = not window_on_top and value_depth == 1 and remaining >= 1
        ops: list[str] = []
        for name, value_arity in OPERATOR_ARITIES:
            if (name in WINDOWED_OPERATORS) != window_on_top:
                continue
            if value_depth < value_arity:
                continue
            if not fits(value_depth - value_arity + 1, False):
                continue
            ops.append(name)
        return legal_terminals, legal_windows, tuple(ops), legal_end
