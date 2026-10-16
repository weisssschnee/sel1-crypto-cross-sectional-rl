"""Sampler for the searched-window VM: END token, precomputed legality tables.

Two things separate this from the legacy broad893 sampler.

Termination.  The old sampler ran ``for step in range(max_len)`` and forced the
stack to land at depth 1 on the final step, so every program was exactly
``max_len`` tokens.  With operators and windows encoded as separate tokens that
would have capped formulas at roughly five operators, strictly weaker than the
baked vocabulary it replaces.  Here a program ends when it emits END, so
``max_len`` is a ceiling and a two-token formula is as reachable as a forty-token
one.

Feasibility.  Legality is a pure function of (value_depth, window_on_top,
remaining), so the whole action mask is precomputed once into a table the sampler
indexes on device.  The legacy sampler recomputed dimension feasibility every
step behind a CPU round-trip and a triple Python loop, which is where its 1.4 s
per update went.

The minimum-completion cost is solved as a shortest path over stack states
rather than written as a closed form, because a per-basin vocabulary exposes only
some operators: if every unary operator is withheld, the cheapest way to fold a
depth-3 stack is genuinely longer, and a closed form derived from the full
operator set would let the sampler walk into a state it cannot close.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .xs_tscs_vm_v2 import (
    OPERATOR_ARITIES,
    OPERATOR_NAMES,
    UNREACHABLE,
    WINDOWED_OPERATORS,
    XsTsCsStackVM,
    solve_min_finish,
)

_OP_ORDER = {name: i for i, name in enumerate(OPERATOR_NAMES)}

_OP_DELTA_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _op_depth_delta(vm: XsTsCsStackVM, device: torch.device) -> torch.Tensor:
    """Stack-depth change per operator token: 1 - value_arity."""
    key = (id(vm), str(device))
    cached = _OP_DELTA_CACHE.get(key)
    if cached is None:
        arity = dict(OPERATOR_ARITIES)
        cached = torch.tensor(
            [1 - arity[name] for name in OPERATOR_NAMES], dtype=torch.long, device=device
        )
        _OP_DELTA_CACHE[key] = cached
    return cached


@dataclass(frozen=True)
class LegalityTable:
    """Precomputed action mask, indexed by (value_depth, window_on_top, remaining).

    ``mask`` holds 0 for a legal token and -inf for an illegal one, so it adds
    directly to the policy logits.  ``any_legal`` marks states with no legal
    action; those are unreachable by construction and asserted against rather
    than silently repaired.
    """

    mask: torch.Tensor  # (max_depth+1, 2, max_len+1, token_count)
    any_legal: torch.Tensor  # (max_depth+1, 2, max_len+1)
    min_finish: torch.Tensor  # (max_depth+1, 2)
    max_depth: int
    max_len: int

    def to(self, device: torch.device | str) -> "LegalityTable":
        return LegalityTable(
            mask=self.mask.to(device),
            any_legal=self.any_legal.to(device),
            min_finish=self.min_finish.to(device),
            max_depth=self.max_depth,
            max_len=self.max_len,
        )


def build_legality_table(
    vm: XsTsCsStackVM,
    max_len: int,
    *,
    available_terminals: tuple[int, ...] | None = None,
    available_ops: tuple[str, ...] | None = None,
    available_windows: tuple[int, ...] | None = None,
    dtype: torch.dtype = torch.float32,
) -> LegalityTable:
    """Enumerate legality for every reachable sampler state.

    The subset arguments are what makes a per-basin vocabulary honest: the
    minimum-completion costs are solved against the operators actually on offer,
    so a basin that withholds every unary operator gets a table that refuses the
    deep states it could not fold, instead of walking into them and raising.
    """
    if max_len < 2:
        raise ValueError("max_len must leave room for a value and END")
    terminals = (
        tuple(range(vm.feature_count)) if available_terminals is None else tuple(available_terminals)
    )
    ops = tuple(n for n, _ in OPERATOR_ARITIES) if available_ops is None else tuple(available_ops)
    windows = tuple(vm.windows) if available_windows is None else tuple(available_windows)
    if not terminals:
        raise ValueError("no terminals available")
    unknown = set(ops) - {n for n, _ in OPERATOR_ARITIES}
    if unknown:
        raise ValueError(f"unknown operators: {sorted(unknown)}")
    if any(w not in vm.windows for w in windows):
        raise ValueError("window not in the VM's window set")
    if any(n in WINDOWED_OPERATORS for n in ops) and not windows:
        raise ValueError("windowed operators offered with no window selectors")

    # A window selector token never raises value depth, so the stack can never
    # exceed the number of value-producing tokens the budget allows.
    max_depth = max_len
    arity = dict(OPERATOR_ARITIES)
    cost = solve_min_finish(
        max_depth,
        terminal_available=bool(terminals),
        window_available=bool(windows),
        available_ops=ops,
    )

    window_tokens = [vm.window_base + vm.windows.index(w) for w in windows]
    op_tokens = {name: vm.operator_base + _OP_ORDER[name] for name in ops}

    mask = torch.full((max_depth + 1, 2, max_len + 1, vm.token_count), float("-inf"), dtype=dtype)
    term_idx = torch.tensor(terminals, dtype=torch.long)
    win_idx = torch.tensor(window_tokens, dtype=torch.long) if window_tokens else None

    for depth in range(max_depth + 1):
        for window in (0, 1):
            for remaining in range(max_len + 1):

                def fits(next_depth: int, next_window: int) -> bool:
                    if next_depth > max_depth:
                        return False
                    c = cost[next_depth][next_window]
                    return c < UNREACHABLE and 1 + c <= remaining

                row = mask[depth][window][remaining]
                if window == 0:
                    if fits(depth + 1, 0):
                        row[term_idx] = 0.0
                    if win_idx is not None and depth >= 1 and fits(depth, 1):
                        row[win_idx] = 0.0
                    if depth == 1 and remaining >= 1:
                        row[vm.end_token] = 0.0
                for name in ops:
                    value_arity = arity[name]
                    if (name in WINDOWED_OPERATORS) != bool(window):
                        continue
                    if depth < value_arity:
                        continue
                    if fits(depth - value_arity + 1, 0):
                        row[op_tokens[name]] = 0.0

    any_legal = torch.isfinite(mask).any(dim=-1)
    min_finish = torch.tensor(
        [[cost[d][w] for w in (0, 1)] for d in range(max_depth + 1)], dtype=torch.long
    )
    return LegalityTable(
        mask=mask, any_legal=any_legal, min_finish=min_finish, max_depth=max_depth, max_len=max_len
    )


@dataclass
class SampledBatch:
    """One batch of programs, plus everything the update needs to stay honest.

    ``lengths`` counts tokens up to and including END.  It matters: the policy
    gradient must see exactly the steps where a decision was made.  Padding steps
    are forced to END, so leaving them in would train the policy on choices it
    never had and skew the entropy term toward zero as programs get shorter.
    """

    formulas: list[tuple[int, ...]]
    context: dict[str, Any]

    @property
    def lengths(self) -> torch.Tensor:
        return self.context["lengths"]


def sample_batch(
    model,
    vm: XsTsCsStackVM,
    table: LegalityTable,
    batch_size: int,
    *,
    prior_bias: torch.Tensor | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> SampledBatch:
    """Roll out ``batch_size`` programs under the legality table.

    The rollout deliberately runs with the autograd graph live.  The existing
    update step reads ``context["effective_logits"]`` and backpropagates through
    it directly, and that file is pinned by a sha256 check, so it cannot be
    changed to recompute the forward pass instead: sampling under ``no_grad``
    would silently leave the policy gradient constant and the policy would never
    learn.  Sampling itself is non-differentiable, so only the stacked logits
    need the graph.

    The returned ``effective_logits`` are the masked logits actually sampled
    from, which is what the policy gradient must differentiate; recomputing them
    later from the model would silently drop the mask.
    """
    device = torch.device(device)
    if table.mask.device != device:
        table = table.to(device)
    max_len = table.max_len
    end = vm.end_token

    bias = torch.zeros(vm.token_count, dtype=table.mask.dtype, device=device)
    if prior_bias is not None:
        bias = prior_bias.to(device=device, dtype=table.mask.dtype)
        if bias.shape != (vm.token_count,):
            raise ValueError(f"prior_bias must be ({vm.token_count},), got {tuple(bias.shape)}")

    depth = torch.zeros(batch_size, dtype=torch.long, device=device)
    window = torch.zeros(batch_size, dtype=torch.long, device=device)
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    lengths = torch.full((batch_size,), max_len, dtype=torch.long, device=device)

    inp = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
    token_steps: list[torch.Tensor] = []
    logit_steps: list[torch.Tensor] = []

    force_end = torch.full((vm.token_count,), float("-inf"), dtype=table.mask.dtype, device=device)
    force_end[end] = 0.0

    for step in range(max_len):
        remaining = torch.full_like(depth, max_len - step)
        legal = table.mask[depth, window, remaining]
        legal = torch.where(done[:, None], force_end[None, :], legal)

        if not bool(torch.isfinite(legal).any(dim=1).all()):
            stuck = torch.nonzero(~torch.isfinite(legal).any(dim=1)).flatten()[:8]
            raise RuntimeError(
                "sampler reached a state with no legal action at step "
                f"{step}: depths={depth[stuck].tolist()} window={window[stuck].tolist()}"
            )

        logits, _, _ = model(inp)
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        effective = logits.to(table.mask.dtype) + legal + torch.where(
            done[:, None], torch.zeros_like(bias)[None, :], bias[None, :]
        )

        probs = torch.softmax(effective, dim=-1)
        action = torch.multinomial(probs, 1, generator=generator).squeeze(1)

        token_steps.append(action)
        logit_steps.append(effective)

        newly_done = (~done) & (action == end)
        lengths = torch.where(newly_done, torch.full_like(lengths, step + 1), lengths)

        is_term = (action < vm.feature_count) & ~done
        is_win = (action >= vm.window_base) & (action < end) & ~done
        is_op = (action >= vm.operator_base) & (action < vm.window_base) & ~done

        delta = torch.zeros_like(depth)
        delta = torch.where(is_term, torch.ones_like(delta), delta)
        if bool(is_op.any()):
            # torch.where evaluates both branches, so the gather index has to be
            # valid for every row, including the window and END rows it discards.
            op_delta = _op_depth_delta(vm, device)
            slot = (action - vm.operator_base).clamp_(0, op_delta.shape[0] - 1)
            delta = torch.where(is_op, op_delta[slot], delta)
        depth = depth + delta
        window = torch.where(is_win, torch.ones_like(window), torch.zeros_like(window))
        done = done | (action == end)
        inp = torch.cat([inp, action[:, None]], dim=1)
        if bool(done.all()):
            break

    if not bool(done.all()):
        raise RuntimeError("sampler exhausted max_len without terminating every program")

    tokens = torch.stack(token_steps, dim=1)
    formulas = [
        tuple(int(t) for t in row[: int(n)].tolist()) for row, n in zip(tokens, lengths.cpu())
    ]
    return SampledBatch(
        formulas=formulas,
        context={
            "tokens": tokens,
            "effective_logits": torch.stack(logit_steps, dim=1),
            "lengths": lengths,
            "max_len": max_len,
        },
    )
