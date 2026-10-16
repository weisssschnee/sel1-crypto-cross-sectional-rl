"""M1 sampling: B-line programs + one HOLD step appended after END.

The HOLD choice is a real sampling step with autograd live, so the pinned
novelty_weighted_risk_update backprops into it through effective_logits --
holding time is jointly learned with the program, per the 2026-10-02 ruling.
Token layout: [terminals | operators | windows | END | HOLD_1..HOLD_n], n = len(HOLDS).
The VM never sees the trailing HOLD (execute stops at END).
"""
from __future__ import annotations

import torch

from okx_alpha_discovery.adapters.xs_tscs_sampler_v2 import sample_batch

import m1_common as _C

HOLDS = _C.HOLDS
HOLD_NAMES = [f"H{m}m" for m in HOLDS]


class BodyView(torch.nn.Module):
    """Shows the B-line sampler only the body vocabulary (first vm.token_count
    logits); hold tokens are invisible during body steps, so the legality mask
    width matches. The full model is kept for the hold step."""

    def __init__(self, model, body_vocab):
        super().__init__()
        self.model = model
        self.body_vocab = body_vocab

    def forward(self, inp):
        logits, a, b = self.model(inp)
        return logits[..., : self.body_vocab], a, b


def hold_token(vm, hold_minutes: int) -> int:
    return vm.end_token + 1 + HOLDS.index(hold_minutes)


def hold_of(vm, tokens) -> int:
    """Program's hold in minutes; assumes the trailing token after END is the hold."""
    end = tokens.index(vm.end_token) if vm.end_token in tokens else len(tokens) - 2
    if end + 1 < len(tokens) and tokens[end + 1] > vm.end_token:
        return HOLDS[tokens[end + 1] - vm.end_token - 1]
    return HOLDS[0]


def strip_hold(vm, tokens):
    end = tokens.index(vm.end_token)
    return tuple(tokens[: end + 1])


def sample_batch_with_hold(model, vm, table, batch_size, *, device, generator=None):
    body = BodyView(model, vm.token_count)
    b = sample_batch(body, vm, table, batch_size, device=device, generator=generator)
    ctx = b.context
    tokens, lengths = ctx["tokens"], ctx["lengths"]
    B = tokens.shape[0]
    V = vm.token_count + len(HOLDS)

    # model input: each program's tokens up to END, padded with END
    L = int(lengths.max())
    inp = torch.full((B, L), vm.end_token, dtype=torch.long, device=device)
    pos = torch.arange(L, device=device)[None, :] < lengths[:, None]
    inp = torch.where(pos, tokens[:, :L], inp)

    logits, _, _ = model(inp)  # (B, L, V) or (B, V) for last-step models
    last = (lengths - 1).clamp(min=0)
    if logits.dim() == 3:
        step_logits = logits[torch.arange(B, device=device), last]  # (B, V)
    else:
        step_logits = logits

    hold_mask = torch.full((V,), float("-inf"), device=device, dtype=step_logits.dtype)
    for i in range(len(HOLDS)):
        hold_mask[vm.end_token + 1 + i] = 0.0
    effective = step_logits + hold_mask[None, :]
    probs = torch.softmax(effective, dim=-1)
    action = torch.multinomial(probs, 1, generator=generator).squeeze(1)  # (B,)

    # extend sequences by one step: the hold token lands right after END.
    # tokens2 width is S+1 and put = lengths <= S, so the index always fits.
    S = tokens.shape[1]
    pad = torch.full((B, 1), vm.end_token, dtype=torch.long, device=device)
    tokens2 = torch.cat([tokens, pad], dim=1)
    put = lengths
    tokens2[torch.arange(B, device=device), put] = action
    lengths2 = lengths + 1

    # effective_logits at uniform width V: body rows carry -inf on the hold
    # columns (those tokens are genuinely illegal mid-body), the hold row
    # carries the real masked logits of the hold decision.
    el = ctx["effective_logits"]
    el_W = el.shape[-1]
    if el_W < V:
        el = torch.cat([el, torch.full((B, el.shape[1], V - el_W), float("-inf"),
                                       device=device, dtype=el.dtype)], dim=2)
    force = torch.full((V,), float("-inf"), device=device, dtype=el.dtype)
    force[vm.end_token] = 0.0
    el_pad = force[None, None, :].expand(B, 1, V)
    el2 = torch.cat([el, el_pad], dim=1).clone()
    el2[torch.arange(B, device=device), put] = effective

    from dataclasses import dataclass

    @dataclass
    class Batch2:
        formulas: list
        context: dict

        @property
        def lengths(self):
            return self.context["lengths"]

    formulas = []
    for row, n in zip(tokens2.tolist(), lengths2.tolist()):
        formulas.append(tuple(row[:n]))
    return Batch2(formulas=formulas, context={
        "tokens": tokens2, "effective_logits": el2, "lengths": lengths2, "max_len": ctx["max_len"] + 1,
        "hold_logits": effective, "hold_tokens": action,
    })
