"""Sub-expression cache for xs_tscs_vm_v2 (XS_SEL1 s3.1 speed-up). New file; the VM is imported, never modified.

CachedExec.execute(tokens, features, chunk) walks the postfix program exactly as XsTsCsStackVM.execute does and calls
the VM's own _apply_operator for every operator it has to evaluate. Each operator's result is cached on the device
under (chunk, the token span of its subtree); a later program (same update or a later one) that contains the same
subtree on the same chunk gets the cached tensor instead of re-running the operator. Same inputs, same kernels, so
the output is the one execute() returns; sel1_trainer's zero-diff check compares the logs line by line.
Malformed stacks are never served from the cache: they go to _apply_operator, which raises as execute() does.
LRU eviction by bytes.
"""
from __future__ import annotations

from collections import OrderedDict

from okx_alpha_discovery.adapters.xs_tscs_vm_v2 import (CandidateLocalInvalid, OPERATOR_ARITIES,
                                                        WINDOWED_OPERATORS)

_ARITY = dict(OPERATOR_ARITIES)
_WIN = frozenset(WINDOWED_OPERATORS)


class CachedExec:
    def __init__(self, vm, budget_gb):
        self.vm = vm
        self.budget = int(float(budget_gb) * 2 ** 30)
        self.cache = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0

    def _put(self, key, value):
        nb = value.element_size() * value.nelement()
        if nb > self.budget:
            return
        self.cache[key] = value
        self.bytes += nb
        while self.bytes > self.budget:
            _, old = self.cache.popitem(last=False)
            self.bytes -= old.element_size() * old.nelement()

    def execute(self, tokens, features, chunk):
        vm = self.vm
        tokens = list(tokens)
        stack, spans = [], []
        for i, raw_token in enumerate(tokens):
            try:
                token = int(raw_token)
            except (TypeError, ValueError) as exc:
                raise CandidateLocalInvalid("NON_INTEGER_TOKEN") from exc
            if token < 0:
                raise CandidateLocalInvalid("NEGATIVE_TOKEN")
            if vm.is_end(token):
                break
            if vm.is_terminal(token):
                stack.append(features[:, token, :])
                spans.append(i)
            elif vm.is_window(token):
                stack.append(vm.window_value(token))
                spans.append(i)
            elif vm.is_operator(token):
                name = vm.operator_name(token)
                k = _ARITY[name] + (1 if name in _WIN else 0)
                wellformed = (len(stack) >= k
                              and (name not in _WIN or isinstance(stack[-1], int))
                              and not any(isinstance(a, int) for a in stack[len(stack) - k:len(stack) - (k - _ARITY[name])]))
                if not wellformed:
                    vm._apply_operator(name, stack)          # raises exactly as execute() would
                    raise CandidateLocalInvalid("CSE_UNREACHABLE")
                start = spans[-k]
                key = (chunk, tuple(int(t) for t in tokens[start:i + 1]))
                hit = self.cache.get(key)
                if hit is not None:
                    del stack[-k:]
                    del spans[-k:]
                    stack.append(hit)
                    self.cache.move_to_end(key)
                    self.hits += 1
                else:
                    vm._apply_operator(name, stack)
                    del spans[-k:]
                    self._put(key, stack[-1])
                    self.misses += 1
                spans.append(start)
            else:
                raise CandidateLocalInvalid("UNKNOWN_TOKEN")
        if len(stack) != 1:
            raise CandidateLocalInvalid("FINAL_STACK_DEPTH")
        if isinstance(stack[0], int):
            raise CandidateLocalInvalid("PROGRAM_IS_A_WINDOW")
        return stack[0]
