"""Source-pinned DSO optimizer math port for AlphaGPT reference testing."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Hashable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RiskSelection:
    quantile: float
    baseline: float
    elite_mask: np.ndarray
    elite_indices: np.ndarray
    clipped_rewards: np.ndarray


def select_risk_elite(rewards: np.ndarray, epsilon: float) -> RiskSelection:
    """Apply official DSO higher-quantile selection and the R_e baseline."""

    rewards = np.asarray(rewards)
    quantile = np.quantile(rewards, 1.0 - epsilon, method="higher")
    elite_mask = rewards >= quantile
    return RiskSelection(
        quantile=float(quantile),
        baseline=float(quantile),
        elite_mask=elite_mask,
        elite_indices=np.flatnonzero(elite_mask),
        clipped_rewards=np.clip(rewards[elite_mask], -1e6, 1e6),
    )


def sequence_neglogp_entropy(
    logits: torch.Tensor,
    actions: torch.Tensor,
    lengths: torch.Tensor,
    entropy_gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return length-masked sequence NLL and gamma-decayed summed entropy."""

    logprobs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(logprobs)
    max_length = actions.shape[1]
    mask = (
        torch.arange(max_length, device=logits.device)[None, :]
        < lengths[:, None]
    ).float()

    actions_one_hot = F.one_hot(actions, num_classes=logits.shape[-1]).float()
    action_logprobs = torch.where(
        actions_one_hot == 0, torch.ones_like(logprobs), logprobs
    )
    neglogp_per_step = -torch.sum(actions_one_hot * action_logprobs, dim=2)
    neglogp = torch.sum(neglogp_per_step * mask, dim=1)

    safe_logprobs = torch.where(probs == 0, torch.ones_like(logprobs), logprobs)
    entropy_per_step = -torch.sum(probs * safe_logprobs, dim=2)
    decay = torch.tensor(
        [entropy_gamma**t for t in range(max_length)],
        dtype=torch.float32,
        device=logits.device,
    )
    entropy = torch.sum(entropy_per_step * decay.unsqueeze(0) * mask, dim=1)
    return neglogp, entropy


@dataclass(frozen=True)
class PolicyGradientTerms:
    neglogp: torch.Tensor
    entropy: torch.Tensor
    entropy_loss: torch.Tensor
    pg_loss: torch.Tensor
    total_loss: torch.Tensor


def policy_gradient_terms(
    logits: torch.Tensor,
    actions: torch.Tensor,
    lengths: torch.Tensor,
    rewards: torch.Tensor,
    baseline: float,
    entropy_weight: float,
    entropy_gamma: float,
) -> PolicyGradientTerms:
    neglogp, entropy = sequence_neglogp_entropy(
        logits, actions, lengths, entropy_gamma
    )
    entropy_loss = -entropy_weight * torch.mean(entropy)
    pg_loss = torch.mean((rewards - baseline) * neglogp)
    return PolicyGradientTerms(
        neglogp=neglogp,
        entropy=entropy,
        entropy_loss=entropy_loss,
        pg_loss=pg_loss,
        total_loss=entropy_loss + pg_loss,
    )


@dataclass(frozen=True)
class PQTTerms:
    sampled_neglogp: torch.Tensor
    sampled_entropy: torch.Tensor
    entropy_loss: torch.Tensor
    pg_loss: torch.Tensor | None
    pqt_neglogp: torch.Tensor
    pqt_loss: torch.Tensor
    total_loss: torch.Tensor


def pqt_terms(
    sampled_logits: torch.Tensor,
    sampled_actions: torch.Tensor,
    sampled_lengths: torch.Tensor,
    sampled_rewards: torch.Tensor,
    baseline: float,
    pqt_logits: torch.Tensor,
    pqt_actions: torch.Tensor,
    pqt_lengths: torch.Tensor,
    entropy_weight: float,
    entropy_gamma: float,
    pqt_weight: float,
    pqt_use_pg: bool,
) -> PQTTerms:
    sampled_neglogp, sampled_entropy = sequence_neglogp_entropy(
        sampled_logits, sampled_actions, sampled_lengths, entropy_gamma
    )
    entropy_loss = -entropy_weight * torch.mean(sampled_entropy)
    total_loss = entropy_loss

    pg_loss = None
    if pqt_use_pg:
        pg_loss = torch.mean((sampled_rewards - baseline) * sampled_neglogp)
        total_loss = total_loss + pg_loss

    pqt_neglogp, _ = sequence_neglogp_entropy(
        pqt_logits, pqt_actions, pqt_lengths, entropy_gamma
    )
    pqt_loss = pqt_weight * torch.mean(pqt_neglogp)
    return PQTTerms(
        sampled_neglogp=sampled_neglogp,
        sampled_entropy=sampled_entropy,
        entropy_loss=entropy_loss,
        pg_loss=pg_loss,
        pqt_neglogp=pqt_neglogp,
        pqt_loss=pqt_loss,
        total_loss=total_loss + pqt_loss,
    )


class _QueueItem:
    def __init__(self, score: float, key: Hashable, payload: Any) -> None:
        self.score = score
        self.key = key
        self.payload = payload

    def __lt__(self, other: "_QueueItem") -> bool:
        return self.score < other.score


class UniqueTopKQueue:
    """Official DSO unique priority-queue and local RandomState semantics."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = capacity
        self.rng = np.random.RandomState(seed)
        self.heap: list[_QueueItem] = []
        self.unique_items: set[Hashable] = set()

    def __len__(self) -> int:
        return len(self.heap)

    def push(self, score: float, key: Hashable, payload: Any) -> None:
        if key in self.unique_items:
            return
        item = _QueueItem(score, key, payload)
        if len(self.heap) >= self.capacity:
            popped = heapq.heappushpop(self.heap, item)
            self.unique_items.add(key)
            self.unique_items.remove(popped.key)
        else:
            heapq.heappush(self.heap, item)
            self.unique_items.add(key)

    def push_best(
        self,
        rewards: Sequence[float],
        keys: Sequence[Hashable],
        payloads: Sequence[Any],
    ) -> int:
        index = int(np.argmax(rewards))
        self.push(rewards[index], keys[index], payloads[index])
        return index

    def sample(self, sample_size: int) -> list[Any]:
        indices = self.rng.choice(len(self.heap), sample_size)
        return [self.heap[index].payload for index in indices]

    def scored_items(self) -> dict[Hashable, float]:
        return {item.key: float(item.score) for item in self.heap}
