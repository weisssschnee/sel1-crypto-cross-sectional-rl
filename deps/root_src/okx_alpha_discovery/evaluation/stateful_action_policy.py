from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class StatefulActionError(ValueError):
    pass


@dataclass(frozen=True)
class StatefulActionContract:
    contract_id: str = "XS_STATEFUL_HOLD_PARTIAL_SWITCH_V1"
    minimum_hold_hours: int = 12
    hold_l1_threshold: float = 0.50
    switch_l1_threshold: float = 1.00
    partial_fraction: float = 0.50
    position_epsilon: float = 1e-14
    forced_retarget_on_ineligible_current_position: bool = True
    initial_nonzero_desired_action: str = "SWITCH"

    def validate(self) -> None:
        if self.minimum_hold_hours < 1:
            raise StatefulActionError("minimum_hold_hours must be >=1")
        if not 0.0 <= self.hold_l1_threshold < self.switch_l1_threshold:
            raise StatefulActionError("invalid L1 thresholds")
        if not 0.0 < self.partial_fraction < 1.0:
            raise StatefulActionError("partial_fraction must be in (0,1)")
        if self.position_epsilon <= 0:
            raise StatefulActionError("position_epsilon must be positive")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "contract_id": self.contract_id,
            "minimum_hold_hours": self.minimum_hold_hours,
            "hold_l1_threshold": self.hold_l1_threshold,
            "switch_l1_threshold": self.switch_l1_threshold,
            "partial_fraction": self.partial_fraction,
            "position_epsilon": self.position_epsilon,
            "forced_retarget_on_ineligible_current_position": self.forced_retarget_on_ineligible_current_position,
            "initial_nonzero_desired_action": self.initial_nonzero_desired_action,
            "state": ["current_target_weights", "hours_since_last_trade"],
            "actions": ["HOLD", "PARTIAL", "SWITCH", "FORCED_SWITCH"],
            "reward_cost_semantics": "ACTUAL_ACCOUNTING_COST_IS_CHARGED_ON_REALIZED_TARGET_DELTA_FROM_PREVIOUS_EQUITY_DRIFTED_POSITION",
        }



def apply_stateful_action_policy_batch(
    desired_weights: np.ndarray,
    *,
    eligible: np.ndarray | bool = True,
    contract: StatefulActionContract | None = None,
    initial_weights: np.ndarray | None = None,
    initial_hours_since_trade: int | np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    """Batch-equivalent HOLD/PARTIAL/SWITCH controller.

    This is numerically authority-preserving rather than an approximate fast
    path: each candidate still uses NumPy's float64 reductions, but the
    candidate axis is vectorized so the Python loop is only over time.
    """
    contract = contract or StatefulActionContract()
    contract.validate()
    desired = np.asarray(desired_weights, dtype=np.float64)
    if desired.ndim != 3:
        raise StatefulActionError("desired_weights must be [batch,time,asset]")
    batch, times, assets = desired.shape
    elig = np.broadcast_to(np.asarray(eligible, dtype=bool), desired.shape)
    out = np.zeros_like(desired)

    if initial_weights is None:
        current = np.zeros((batch, assets), dtype=np.float64)
    else:
        raw_initial = np.asarray(initial_weights, dtype=np.float64)
        if raw_initial.shape == (assets,):
            current = np.broadcast_to(raw_initial, (batch, assets)).copy()
        elif raw_initial.shape == (batch, assets):
            current = raw_initial.copy()
        else:
            raise StatefulActionError("initial_weights shape mismatch")

    if initial_hours_since_trade is None:
        age = np.full(batch, contract.minimum_hold_hours, dtype=np.int64)
    else:
        raw_age = np.asarray(initial_hours_since_trade, dtype=np.int64)
        if raw_age.ndim == 0:
            age = np.full(batch, int(raw_age), dtype=np.int64)
        elif raw_age.shape == (batch,):
            age = raw_age.copy()
        else:
            raise StatefulActionError("initial_hours_since_trade shape mismatch")

    count_matrix = np.zeros((batch, 4), dtype=np.int64)
    desired_sums = np.zeros(batch, dtype=np.float64)
    realized_sums = np.zeros(batch, dtype=np.float64)
    eps = contract.position_epsilon

    for t in range(times):
        legal = elig[:, t, :]
        target = np.where(legal & np.isfinite(desired[:, t, :]), desired[:, t, :], 0.0)
        prev = current.copy()
        illegal_current = np.any((np.abs(current) > eps) & ~legal, axis=1)
        current_active = np.any(np.abs(current) > eps, axis=1)
        target_active = np.any(np.abs(target) > eps, axis=1)
        desired_l1 = np.sum(np.abs(target - current), axis=1)
        desired_sums += desired_l1

        forced = illegal_current & contract.forced_retarget_on_ineligible_current_position
        initial = (~forced) & (~current_active) & target_active
        minimum_hold = (~forced) & (~initial) & (age < contract.minimum_hold_hours)
        hold = (~forced) & (~initial) & (~minimum_hold) & (desired_l1 < contract.hold_l1_threshold)
        partial = (
            (~forced)
            & (~initial)
            & (~minimum_hold)
            & (~hold)
            & (desired_l1 < contract.switch_l1_threshold)
        )
        switch = (~forced) & (~initial) & (~minimum_hold) & (~hold) & (~partial)

        if np.any(forced):
            current[forced] = target[forced]
            age[forced] = 0
            count_matrix[forced, 3] += 1
        if np.any(initial):
            current[initial] = target[initial]
            age[initial] = 0
            count_matrix[initial, 2] += 1
        if np.any(minimum_hold):
            age[minimum_hold] += 1
            count_matrix[minimum_hold, 0] += 1
        if np.any(hold):
            age[hold] += 1
            count_matrix[hold, 0] += 1
        if np.any(partial):
            current[partial] = current[partial] + contract.partial_fraction * (
                target[partial] - current[partial]
            )
            current[partial] = np.where(legal[partial], current[partial], 0.0)
            age[partial] = 0
            count_matrix[partial, 1] += 1
        if np.any(switch):
            current[switch] = target[switch]
            age[switch] = 0
            count_matrix[switch, 2] += 1

        illegal_after = np.any((np.abs(current) > eps) & ~legal, axis=1)
        if np.any(illegal_after):
            i = int(np.flatnonzero(illegal_after)[0])
            raise StatefulActionError(f"nonzero state on ineligible coordinate batch={i} t={t}")
        gross = np.sum(np.abs(current), axis=1)
        if np.any(gross > 1.0 + 1e-10):
            i = int(np.flatnonzero(gross > 1.0 + 1e-10)[0])
            raise StatefulActionError(f"gross exposure exceeds 1 batch={i} t={t}: {gross[i]}")

        realized_sums += np.sum(np.abs(current - prev), axis=1)
        out[:, t, :] = current

    action_names = ("HOLD", "PARTIAL", "SWITCH", "FORCED_SWITCH")
    action_counts = {
        name: int(sum(int(v) for v in count_matrix[:, k]))
        for k, name in enumerate(action_names)
    }
    desired_total = float(sum(float(v) for v in desired_sums))
    realized_total = float(sum(float(v) for v in realized_sums))
    diagnostics = {
        "contract": contract.to_dict(),
        "action_counts": action_counts,
        "per_candidate_action_counts": count_matrix.copy(),
        "desired_target_l1_delta_sum": desired_total,
        "realized_target_l1_delta_sum": realized_total,
        "per_candidate_desired_target_l1_delta_sum": desired_sums.copy(),
        "per_candidate_realized_target_l1_delta_sum": realized_sums.copy(),
        "realized_vs_desired_l1_ratio": (realized_total / desired_total) if desired_total > 0 else 0.0,
        "time_count": times,
        "asset_count": assets,
        "batch_count": batch,
        "future_outcome_inputs_used": False,
    }
    return out, diagnostics, current.copy(), age.copy()



def apply_stateful_action_policy(
    desired_weights: np.ndarray,
    *,
    eligible: np.ndarray | bool = True,
    contract: StatefulActionContract | None = None,
    initial_weights: np.ndarray | None = None,
    initial_hours_since_trade: int | None = None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, int]:
    """Convert stateless desired targets into a causal HOLD/PARTIAL/SWITCH path.

    The controller never sees forward returns, funding outcomes, NAV, reward, or
    any future signal.  Decisions at t use only desired_weights[t], current
    target state carried from t-1, eligibility[t], and elapsed holding time.

    Returns (realized_target_path, diagnostics, final_weights, final_age).
    """
    contract = contract or StatefulActionContract()
    contract.validate()
    desired = np.asarray(desired_weights, dtype=np.float64)
    if desired.ndim != 2:
        raise StatefulActionError("desired_weights must be [time,asset]")
    times, assets = desired.shape
    elig = np.broadcast_to(np.asarray(eligible, dtype=bool), desired.shape)
    out = np.zeros_like(desired)
    if initial_weights is None:
        current = np.zeros(assets, dtype=np.float64)
    else:
        current = np.asarray(initial_weights, dtype=np.float64).copy()
        if current.shape != (assets,):
            raise StatefulActionError("initial_weights shape mismatch")
    age = int(contract.minimum_hold_hours if initial_hours_since_trade is None else initial_hours_since_trade)
    counts = {"HOLD": 0, "PARTIAL": 0, "SWITCH": 0, "FORCED_SWITCH": 0}
    l1_desired_delta_sum = 0.0
    realized_l1_delta_sum = 0.0
    eps = contract.position_epsilon

    for t in range(times):
        legal = elig[t]
        target = np.where(legal & np.isfinite(desired[t]), desired[t], 0.0)
        prev = current.copy()
        illegal_current = bool(np.any((np.abs(current) > eps) & ~legal))
        current_active = bool(np.any(np.abs(current) > eps))
        target_active = bool(np.any(np.abs(target) > eps))
        desired_l1 = float(np.sum(np.abs(target - current)))
        l1_desired_delta_sum += desired_l1

        if illegal_current and contract.forced_retarget_on_ineligible_current_position:
            current = target
            action = "FORCED_SWITCH"
            age = 0
        elif not current_active and target_active:
            current = target
            action = "SWITCH"
            age = 0
        elif age < contract.minimum_hold_hours:
            action = "HOLD"
            age += 1
        elif desired_l1 < contract.hold_l1_threshold:
            action = "HOLD"
            age += 1
        elif desired_l1 < contract.switch_l1_threshold:
            current = current + contract.partial_fraction * (target - current)
            current[~legal] = 0.0
            action = "PARTIAL"
            age = 0
        else:
            current = target
            action = "SWITCH"
            age = 0

        if np.any((np.abs(current) > eps) & ~legal):
            raise StatefulActionError(f"nonzero state on ineligible coordinate at t={t}")
        gross = float(np.sum(np.abs(current)))
        if gross > 1.0 + 1e-10:
            raise StatefulActionError(f"gross exposure exceeds 1 at t={t}: {gross}")
        realized_l1_delta_sum += float(np.sum(np.abs(current - prev)))
        out[t] = current
        counts[action] += 1

    diagnostics = {
        "contract": contract.to_dict(),
        "action_counts": counts,
        "desired_target_l1_delta_sum": l1_desired_delta_sum,
        "realized_target_l1_delta_sum": realized_l1_delta_sum,
        "realized_vs_desired_l1_ratio": (realized_l1_delta_sum / l1_desired_delta_sum) if l1_desired_delta_sum > 0 else 0.0,
        "time_count": times,
        "asset_count": assets,
        "future_outcome_inputs_used": False,
    }
    return out, diagnostics, current.copy(), age
