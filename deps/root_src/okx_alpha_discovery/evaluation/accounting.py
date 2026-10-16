from __future__ import annotations

"""Mode-neutral portfolio accounting contract for M0 evaluator backends.

This layer does not decide how a factor becomes a position. It accepts already
normalized target weights and applies one frozen accounting definition. TS and
XS position mappers must therefore compete under identical accounting.
"""

from dataclasses import dataclass
from typing import Any
import hashlib
import json

import numpy as np


class AccountingError(ValueError):
    pass


@dataclass(frozen=True)
class AccountingContract:
    contract_id: str = "CRYPTO_TARGET_WEIGHT_ACCOUNTING_V0"
    max_gross_exposure: float = 1.0
    gross_tolerance: float = 1e-10
    position_epsilon: float = 1e-14
    require_finite_costs_when_traded: bool = True
    return_timing: str = "TARGET_WEIGHT_AT_DECISION_T_APPLIES_TO_FORWARD_RETURN_T"
    turnover_semantics: str = "ONE_WAY_NOTIONAL_SUM_ABS_TARGET_MINUS_PREVIOUS_NET_EQUITY_DRIFTED_WEIGHT"
    funding_semantics: str = "SUBTRACT_SIGNED_POSITION_TIMES_FUNDING_RATE_ON_EVENT"
    missing_semantics: str = "NONZERO_POSITION_ON_MISSING_OR_INELIGIBLE_COORDINATE_FAILS_CLOSED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "max_gross_exposure": self.max_gross_exposure,
            "gross_tolerance": self.gross_tolerance,
            "position_epsilon": self.position_epsilon,
            "require_finite_costs_when_traded": self.require_finite_costs_when_traded,
            "return_timing": self.return_timing,
            "turnover_semantics": self.turnover_semantics,
            "funding_semantics": self.funding_semantics,
            "missing_semantics": self.missing_semantics,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class AccountingResult:
    gross_return: np.ndarray
    turnover: np.ndarray
    transaction_cost: np.ndarray
    funding_cashflow: np.ndarray
    net_return: np.ndarray
    nav: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "candidate_count": int(self.net_return.shape[0]),
            "period_count": int(self.net_return.shape[1]),
            "net_return_sum": self.net_return.sum(axis=1).tolist(),
            "gross_return_sum": self.gross_return.sum(axis=1).tolist(),
            "turnover_sum": self.turnover.sum(axis=1).tolist(),
            "transaction_cost_sum": self.transaction_cost.sum(axis=1).tolist(),
            "funding_cashflow_sum": self.funding_cashflow.sum(axis=1).tolist(),
            "final_nav": self.nav[:, -1].tolist() if self.nav.shape[1] else [],
        }


def _as_3d_weights(target_weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(target_weights, dtype=np.float64)
    if weights.ndim == 2:
        weights = weights[None, ...]
    if weights.ndim != 3:
        raise AccountingError("target_weights must have shape [candidate,time,asset] or [time,asset]")
    return weights


def _surface(name: str, value: np.ndarray | float, shape: tuple[int, int]) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    try:
        return np.broadcast_to(raw, shape).astype(np.float64, copy=False)
    except ValueError as exc:
        raise AccountingError(f"{name} cannot broadcast to [time,asset]={shape}: {raw.shape}") from exc


def evaluate_target_weights_numpy(
    *,
    target_weights: np.ndarray,
    forward_returns: np.ndarray,
    fee_rate: np.ndarray | float = 0.0,
    slippage_rate: np.ndarray | float = 0.0,
    funding_rate: np.ndarray | float = 0.0,
    funding_event: np.ndarray | bool = False,
    eligible: np.ndarray | bool = True,
    contract: AccountingContract | None = None,
) -> AccountingResult:
    """Reference accounting oracle.

    Rates are decimal fractions of notional, not bps. Positive funding means
    longs pay and shorts receive. Transaction cost is charged on absolute target
    weight change, including initial entry from zero.
    """

    contract = contract or AccountingContract()
    weights = _as_3d_weights(target_weights)
    candidates, times, assets = weights.shape
    returns = np.asarray(forward_returns, dtype=np.float64)
    if returns.shape != (times, assets):
        raise AccountingError(f"forward_returns shape {returns.shape} != {(times, assets)}")
    fees = _surface("fee_rate", fee_rate, (times, assets))
    slippage = _surface("slippage_rate", slippage_rate, (times, assets))
    funding = _surface("funding_rate", funding_rate, (times, assets))
    event = np.broadcast_to(np.asarray(funding_event, dtype=bool), (times, assets))
    eligibility = np.broadcast_to(np.asarray(eligible, dtype=bool), (times, assets))

    if np.any(np.isinf(weights)):
        raise AccountingError("target weights contain infinity")
    gross_exposure = np.nansum(np.abs(weights), axis=2)
    if np.any(gross_exposure > contract.max_gross_exposure + contract.gross_tolerance):
        bad = np.argwhere(gross_exposure > contract.max_gross_exposure + contract.gross_tolerance)[0]
        raise AccountingError(
            f"gross exposure exceeds contract at candidate={int(bad[0])} time={int(bad[1])}: "
            f"{gross_exposure[tuple(bad)]}"
        )

    active = np.abs(weights) > contract.position_epsilon
    illegal_position = active & ~eligibility[None, :, :]
    if np.any(illegal_position):
        c, t, a = np.argwhere(illegal_position)[0]
        raise AccountingError(f"nonzero position on ineligible coordinate c={c} t={t} a={a}")
    missing_return = active & ~np.isfinite(returns)[None, :, :]
    if np.any(missing_return):
        c, t, a = np.argwhere(missing_return)[0]
        raise AccountingError(f"nonzero position on missing forward return c={c} t={t} a={a}")

    all_cost = fees + slippage
    funding_active = active & event[None, :, :]
    missing_funding = funding_active & ~np.isfinite(funding)[None, :, :]
    if np.any(missing_funding):
        c, t, a = np.argwhere(missing_funding)[0]
        raise AccountingError(f"funding event with missing rate c={c} t={t} a={a}")

    safe_returns = np.where(np.isfinite(returns), returns, 0.0)
    safe_cost = np.where(np.isfinite(all_cost), all_cost, 0.0)
    safe_funding = np.where(event & np.isfinite(funding), funding, 0.0)

    gross_return = np.zeros((candidates, times), dtype=np.float64)
    turnover = np.zeros((candidates, times), dtype=np.float64)
    transaction_cost = np.zeros((candidates, times), dtype=np.float64)
    funding_cashflow = np.zeros((candidates, times), dtype=np.float64)
    net_return = np.zeros((candidates, times), dtype=np.float64)
    nav = np.ones((candidates, times), dtype=np.float64)

    # Rebalance sequentially because the next pre-trade portfolio weights are
    # normalized by the previous period's *net* equity after transaction cost
    # and funding, not merely by gross market PnL. Asset positions themselves
    # drift with asset returns; cash costs/funding alter the equity denominator.
    drifted_prev = np.zeros((candidates, assets), dtype=np.float64)
    nav_value = np.ones(candidates, dtype=np.float64)
    for t in range(times):
        delta = weights[:, t, :] - drifted_prev
        turnover_by_asset = np.abs(delta)
        traded = turnover_by_asset > contract.position_epsilon
        illegal_trade = traded & ~eligibility[t][None, :]
        if np.any(illegal_trade):
            c, a = np.argwhere(illegal_trade)[0]
            raise AccountingError(f"trade required on ineligible coordinate c={c} t={t} a={a}")
        if contract.require_finite_costs_when_traded:
            missing_cost = traded & ~np.isfinite(all_cost[t])[None, :]
            if np.any(missing_cost):
                c, a = np.argwhere(missing_cost)[0]
                raise AccountingError(f"trade on missing fee/slippage c={c} t={t} a={a}")

        gross_t = np.sum(weights[:, t, :] * safe_returns[t][None, :], axis=1)
        turnover_t = np.sum(turnover_by_asset, axis=1)
        tx_t = np.sum(turnover_by_asset * safe_cost[t][None, :], axis=1)
        funding_t = np.sum(weights[:, t, :] * safe_funding[t][None, :], axis=1)
        net_t = gross_t - tx_t - funding_t
        if np.any(net_t <= -1.0):
            c = int(np.argwhere(net_t <= -1.0)[0, 0])
            raise AccountingError(f"period net return <= -100% c={c} t={t}: {net_t[c]}")

        nav_value *= 1.0 + net_t
        gross_return[:, t] = gross_t
        turnover[:, t] = turnover_t
        transaction_cost[:, t] = tx_t
        funding_cashflow[:, t] = funding_t
        net_return[:, t] = net_t
        nav[:, t] = nav_value

        denom = 1.0 + net_t
        if np.any(denom <= 0.0):
            c = int(np.argwhere(denom <= 0.0)[0, 0])
            raise AccountingError(f"previous net portfolio equity non-positive c={c} t={t}")
        drifted_prev = weights[:, t, :] * (1.0 + safe_returns[t][None, :]) / denom[:, None]
    return AccountingResult(
        gross_return=gross_return,
        turnover=turnover,
        transaction_cost=transaction_cost,
        funding_cashflow=funding_cashflow,
        net_return=net_return,
        nav=nav,
    )
