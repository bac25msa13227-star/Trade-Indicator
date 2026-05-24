from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

AgentAction = Literal["TAKE", "SKIP", "REDUCE", "BOOST"]


@dataclass(frozen=True)
class AgentDecision:
    action: AgentAction
    risk_multiplier: float
    thesis: str
    memory_note: str
    risk_note: str


class TradingAgentDecisionGate:
    """
    Deterministic TradingAgents-style gate.

    It uses only information available before each trade:
    - market perspective: side, confidence bucket, session
    - memory perspective: prior-fold outcomes for similar buckets
    - risk perspective: current fold drawdown path
    """

    def __init__(
        self,
        *,
        min_memory_trades: int = 25,
        bad_expectancy_usd: float = -3.0,
        good_expectancy_usd: float = 7.5,
        bad_win_rate: float = 0.42,
        reduce_multiplier: float = 0.50,
        boost_multiplier: float = 1.20,
        dd_reduce_pct: float = -8.0,
        dd_skip_pct: float = -14.0,
    ) -> None:
        self.min_memory_trades = int(min_memory_trades)
        self.bad_expectancy_usd = float(bad_expectancy_usd)
        self.good_expectancy_usd = float(good_expectancy_usd)
        self.bad_win_rate = float(bad_win_rate)
        self.reduce_multiplier = float(reduce_multiplier)
        self.boost_multiplier = float(boost_multiplier)
        self.dd_reduce_pct = float(dd_reduce_pct)
        self.dd_skip_pct = float(dd_skip_pct)

    @staticmethod
    def session_from_time(value: object) -> str:
        ts = pd.to_datetime(str(value), errors="coerce")
        if pd.isna(ts):
            return "unknown"
        hour = int(ts.hour)
        if 0 <= hour < 7:
            return "asia"
        if 7 <= hour < 13:
            return "london"
        if 13 <= hour < 21:
            return "ny"
        return "rollover"

    @staticmethod
    def confidence_bucket(probability: object) -> str:
        prob = pd.to_numeric(pd.Series([probability]), errors="coerce").iloc[0]
        if pd.isna(prob):
            return "unknown"
        if float(prob) < 0.55:
            return "low"
        if float(prob) < 0.70:
            return "mid"
        return "high"

    def decide(
        self,
        *,
        side: str,
        probability: float | None,
        open_time: object,
        prior_trades: pd.DataFrame,
        current_dd_pct: float,
    ) -> AgentDecision:
        session = self.session_from_time(open_time)
        bucket = self.confidence_bucket(probability)
        side_norm = str(side).strip().lower()
        thesis = f"market={side_norm} confidence_bucket={bucket} session={session}"

        if current_dd_pct <= self.dd_skip_pct:
            return AgentDecision(
                action="SKIP",
                risk_multiplier=0.0,
                thesis=thesis,
                memory_note="memory bypassed because fold risk state is already stressed",
                risk_note=f"skip: current_dd={current_dd_pct:.2f}% <= {self.dd_skip_pct:.2f}%",
            )

        risk_cap = 1.0
        risk_note = "risk ok"
        if current_dd_pct <= self.dd_reduce_pct:
            risk_cap = self.reduce_multiplier
            risk_note = f"reduce: current_dd={current_dd_pct:.2f}% <= {self.dd_reduce_pct:.2f}%"

        similar = self._similar_prior(prior_trades, side_norm, session, bucket)
        memory_note = "insufficient similar prior trades"
        action: AgentAction = "TAKE"
        memory_mult = 1.0

        if len(similar) >= self.min_memory_trades:
            expectancy = float(similar["pnl"].mean())
            win_rate = float((similar["pnl"] > 0).mean())
            memory_note = f"similar_n={len(similar)} expectancy=${expectancy:.2f} win_rate={win_rate:.1%}"
            if expectancy <= self.bad_expectancy_usd and win_rate <= self.bad_win_rate:
                action = "SKIP"
                memory_mult = 0.0
            elif expectancy <= 0.0:
                action = "REDUCE"
                memory_mult = self.reduce_multiplier
            elif expectancy >= self.good_expectancy_usd and win_rate >= 0.50:
                action = "BOOST"
                memory_mult = self.boost_multiplier

        final_mult = min(memory_mult, risk_cap) if memory_mult <= 1.0 or risk_cap < 1.0 else memory_mult
        if final_mult <= 0.0:
            return AgentDecision("SKIP", 0.0, thesis, memory_note, risk_note)
        if final_mult < 1.0:
            return AgentDecision("REDUCE", final_mult, thesis, memory_note, risk_note)
        if final_mult > 1.0:
            return AgentDecision("BOOST", final_mult, thesis, memory_note, risk_note)
        return AgentDecision(action="TAKE", risk_multiplier=1.0, thesis=thesis, memory_note=memory_note, risk_note=risk_note)

    @staticmethod
    def _similar_prior(prior_trades: pd.DataFrame, side: str, session: str, bucket: str) -> pd.DataFrame:
        if prior_trades.empty:
            return prior_trades
        frame = prior_trades
        exact = frame[
            (frame["side_key"] == side)
            & (frame["session_key"] == session)
            & (frame["confidence_bucket"] == bucket)
        ]
        if len(exact) >= 10:
            return exact
        # Back off to side + confidence bucket when exact session has too few samples.
        return frame[(frame["side_key"] == side) & (frame["confidence_bucket"] == bucket)]
