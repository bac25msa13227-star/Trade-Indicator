"""Paper trading logger for shadow execution and validation."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from xauusd_ai.learning.decision_log import DecisionLog
from xauusd_ai.strategies.rating import rating_from_decision


class PaperTradeLogger:
    """
    Logs paper trades for comparison with live performance.
    
    Paper trading mode generates signals and logs all decisions without
    placing real orders. This allows:
    - Zero-risk testing of new strategies
    - Validation of slippage model accuracy
    - Comparison of predicted vs actual P&L
    """
    
    def __init__(
        self,
        output_path: str = "outputs/paper_trades.jsonl",
        decision_log: DecisionLog | None = None,
        enable_decision_log: bool = True,
    ):
        """
        Initialize paper trade logger.
        
        Args:
            output_path: Path to JSONL file for logging trades
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.trade_counter = 0
        self.decision_log = decision_log if enable_decision_log else None
        if self.decision_log is None and enable_decision_log:
            self.decision_log = DecisionLog(output_dir=self.output_path.parent)
    
    def log_entry_signal(
        self,
        signal: Dict[str, Any],
        predicted_slippage_pips: float,
        market_data: Dict[str, Any],
    ) -> int:
        """
        Log paper trade entry signal.
        
        Args:
            signal: Trade signal dict with entry_price, stop_loss, take_profit, side, etc.
            predicted_slippage_pips: Predicted slippage from dynamic model
            market_data: Current market conditions (ATR, spread, volume, session)
        
        Returns:
            Trade ID for tracking
        """
        self.trade_counter += 1
        trade_id = self.trade_counter
        
        rating = signal.get("rating") or rating_from_decision(
            signal.get("side", "hold"),
            float(signal.get("confidence", signal.get("probability", 0.0)) or 0.0),
            should_trade=True,
        )

        entry = {
            "type": "entry",
            "trade_id": trade_id,
            "timestamp": datetime.utcnow().isoformat(),
            "side": signal.get("side"),
            "rating": rating,
            "entry_price": signal.get("entry_price"),
            "stop_loss": signal.get("stop_loss"),
            "take_profit": signal.get("take_profit"),
            "predicted_slippage_pips": predicted_slippage_pips,
            "predicted_slippage_rr": self._calc_slippage_rr(
                predicted_slippage_pips,
                signal.get("entry_price"),
                signal.get("stop_loss"),
            ),
            "atr": market_data.get("atr"),
            "atr_mean": market_data.get("atr_mean"),
            "spread_pips": market_data.get("spread_pips"),
            "volume_ratio": market_data.get("volume_ratio"),
            "session": market_data.get("session"),
            "probability": signal.get("probability"),
            "risk_fraction": signal.get("risk_fraction"),
            "lot_size": signal.get("lot_size"),
        }
        
        self._write_entry(entry)
        if self.decision_log is not None:
            decision_signal = {**signal, "rating": rating}
            self.decision_log.log_entry(trade_id, decision_signal, market_data)
        return trade_id
    
    def log_exit(
        self,
        trade_id: int,
        exit_price: float,
        exit_reason: str,
        bars_held: int,
        realized_rr: float,
        actual_slippage_pips: Optional[float] = None,
    ) -> None:
        """
        Log paper trade exit.
        
        Args:
            trade_id: ID from log_entry_signal
            exit_price: Actual exit price
            exit_reason: Reason for exit (stop_loss, take_profit, trailing, etc.)
            bars_held: Number of bars position was held
            realized_rr: Realized return in R (risk multiples)
            actual_slippage_pips: Actual slippage observed (if available)
        """
        entry = {
            "type": "exit",
            "trade_id": trade_id,
            "timestamp": datetime.utcnow().isoformat(),
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "bars_held": bars_held,
            "realized_rr": realized_rr,
            "actual_slippage_pips": actual_slippage_pips,
        }
        
        self._write_entry(entry)
        if self.decision_log is not None:
            self.decision_log.log_exit(
                trade_id=trade_id,
                exit_price=exit_price,
                exit_reason=exit_reason,
                bars_held=bars_held,
                realized_rr=realized_rr,
                actual_slippage_pips=actual_slippage_pips,
            )
    
    def log_skip(
        self,
        reason: str,
        signal: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log skipped trade signal and reason.
        
        Args:
            reason: Why signal was skipped (confidence too low, risk limits, etc.)
            signal: Signal that was skipped (optional)
            market_data: Market conditions at time of skip (optional)
        """
        entry = {
            "type": "skip",
            "timestamp": datetime.utcnow().isoformat(),
            "reason": reason,
            "signal": signal,
            "market_data": market_data,
        }
        
        self._write_entry(entry)
        if self.decision_log is not None:
            enriched_signal = None
            if signal is not None:
                try:
                    enriched_signal = {
                        **signal,
                        "rating": signal.get("rating") or rating_from_decision(
                            signal.get("side", "hold"),
                            float(signal.get("confidence", signal.get("probability", 0.0)) or 0.0),
                            should_trade=False,
                        ),
                    }
                except Exception:
                    enriched_signal = signal
            self.decision_log.log_skip(reason=reason, signal=enriched_signal, market_data=market_data)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Calculate summary statistics from paper trades.
        
        Returns:
            Dict with win_rate, profit_factor, avg_rr, slippage_accuracy, etc.
        """
        if not self.output_path.exists():
            return {}
        
        # Load all entries
        entries = []
        with open(self.output_path) as f:
            for line in f:
                entries.append(json.loads(line))
        
        # Separate entry and exit events
        entry_events = [e for e in entries if e.get("type") == "entry"]
        exit_events = [e for e in entries if e.get("type") == "exit"]
        
        if not entry_events or not exit_events:
            return {"total_signals": len(entry_events)}
        
        # Match entry/exit pairs
        trades = []
        for exit_event in exit_events:
            trade_id = exit_event["trade_id"]
            entry_event = next(
                (e for e in entry_events if e["trade_id"] == trade_id), None
            )
            if entry_event:
                trades.append({**entry_event, **exit_event})
        
        if not trades:
            return {"total_signals": len(entry_events)}
        
        # Calculate stats
        df = pd.DataFrame(trades)
        
        wins = len(df[df["realized_rr"] > 0])
        losses = len(df[df["realized_rr"] < 0])
        total = len(df)
        
        win_rate = wins / total if total > 0 else 0
        
        gross_profit = df[df["realized_rr"] > 0]["realized_rr"].sum()
        gross_loss = abs(df[df["realized_rr"] < 0]["realized_rr"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Slippage accuracy (predicted vs actual)
        slippage_entries = df[df["actual_slippage_pips"].notna()]
        if len(slippage_entries) > 0:
            slippage_error = (
                slippage_entries["predicted_slippage_pips"]
                - slippage_entries["actual_slippage_pips"]
            ).abs().mean()
        else:
            slippage_error = None
        
        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_rr": df["realized_rr"].mean(),
            "total_rr": df["realized_rr"].sum(),
            "avg_bars_held": df["bars_held"].mean(),
            "slippage_accuracy_pips": slippage_error,
        }
    
    def _calc_slippage_rr(
        self, slippage_pips: float, entry_price: float, stop_loss: float
    ) -> float:
        """Convert slippage from pips to R (risk multiples)."""
        if entry_price is None or stop_loss is None:
            return 0.0
        
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance == 0:
            return 0.0
        
        # Convert pips to price (XAUUSD: 1 pip = 0.10)
        slippage_price = slippage_pips * 0.10
        
        # Convert to RR
        slippage_rr = slippage_price / stop_distance
        return slippage_rr
    
    def _write_entry(self, entry: Dict[str, Any]) -> None:
        """Write entry to JSONL file."""
        with open(self.output_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
