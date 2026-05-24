"""
Minimum Profit Filter

Skips trades with expected profit below transaction costs (spread + swap).
Use this to prevent high-frequency trading with small wins that lose money
after spread costs.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)


class MinimumProfitFilter:
    """
    Filter trades based on minimum expected profit threshold.
    
    Example:
        - Spread cost per trade: $10 (2× × 0.5 pips × $10)
        - Minimum expected profit: $15 (must exceed spread + buffer)
        - Skip any trade with predicted profit < $15
    
    This prevents "death by a thousand cuts" where many small wins
    are wiped out by transaction costs.
    """
    
    def __init__(
        self,
        min_expected_profit: float = 15.0,
        min_expected_profit_r: float = 0.0,
        spread_pips: float = 0.5,
        pip_value: float = 10.0,
        enabled: bool = True,
    ):
        """
        Initialize minimum profit filter.
        
        Args:
            min_expected_profit: Minimum $ profit per trade (default $15)
            min_expected_profit_r: Minimum expected profit in R. When >0, this
                takes precedence over min_expected_profit.
            spread_pips: Average bid-ask spread in pips (default 0.5)
            pip_value: $ per pip for 1.0 lot (default $10 for XAUUSD)
            enabled: Whether filter is active (default True)
        """
        self.min_expected_profit = min_expected_profit
        self.min_expected_profit_r = min_expected_profit_r
        self.spread_pips = spread_pips
        self.pip_value = pip_value
        self.enabled = enabled
        
        # Calculate actual spread cost per trade
        self.spread_cost_per_trade = 2 * spread_pips * pip_value  # Entry + exit
        
        LOGGER.info(
            f"MinimumProfitFilter initialized: "
            f"min_profit=${min_expected_profit:.2f}, "
            f"min_profit_r={min_expected_profit_r:.2f}R, "
            f"spread_cost=${self.spread_cost_per_trade:.2f}, "
            f"enabled={enabled}"
        )
    
    def should_skip_trade(
        self,
        predicted_profit: float,
        predicted_rr: Optional[float] = None,
        predicted_profit_r: Optional[float] = None,
        lot_size: float = 1.0,
    ) -> tuple[bool, str]:
        """
        Determine if trade should be skipped based on profit threshold.
        
        Args:
            predicted_profit: Expected profit in $ (from model or strategy)
            predicted_rr: Risk-reward ratio (optional, for logging)
            predicted_profit_r: Expected profit in R. Required when
                min_expected_profit_r is enabled.
            lot_size: Position size in lots (default 1.0)
        
        Returns:
            (should_skip, reason) tuple
            - should_skip: True if trade should be skipped
            - reason: String explanation (for logging)
        """
        if not self.enabled:
            return False, "filter_disabled"

        if self.min_expected_profit_r > 0:
            if predicted_profit_r is None:
                return True, "profit_r_missing"
            if predicted_profit_r <= self.min_expected_profit_r:
                reason = (
                    f"profit_r_too_low: predicted={predicted_profit_r:.2f}R "
                    f"< min={self.min_expected_profit_r:.2f}R"
                )
                if predicted_rr is not None:
                    reason += f" | RR={predicted_rr:.2f}"

                LOGGER.debug(reason)
                return True, reason

            return False, "ok"
        
        # Adjust for lot size
        effective_min_profit = self.min_expected_profit * lot_size
        effective_spread_cost = self.spread_cost_per_trade * lot_size
        
        # Skip if predicted profit doesn't exceed minimum (use strict >)
        if predicted_profit <= effective_min_profit:
            reason = (
                f"profit_too_low: predicted=${predicted_profit:.2f} "
                f"< min=${effective_min_profit:.2f} "
                f"(spread=${effective_spread_cost:.2f})"
            )
            if predicted_rr is not None:
                reason += f" | RR={predicted_rr:.2f}"
            
            LOGGER.debug(reason)
            return True, reason
        
        return False, "ok"
    
    def filter_signals(
        self,
        signals_df: pd.DataFrame,
        predicted_profit_col: str = "predicted_profit",
        predicted_profit_r_col: str = "predicted_profit_r",
        predicted_rr_col: str = "predicted_rr",
        lot_size_col: str = "lot_size",
    ) -> tuple[pd.DataFrame, dict]:
        """
        Filter DataFrame of trading signals.
        
        Args:
            signals_df: DataFrame with trade signals
            predicted_profit_col: Column name for predicted profit
            predicted_profit_r_col: Column name for predicted profit in R
            predicted_rr_col: Column name for predicted RR
            lot_size_col: Column name for lot size (default 1.0 if not present)
        
        Returns:
            (filtered_df, stats) tuple
            - filtered_df: Signals that pass the filter
            - stats: Dictionary with filter statistics
        """
        if not self.enabled:
            return signals_df, {
                "total_signals": len(signals_df),
                "filtered_signals": len(signals_df),
                "skipped_signals": 0,
                "skip_rate": 0.0,
            }
        
        required_profit_col = (
            predicted_profit_r_col
            if self.min_expected_profit_r > 0
            else predicted_profit_col
        )
        if required_profit_col not in signals_df.columns:
            LOGGER.warning(
                f"Column '{required_profit_col}' not found in signals_df. "
                f"Returning all signals."
            )
            return signals_df, {
                "total_signals": len(signals_df),
                "filtered_signals": len(signals_df),
                "skipped_signals": 0,
                "skip_rate": 0.0,
            }
        
        # Add lot_size column if missing (default 1.0)
        if lot_size_col not in signals_df.columns:
            signals_df = signals_df.copy()
            signals_df[lot_size_col] = 1.0
        
        total_signals = len(signals_df)
        
        # Apply filter row by row
        keep_mask = []
        for _, row in signals_df.iterrows():
            predicted_profit = row.get(predicted_profit_col, 0.0)
            predicted_profit_r = row.get(predicted_profit_r_col, None)
            predicted_rr = row.get(predicted_rr_col, None)
            lot_size = row.get(lot_size_col, 1.0)
            
            should_skip, _ = self.should_skip_trade(
                predicted_profit=predicted_profit,
                predicted_rr=predicted_rr,
                predicted_profit_r=predicted_profit_r,
                lot_size=lot_size,
            )
            keep_mask.append(not should_skip)
        
        filtered_df = signals_df[keep_mask].copy()
        skipped_signals = total_signals - len(filtered_df)
        
        stats = {
            "total_signals": total_signals,
            "filtered_signals": len(filtered_df),
            "skipped_signals": skipped_signals,
            "skip_rate": skipped_signals / total_signals if total_signals > 0 else 0.0,
        }
        
        LOGGER.info(
            f"Profit filter stats: {skipped_signals}/{total_signals} skipped "
            f"({stats['skip_rate']:.1%})"
        )
        
        return filtered_df, stats
    
    def update_config(
        self,
        min_expected_profit: Optional[float] = None,
        min_expected_profit_r: Optional[float] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """
        Update filter configuration dynamically.
        
        Args:
            min_expected_profit: New minimum profit threshold
            min_expected_profit_r: New minimum R threshold
            enabled: Enable/disable filter
        """
        if min_expected_profit is not None:
            old_value = self.min_expected_profit
            self.min_expected_profit = min_expected_profit
            LOGGER.info(
                f"Updated min_expected_profit: ${old_value:.2f} → ${min_expected_profit:.2f}"
            )
        
        if min_expected_profit_r is not None:
            old_value = self.min_expected_profit_r
            self.min_expected_profit_r = min_expected_profit_r
            LOGGER.info(
                f"Updated min_expected_profit_r: {old_value:.2f}R -> {min_expected_profit_r:.2f}R"
            )

        if enabled is not None:
            old_state = self.enabled
            self.enabled = enabled
            LOGGER.info(f"Filter {'enabled' if enabled else 'disabled'} (was {old_state})")
