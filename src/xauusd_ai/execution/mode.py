"""Execution mode enum for XAU/USD trading system."""

from enum import Enum


class ExecutionMode(Enum):
    """Trading execution modes."""
    
    LIVE = "live"           # Real execution with MT5
    PAPER = "paper"         # Shadow logging only, no real orders
    BACKTEST = "backtest"   # Historical simulation
    
    def __str__(self) -> str:
        return self.value
    
    @property
    def is_live(self) -> bool:
        """Check if mode involves real execution."""
        return self == ExecutionMode.LIVE
    
    @property
    def is_simulated(self) -> bool:
        """Check if mode is simulated (paper or backtest)."""
        return self in (ExecutionMode.PAPER, ExecutionMode.BACKTEST)
    
    @classmethod
    def from_string(cls, value: str) -> "ExecutionMode":
        """Create ExecutionMode from string."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(
                f"Invalid execution mode: {value}. "
                f"Valid modes: {', '.join(m.value for m in cls)}"
            )
