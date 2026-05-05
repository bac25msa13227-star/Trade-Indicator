#!/usr/bin/env python3
"""Unit tests for execution mode enum."""

import pytest
from src.xauusd_ai.execution.mode import ExecutionMode


def test_execution_mode_values():
    """Test enum values are correct."""
    assert ExecutionMode.LIVE.value == "live"
    assert ExecutionMode.PAPER.value == "paper"
    assert ExecutionMode.BACKTEST.value == "backtest"


def test_execution_mode_str():
    """Test string conversion."""
    assert str(ExecutionMode.LIVE) == "live"
    assert str(ExecutionMode.PAPER) == "paper"
    assert str(ExecutionMode.BACKTEST) == "backtest"


def test_is_live_property():
    """Test is_live property."""
    assert ExecutionMode.LIVE.is_live == True
    assert ExecutionMode.PAPER.is_live == False
    assert ExecutionMode.BACKTEST.is_live == False


def test_is_simulated_property():
    """Test is_simulated property."""
    assert ExecutionMode.LIVE.is_simulated == False
    assert ExecutionMode.PAPER.is_simulated == True
    assert ExecutionMode.BACKTEST.is_simulated == True


def test_from_string_valid():
    """Test creating mode from valid string."""
    assert ExecutionMode.from_string("live") == ExecutionMode.LIVE
    assert ExecutionMode.from_string("LIVE") == ExecutionMode.LIVE
    assert ExecutionMode.from_string("paper") == ExecutionMode.PAPER
    assert ExecutionMode.from_string("PAPER") == ExecutionMode.PAPER
    assert ExecutionMode.from_string("backtest") == ExecutionMode.BACKTEST


def test_from_string_invalid():
    """Test creating mode from invalid string raises error."""
    with pytest.raises(ValueError, match="Invalid execution mode"):
        ExecutionMode.from_string("invalid")
    
    with pytest.raises(ValueError, match="Invalid execution mode"):
        ExecutionMode.from_string("demo")


def test_enum_members():
    """Test all enum members are present."""
    members = list(ExecutionMode)
    assert len(members) == 3
    assert ExecutionMode.LIVE in members
    assert ExecutionMode.PAPER in members
    assert ExecutionMode.BACKTEST in members


def test_enum_equality():
    """Test enum equality checks."""
    assert ExecutionMode.LIVE == ExecutionMode.LIVE
    assert ExecutionMode.PAPER != ExecutionMode.LIVE
    assert ExecutionMode.from_string("paper") == ExecutionMode.PAPER


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
