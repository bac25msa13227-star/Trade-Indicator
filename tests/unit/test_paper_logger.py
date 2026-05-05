#!/usr/bin/env python3
"""Unit tests for paper trade logger."""

import pytest
import json
import tempfile
from pathlib import Path
from src.xauusd_ai.execution.paper_logger import PaperTradeLogger


@pytest.fixture
def temp_output_file():
    """Create temporary output file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        temp_path = f.name
    yield temp_path
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


@pytest.fixture
def logger(temp_output_file):
    """Create logger instance with temp file."""
    return PaperTradeLogger(output_path=temp_output_file)


def test_logger_initialization(temp_output_file):
    """Test logger initializes correctly."""
    logger = PaperTradeLogger(output_path=temp_output_file)
    assert logger.output_path == Path(temp_output_file)
    assert logger.trade_counter == 0
    assert logger.output_path.parent.exists()


def test_log_entry_signal(logger, temp_output_file):
    """Test logging entry signal."""
    signal = {
        "side": "buy",
        "entry_price": 2000.0,
        "stop_loss": 1990.0,
        "take_profit": 2050.0,
        "probability": 0.75,
        "risk_fraction": 0.03,
        "lot_size": 0.01,
    }
    
    market_data = {
        "atr": 5.0,
        "atr_mean": 4.8,
        "spread_pips": 0.3,
        "volume_ratio": 1.2,
        "session": "NY",
    }
    
    trade_id = logger.log_entry_signal(signal, predicted_slippage_pips=1.5, market_data=market_data)
    
    assert trade_id == 1
    assert logger.trade_counter == 1
    
    # Verify file content
    with open(temp_output_file) as f:
        entry = json.loads(f.readline())
        assert entry["type"] == "entry"
        assert entry["trade_id"] == 1
        assert entry["side"] == "buy"
        assert entry["entry_price"] == 2000.0
        assert entry["predicted_slippage_pips"] == 1.5
        assert entry["atr"] == 5.0
        assert entry["session"] == "NY"


def test_log_exit(logger, temp_output_file):
    """Test logging exit."""
    logger.log_exit(
        trade_id=1,
        exit_price=2010.0,
        exit_reason="take_profit",
        bars_held=10,
        realized_rr=2.0,
        actual_slippage_pips=1.2,
    )
    
    with open(temp_output_file) as f:
        entry = json.loads(f.readline())
        assert entry["type"] == "exit"
        assert entry["trade_id"] == 1
        assert entry["exit_price"] == 2010.0
        assert entry["exit_reason"] == "take_profit"
        assert entry["realized_rr"] == 2.0
        assert entry["actual_slippage_pips"] == 1.2


def test_log_skip(logger, temp_output_file):
    """Test logging skipped signal."""
    signal = {"side": "sell", "probability": 0.5}
    market_data = {"atr": 3.0}
    
    logger.log_skip(reason="confidence_too_low", signal=signal, market_data=market_data)
    
    with open(temp_output_file) as f:
        entry = json.loads(f.readline())
        assert entry["type"] == "skip"
        assert entry["reason"] == "confidence_too_low"
        assert entry["signal"]["side"] == "sell"


def test_calc_slippage_rr():
    """Test slippage RR calculation."""
    logger = PaperTradeLogger()
    
    # 1.5 pips slippage, 10 pip stop
    slippage_rr = logger._calc_slippage_rr(1.5, 2000.0, 1990.0)
    # 1.5 pips = 0.15, 10 pips = 10.0, RR = 0.15/10.0 = 0.015
    assert 0.014 < slippage_rr < 0.016
    
    # Zero stop distance
    slippage_rr = logger._calc_slippage_rr(1.5, 2000.0, 2000.0)
    assert slippage_rr == 0.0
    
    # None inputs
    slippage_rr = logger._calc_slippage_rr(1.5, None, 1990.0)
    assert slippage_rr == 0.0


def test_multiple_trades(logger, temp_output_file):
    """Test logging multiple trades."""
    # Log 3 entry signals
    for i in range(3):
        signal = {"side": "buy", "entry_price": 2000.0 + i, "stop_loss": 1990.0, "take_profit": 2050.0}
        market_data = {"atr": 5.0, "spread_pips": 0.3}
        trade_id = logger.log_entry_signal(signal, predicted_slippage_pips=1.5, market_data=market_data)
        assert trade_id == i + 1
    
    assert logger.trade_counter == 3
    
    # Verify file has 3 entries
    with open(temp_output_file) as f:
        lines = f.readlines()
        assert len(lines) == 3


def test_get_summary_stats_empty():
    """Test summary stats with no trades."""
    logger = PaperTradeLogger(output_path="nonexistent_file.jsonl")
    stats = logger.get_summary_stats()
    assert stats == {}


def test_get_summary_stats_with_trades(logger, temp_output_file):
    """Test summary stats calculation."""
    # Log 5 trades (3 wins, 2 losses)
    for i in range(5):
        signal = {"side": "buy", "entry_price": 2000.0, "stop_loss": 1990.0, "take_profit": 2050.0}
        market_data = {"atr": 5.0}
        trade_id = logger.log_entry_signal(signal, predicted_slippage_pips=1.5, market_data=market_data)
        
        # Exit: 3 wins (+2R, +1.5R, +3R), 2 losses (-1R, -0.5R)
        if i < 3:
            realized_rr = [2.0, 1.5, 3.0][i]
        else:
            realized_rr = [-1.0, -0.5][i - 3]
        
        logger.log_exit(
            trade_id=trade_id,
            exit_price=2010.0,
            exit_reason="stop_loss" if realized_rr < 0 else "take_profit",
            bars_held=10,
            realized_rr=realized_rr,
            actual_slippage_pips=1.4,
        )
    
    stats = logger.get_summary_stats()
    
    assert stats["total_trades"] == 5
    assert stats["wins"] == 3
    assert stats["losses"] == 2
    assert stats["win_rate"] == 0.6
    # Gross profit: 2 + 1.5 + 3 = 6.5, Gross loss: 1 + 0.5 = 1.5, PF = 6.5/1.5 ≈ 4.33
    assert 4.0 < stats["profit_factor"] < 5.0
    # Total RR: 2 + 1.5 + 3 - 1 - 0.5 = 5
    assert stats["total_rr"] == 5.0
    # Slippage accuracy: predicted 1.5, actual 1.4, error = 0.1
    assert 0.09 < stats["slippage_accuracy_pips"] < 0.11


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
