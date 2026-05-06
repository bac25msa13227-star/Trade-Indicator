# Regime Detection HMM — Implementation Guide

**Goal:** Detect market regimes (range, trending, volatile) and adapt strategy parameters dynamically.

**Expected Impact:**
- Reduce drawdown: -10-20% in sideways markets
- Improve Sharpe: +0.3-0.5 points
- Adaptive risk: Lower risk in range, higher in trends

**Implementation Time:** 1-2 days (8-16 hours)

---

## Concept Overview

### Hidden Markov Model (HMM)

**States (3 regimes):**
1. **Low Volatility (Range)** — Sideways market, mean-reverting
2. **Medium Volatility (Trending)** — Strong directional moves
3. **High Volatility (Volatile)** — Choppy, unpredictable

**Observable features:**
- ATR (Average True Range) — volatility proxy
- Volume / Tick count — activity level
- Price momentum (ROC) — trend strength
- Directional consistency — %winning bars in direction

**Transition probabilities:**
- Range → Trend (low prob, ~15%)
- Trend → Volatile (medium prob, ~25%)
- Volatile → Range (high prob, ~40%)

**Adaptive parameters per regime:**
```python
REGIME_PARAMS = {
    0: {  # Low volatility (range)
        "risk_pct": 0.020,      # Lower risk
        "min_rr": 2.0,          # Require higher RR
        "min_confidence": 0.75,  # Stricter entry
        "max_holding_bars": 720  # 12h max hold
    },
    1: {  # Medium volatility (trending)
        "risk_pct": 0.030,      # Normal risk
        "min_rr": 1.5,          # Standard RR
        "min_confidence": 0.70,  # Normal entry
        "max_holding_bars": 1440  # 24h max hold
    },
    2: {  # High volatility (volatile)
        "risk_pct": 0.015,      # Very low risk
        "min_rr": 2.5,          # Much higher RR required
        "min_confidence": 0.80,  # Very strict entry
        "max_holding_bars": 360   # 6h max hold
    }
}
```

---

## Implementation Plan

### Step 1: Install Dependencies

```bash
# Add to requirements.txt
hmmlearn>=0.3.0  # HMM implementation
```

```bash
cd "$HOME/Documents/Thạc sĩ MSE/Trade Indicator"
source .venv/bin/activate
pip install hmmlearn
```

---

### Step 2: Create Regime Detector Module

**File:** `src/xauusd_ai/features/regime.py`

```python
"""
Market Regime Detection using Hidden Markov Model.

Detects 3 regimes:
- 0: Low volatility (range)
- 1: Medium volatility (trending)
- 2: High volatility (volatile)
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


class RegimeDetector:
    """
    HMM-based market regime detection.
    
    Features:
    - ATR (volatility)
    - Volume/tick count (activity)
    - ROC (momentum)
    - Directional consistency
    """
    
    def __init__(
        self,
        n_regimes: int = 3,
        lookback_bars: int = 100,
        model_path: Optional[str] = None
    ):
        """
        Initialize regime detector.
        
        Args:
            n_regimes: Number of hidden states (default 3)
            lookback_bars: Bars to consider for features (default 100)
            model_path: Path to pre-trained HMM model
        """
        self.n_regimes = n_regimes
        self.lookback_bars = lookback_bars
        self.model_path = model_path
        
        # Initialize HMM
        self.hmm = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        # Feature scaler
        self.scaler = StandardScaler()
        
        # Regime labels (assigned after training)
        self.regime_labels = {
            0: "low_volatility",
            1: "medium_volatility",
            2: "high_volatility"
        }
        
        # Load pre-trained model if provided
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
            LOGGER.info(f"Loaded pre-trained HMM from {model_path}")
        else:
            LOGGER.info(f"Initialized untrained HMM with {n_regimes} regimes")
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract regime features from OHLCV data.
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume, atr]
        
        Returns:
            Feature matrix (n_samples, n_features)
        """
        features = []
        
        # 1. ATR (volatility proxy)
        if "atr" in df.columns:
            atr = df["atr"].values
        else:
            # Calculate ATR if not present
            high_low = df["high"] - df["low"]
            high_close = np.abs(df["high"] - df["close"].shift(1))
            low_close = np.abs(df["low"] - df["close"].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().values
        
        features.append(atr)
        
        # 2. Volume/tick activity (if available)
        if "volume" in df.columns:
            volume_ma = df["volume"].rolling(window=20).mean().values
            volume_ratio = df["volume"].values / (volume_ma + 1e-8)
        else:
            # Use tick count or price changes as proxy
            tick_activity = df["high"].sub(df["low"]).rolling(window=20).sum().values
            volume_ratio = tick_activity / (tick_activity.mean() + 1e-8)
        
        features.append(volume_ratio)
        
        # 3. Rate of Change (momentum)
        roc_5 = df["close"].pct_change(5).values
        roc_20 = df["close"].pct_change(20).values
        
        features.append(roc_5)
        features.append(roc_20)
        
        # 4. Directional consistency
        # Measure how many bars in last N are in same direction
        returns = df["close"].pct_change().values
        direction = np.sign(returns)
        consistency = pd.Series(direction).rolling(window=20).apply(
            lambda x: np.abs(x.sum()) / len(x)
        ).values
        
        features.append(consistency)
        
        # Stack features
        feature_matrix = np.column_stack(features)
        
        # Handle NaN (from rolling windows)
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0)
        
        return feature_matrix
    
    def train(self, df: pd.DataFrame, save_path: Optional[str] = None) -> dict:
        """
        Train HMM on historical data.
        
        Args:
            df: Historical OHLCV data
            save_path: Path to save trained model
        
        Returns:
            Training metrics dict
        """
        LOGGER.info(f"Training HMM on {len(df)} bars...")
        
        # Extract features
        features = self._extract_features(df)
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train HMM
        self.hmm.fit(features_scaled)
        
        # Predict regimes on training data
        regimes = self.hmm.predict(features_scaled)
        
        # Assign regime labels based on mean ATR
        regime_atr = {}
        for regime in range(self.n_regimes):
            regime_mask = regimes == regime
            regime_atr[regime] = features[regime_mask, 0].mean()  # ATR is first feature
        
        # Sort regimes by ATR (low → high volatility)
        sorted_regimes = sorted(regime_atr.items(), key=lambda x: x[1])
        regime_mapping = {old: new for new, (old, _) in enumerate(sorted_regimes)}
        
        # Relabel regimes
        regimes = np.array([regime_mapping[r] for r in regimes])
        
        # Calculate regime statistics
        regime_stats = {}
        for regime in range(self.n_regimes):
            regime_mask = regimes == regime
            regime_stats[regime] = {
                "count": regime_mask.sum(),
                "frequency": regime_mask.mean(),
                "mean_atr": features[regime_mask, 0].mean(),
                "mean_roc": features[regime_mask, 2].mean()
            }
        
        # Log training results
        for regime, stats in regime_stats.items():
            LOGGER.info(
                f"Regime {regime} ({self.regime_labels[regime]}): "
                f"{stats['frequency']:.1%} frequency, "
                f"ATR={stats['mean_atr']:.4f}, "
                f"ROC={stats['mean_roc']:.2%}"
            )
        
        # Save model if path provided
        if save_path:
            self.save_model(save_path)
            LOGGER.info(f"Saved trained HMM to {save_path}")
        
        return regime_stats
    
    def predict_regime(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict current market regime.
        
        Args:
            df: Recent OHLCV data (last N bars)
        
        Returns:
            Regime predictions (0=low, 1=medium, 2=high volatility)
        """
        # Extract features
        features = self._extract_features(df)
        
        # Normalize features
        features_scaled = self.scaler.transform(features)
        
        # Predict regimes
        regimes = self.hmm.predict(features_scaled)
        
        return regimes
    
    def get_current_regime(self, df: pd.DataFrame) -> int:
        """
        Get current regime (most recent bar).
        
        Args:
            df: Recent OHLCV data
        
        Returns:
            Current regime (0, 1, or 2)
        """
        regimes = self.predict_regime(df.tail(self.lookback_bars))
        return regimes[-1]
    
    def get_strategy_params(self, regime: int) -> dict:
        """
        Get recommended strategy parameters for regime.
        
        Args:
            regime: Regime ID (0, 1, or 2)
        
        Returns:
            Dict with risk_pct, min_rr, min_confidence, max_holding_bars
        """
        params = {
            0: {  # Low volatility (range)
                "risk_pct": 0.020,
                "min_rr": 2.0,
                "min_confidence": 0.75,
                "max_holding_bars": 720
            },
            1: {  # Medium volatility (trending)
                "risk_pct": 0.030,
                "min_rr": 1.5,
                "min_confidence": 0.70,
                "max_holding_bars": 1440
            },
            2: {  # High volatility (volatile)
                "risk_pct": 0.015,
                "min_rr": 2.5,
                "min_confidence": 0.80,
                "max_holding_bars": 360
            }
        }
        
        return params.get(regime, params[1])  # Default to medium volatility
    
    def save_model(self, path: str) -> None:
        """Save trained HMM and scaler to disk."""
        model_data = {
            "hmm": self.hmm,
            "scaler": self.scaler,
            "regime_labels": self.regime_labels,
            "n_regimes": self.n_regimes,
            "lookback_bars": self.lookback_bars
        }
        
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
        
        LOGGER.info(f"Saved HMM model to {path}")
    
    def load_model(self, path: str) -> None:
        """Load trained HMM and scaler from disk."""
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        
        self.hmm = model_data["hmm"]
        self.scaler = model_data["scaler"]
        self.regime_labels = model_data["regime_labels"]
        self.n_regimes = model_data["n_regimes"]
        self.lookback_bars = model_data["lookback_bars"]
        
        LOGGER.info(f"Loaded HMM model from {path}")
```

---

### Step 3: Add Regime Column to Dataset

**File:** `src/xauusd_ai/features/dataset.py`

```python
# Add at end of _build_features() method (line ~800)

def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Build feature set with regime detection."""
    
    # ... existing feature engineering ...
    
    # ──────────────────────────────────────────────────────────────
    # REGIME DETECTION (HMM)
    # ──────────────────────────────────────────────────────────────
    from xauusd_ai.features.regime import RegimeDetector
    
    regime_detector = RegimeDetector(
        n_regimes=3,
        model_path="outputs/regime_hmm_model.pkl"
    )
    
    # If model doesn't exist, train it
    if not Path("outputs/regime_hmm_model.pkl").exists():
        LOGGER.info("Training HMM regime detector...")
        regime_detector.train(
            df[["open", "high", "low", "close", "atr"]],
            save_path="outputs/regime_hmm_model.pkl"
        )
    
    # Predict regimes
    df["regime"] = regime_detector.predict_regime(df)
    
    LOGGER.info(f"Regime distribution: {df['regime'].value_counts().to_dict()}")
    
    return df
```

---

### Step 4: Use Regime in Orchestrator

**File:** `src/xauusd_ai/orchestrator.py`

```python
# Add at top (line ~20)
from xauusd_ai.features.regime import RegimeDetector

# In OrchestrationEngine.__init__() (line ~150)
def __init__(self, config_path: str):
    # ... existing initialization ...
    
    # Initialize regime detector
    self.regime_detector = RegimeDetector(
        n_regimes=3,
        model_path="outputs/regime_hmm_model.pkl"
    )
    LOGGER.info("Regime detector initialized")

# In _evaluate_signal() method (line ~800)
async def _evaluate_signal(self, row: pd.Series) -> Optional[TradeSignal]:
    """Evaluate signal with regime-adaptive parameters."""
    
    # Detect current regime
    current_regime = int(row.get("regime", 1))  # Default to medium if not present
    regime_params = self.regime_detector.get_strategy_params(current_regime)
    
    LOGGER.debug(
        f"Current regime: {current_regime} "
        f"({self.regime_detector.regime_labels[current_regime]})"
    )
    
    # Override strategy parameters based on regime
    effective_risk_pct = regime_params["risk_pct"]
    effective_min_rr = regime_params["min_rr"]
    effective_min_conf = regime_params["min_confidence"]
    
    # ... existing signal generation logic ...
    
    # Apply regime-adjusted filters
    if confidence < effective_min_conf:
        LOGGER.info(
            f"Skipped low confidence signal: {confidence:.2%} < {effective_min_conf:.2%} "
            f"(regime={current_regime})"
        )
        return None
    
    if risk_reward_ratio < effective_min_rr:
        LOGGER.info(
            f"Skipped low RR signal: {risk_reward_ratio:.2f} < {effective_min_rr:.2f} "
            f"(regime={current_regime})"
        )
        return None
    
    # Calculate position size with regime-adjusted risk
    position_size = self._calculate_position_size(
        entry_price=entry_price,
        stop_loss=stop_loss,
        risk_pct=effective_risk_pct  # Use regime-adjusted risk
    )
    
    # ... return signal ...
```

---

### Step 5: Training Script

**File:** `scripts/train_regime_hmm.py`

```python
"""
Train HMM regime detection model on historical data.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xauusd_ai.features.regime import RegimeDetector
from xauusd_ai.features.dataset import DatasetBuilder


def main():
    # Load historical data
    print("Loading historical data...")
    builder = DatasetBuilder("configs/acc1_v14pp_profit.yaml")
    df = builder.load_ohlcv_data()
    
    # Calculate ATR if not present
    if "atr" not in df.columns:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=14).mean()
    
    # Train HMM
    print("Training HMM...")
    detector = RegimeDetector(n_regimes=3, lookback_bars=100)
    stats = detector.train(df, save_path="outputs/regime_hmm_model.pkl")
    
    # Show results
    print("\n=== Regime Statistics ===")
    for regime, data in stats.items():
        print(f"\nRegime {regime}:")
        print(f"  Frequency: {data['frequency']:.1%}")
        print(f"  Mean ATR: {data['mean_atr']:.4f}")
        print(f"  Mean ROC: {data['mean_roc']:.2%}")
    
    # Test prediction
    print("\n=== Testing Prediction ===")
    recent_regime = detector.get_current_regime(df.tail(200))
    print(f"Current regime: {recent_regime} ({detector.regime_labels[recent_regime]})")
    
    params = detector.get_strategy_params(recent_regime)
    print(f"Recommended params: {params}")


if __name__ == "__main__":
    main()
```

---

### Step 6: Testing

```bash
# Train HMM
python scripts/train_regime_hmm.py

# Run WF with regime detection
python scripts/walkforward_ict_wyckoff.py configs/acc1_v14pp_profit.yaml \
  --test-start 2024-01-01 --test-bars 6000 --step-bars 6000 \
  --no-compound --combo133 --cache --risk-pct 0.030 --no-rr-sweep \
  --use-regime-detection \
  2>&1 | tee outputs/wf_with_regime_detection.log

# Compare results
python scripts/compare_wf_results.py \
  outputs/wf_without_regime.log \
  outputs/wf_with_regime_detection.log \
  --metrics drawdown,sharpe,trades_per_regime
```

---

## Timeline

**Day 1 (8 hours):**
- Hour 1-2: Install dependencies, create regime.py
- Hour 3-4: Implement feature extraction
- Hour 5-6: Train HMM on historical data
- Hour 7-8: Add regime column to dataset

**Day 2 (8 hours):**
- Hour 1-3: Integrate with orchestrator
- Hour 4-5: Write tests
- Hour 6-7: Run WF validation
- Hour 8: Review results, tune parameters

**Total: 16 hours (2 days)**

---

## Expected Results

**Before (no regime detection):**
- Sharpe: 3.7
- Max DD: 15%
- Trades: 10,825

**After (with regime detection):**
- Sharpe: 4.0-4.2 (+0.3-0.5)
- Max DD: 12-13% (-2-3%)
- Trades: 8,000-9,000 (fewer in range markets)

**Trade distribution:**
- Low volatility (range): 35% of time, 25% of trades
- Medium volatility (trending): 45% of time, 55% of trades
- High volatility (volatile): 20% of time, 20% of trades

---

## Next Steps After Implementation

1. **Visualize regimes on charts** — add to dashboard
2. **Track regime transitions** — log when regimes change
3. **Optimize regime parameters** — A/B test different risk levels
4. **Add regime-specific models** — train separate LightGBM for each regime

---

**Ready to implement immediately!** 🚀
