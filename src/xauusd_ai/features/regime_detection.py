"""
Regime Detection for XAUUSD using HMM and volatility metrics.
Detects trending, sideway, and volatile market regimes.

Usage:
    from xauusd_ai.features.regime_detection import add_regime_features
    df = add_regime_features(df)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def add_regime_features(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
    """
    Add regime detection features to dataframe.
    
    Regimes:
    - trending: ADX > 25, ATR stable
    - sideway: ADX < 20, low volatility
    - volatile: ATR spike, high volume
    
    Args:
        df: DataFrame with OHLC and indicators
        lookback: Window for rolling calculations
        
    Returns:
        DataFrame with regime columns added
    """
    df = df.copy()
    
    # 1. ADX-based regime (requires ADX in df)
    if 'adx_h4' in df.columns:
        df['regime_trending'] = (df['adx_h4'] > 25).astype(int)
        df['regime_sideway'] = (df['adx_h4'] < 20).astype(int)
    else:
        # Fallback: use price momentum
        df['price_momentum'] = df['close'].pct_change(20)
        df['regime_trending'] = (df['price_momentum'].abs() > 0.02).astype(int)
        df['regime_sideway'] = (df['price_momentum'].abs() < 0.01).astype(int)
    
    # 2. Volatility regime (requires ATR)
    if 'atr_h4' in df.columns:
        atr_ma = df['atr_h4'].rolling(lookback).mean()
        atr_std = df['atr_h4'].rolling(lookback).std()
        
        # Volatile: ATR > mean + 1.5*std
        df['regime_volatile'] = (df['atr_h4'] > atr_ma + 1.5 * atr_std).astype(int)
        
        # Calm: ATR < mean - 0.5*std
        df['regime_calm'] = (df['atr_h4'] < atr_ma - 0.5 * atr_std).astype(int)
    else:
        # Fallback: use close range
        high_low_pct = (df['high'] - df['low']) / df['close']
        hl_ma = high_low_pct.rolling(lookback).mean()
        hl_std = high_low_pct.rolling(lookback).std()
        
        df['regime_volatile'] = (high_low_pct > hl_ma + 1.5 * hl_std).astype(int)
        df['regime_calm'] = (high_low_pct < hl_ma - 0.5 * hl_std).astype(int)
    
    # 3. Volume regime (if tick_volume available)
    if 'tick_volume' in df.columns:
        vol_ma = df['tick_volume'].rolling(lookback).mean()
        df['regime_high_volume'] = (df['tick_volume'] > vol_ma * 1.3).astype(int)
        df['regime_low_volume'] = (df['tick_volume'] < vol_ma * 0.7).astype(int)
    
    # 4. Composite regime score (-1 = avoid, 0 = neutral, 1 = good)
    # Good for trading: trending + calm volume + not volatile
    regime_score = 0
    regime_score += df.get('regime_trending', 0) * 1.0  # Trending good
    regime_score -= df.get('regime_sideway', 0) * 1.0   # Sideway bad
    regime_score -= df.get('regime_volatile', 0) * 0.5  # Volatile risky
    
    df['regime_score'] = regime_score
    df['regime_favorable'] = (regime_score > 0.5).astype(int)
    
    return df


def calculate_hmm_regimes(
    df: pd.DataFrame,
    n_regimes: int = 3,
    features: Optional[list] = None
) -> pd.DataFrame:
    """
    Use Hidden Markov Model to detect market regimes.
    
    REQUIRES: hmmlearn package (pip install hmmlearn)
    
    Args:
        df: DataFrame with price data
        n_regimes: Number of hidden states (2-4 typical)
        features: List of feature columns to use, defaults to ['returns', 'volatility']
        
    Returns:
        DataFrame with 'hmm_regime' column (0 to n_regimes-1)
    """
    try:
        from hmmlearn import hmm
    except ImportError:
        print("⚠️  hmmlearn not installed. Run: pip install hmmlearn")
        df['hmm_regime'] = 1  # Default to neutral regime
        return df
    
    df = df.copy()
    
    # Default features: returns and volatility
    if features is None:
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        features = ['returns', 'volatility']
    
    # Prepare feature matrix
    X = df[features].fillna(0).values
    
    # Fit Gaussian HMM
    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=100,
        random_state=42
    )
    
    model.fit(X)
    hidden_states = model.predict(X)
    
    df['hmm_regime'] = hidden_states
    
    # Label regimes by mean return (0=bearish, 1=sideway, 2=bullish)
    regime_means = []
    for regime in range(n_regimes):
        mask = hidden_states == regime
        regime_return = df.loc[mask, 'returns'].mean() if 'returns' in df else 0
        regime_means.append(regime_return)
    
    # Sort regimes by return (lowest to highest)
    regime_order = np.argsort(regime_means)
    regime_map = {old: new for new, old in enumerate(regime_order)}
    df['hmm_regime'] = df['hmm_regime'].map(regime_map)
    
    return df


def should_trade_in_regime(
    regime_score: float,
    regime_favorable: int,
    min_score: float = 0.0
) -> bool:
    """
    Decision function: should we trade in this regime?
    
    Args:
        regime_score: Composite regime score (-1 to 1)
        regime_favorable: Binary favorable flag
        min_score: Minimum score to trade
        
    Returns:
        True if should trade, False if should skip
    """
    return regime_score >= min_score and regime_favorable == 1
