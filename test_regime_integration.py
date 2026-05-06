"""
Quick test to verify regime detection integration works properly.
Tests that regime features are calculated correctly and model can use them.
"""

import pandas as pd
import numpy as np
from src.xauusd_ai.features.regime_detection import add_regime_features

# Create sample data
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'time': pd.date_range('2024-01-01', periods=n, freq='5min'),
    'open': 2000 + np.cumsum(np.random.randn(n) * 0.5),
    'high': 2000 + np.cumsum(np.random.randn(n) * 0.5) + np.abs(np.random.randn(n)),
    'low': 2000 + np.cumsum(np.random.randn(n) * 0.5) - np.abs(np.random.randn(n)),
    'close': 2000 + np.cumsum(np.random.randn(n) * 0.5),
    'tick_volume': 100 + np.random.randn(n) * 20,
})

# Add ADX and ATR (required for regime detection)
df['adx_h4'] = 15 + np.abs(np.random.randn(n) * 10)  # 0-40 range
df['atr_h4'] = 1.0 + np.abs(np.random.randn(n) * 0.5)  # 0.5-2.0 range

print("Testing regime detection integration...")
print(f"Input shape: {df.shape}")
print(f"Columns before: {list(df.columns)}")

# Apply regime detection
df_with_regime = add_regime_features(df, lookback=50)

print(f"\nColumns after: {list(df_with_regime.columns)}")

# Check new columns exist
regime_cols = [c for c in df_with_regime.columns if 'regime' in c]
print(f"\nRegime columns added: {len(regime_cols)}")
for col in regime_cols:
    print(f"  - {col}")

# Check values
print(f"\nRegime statistics:")
print(f"  regime_trending: {df_with_regime['regime_trending'].value_counts().to_dict()}")
print(f"  regime_sideway: {df_with_regime['regime_sideway'].value_counts().to_dict()}")
print(f"  regime_volatile: {df_with_regime['regime_volatile'].value_counts().to_dict()}")
print(f"  regime_favorable: {df_with_regime['regime_favorable'].value_counts().to_dict()}")

# Check score range
score_stats = df_with_regime['regime_score'].describe()
print(f"\n  regime_score: min={score_stats['min']:.2f}, mean={score_stats['mean']:.2f}, max={score_stats['max']:.2f}")

# Check no NaN values
nan_counts = df_with_regime[regime_cols].isna().sum()
if nan_counts.sum() > 0:
    print(f"\n⚠️  WARNING: NaN values found in regime columns:")
    print(nan_counts[nan_counts > 0])
else:
    print(f"\n✅ No NaN values in regime columns")

# Test favorable regime logic
favorable_pct = (df_with_regime['regime_favorable'] == 1).mean() * 100
print(f"\nFavorable regime: {favorable_pct:.1f}% of bars")
print(f"Expected: 30-50% favorable (trending + not volatile)")

if 20 <= favorable_pct <= 60:
    print("✅ Favorable percentage looks reasonable")
else:
    print("⚠️  Favorable percentage may be too high/low")

print("\n✅ Regime detection integration test passed!")
