"""
Feature Importance Stability Tracker for Walk-Forward Validation.
Tracks how feature importance changes across folds to detect drift.

Usage:
    from xauusd_ai.monitoring.feature_stability import FeatureStabilityTracker
    
    tracker = FeatureStabilityTracker()
    tracker.log_fold_importance(fold_id=1, feature_importance_dict)
    stability_report = tracker.analyze_stability()
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class FeatureStabilityTracker:
    """
    Track feature importance across walk-forward folds.
    Detect drift and unstable features.
    """
    
    def __init__(self, log_file: str = "outputs/feature_stability.jsonl"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
    def log_fold_importance(
        self,
        fold_id: int,
        feature_importance: Dict[str, float],
        test_period: str = "",
        metadata: Optional[Dict] = None
    ):
        """
        Log feature importance for a fold.
        
        Args:
            fold_id: Fold number
            feature_importance: Dict of {feature_name: importance_score}
            test_period: Date range string (e.g., "2024-01-01 -> 2024-03-31")
            metadata: Optional dict with AUC, win_rate, etc.
        """
        entry = {
            "fold_id": fold_id,
            "timestamp": datetime.now().isoformat(),
            "test_period": test_period,
            "feature_importance": feature_importance,
            "metadata": metadata or {}
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def load_history(self) -> pd.DataFrame:
        """Load all logged feature importance data."""
        if not self.log_file.exists():
            return pd.DataFrame()
        
        records = []
        with open(self.log_file, 'r') as f:
            for line in f:
                records.append(json.loads(line))
        
        return pd.DataFrame(records)
    
    def analyze_stability(self, top_n: int = 20) -> Dict:
        """
        Analyze feature importance stability across folds.
        
        Args:
            top_n: Number of top features to analyze
            
        Returns:
            Dict with stability metrics:
            - stable_features: Features with low variance
            - unstable_features: Features with high variance
            - drifting_features: Features trending up/down
            - top_features_by_avg: Average importance ranking
        """
        df = self.load_history()
        
        if df.empty or len(df) < 3:
            return {
                "error": "Need at least 3 folds to analyze stability",
                "folds_available": len(df)
            }
        
        # Convert feature_importance dicts to DataFrame
        importance_records = []
        for _, row in df.iterrows():
            fold_id = row['fold_id']
            for feature, importance in row['feature_importance'].items():
                importance_records.append({
                    'fold_id': fold_id,
                    'feature': feature,
                    'importance': importance
                })
        
        importance_df = pd.DataFrame(importance_records)
        
        # Calculate statistics per feature
        feature_stats = importance_df.groupby('feature')['importance'].agg([
            ('mean', 'mean'),
            ('std', 'std'),
            ('cv', lambda x: x.std() / x.mean() if x.mean() > 0 else 0),  # Coefficient of variation
            ('min', 'min'),
            ('max', 'max'),
            ('range', lambda x: x.max() - x.min())
        ]).reset_index()
        
        # Rank by average importance
        feature_stats = feature_stats.sort_values('mean', ascending=False)
        
        # Classify stability (CV = coefficient of variation)
        # CV < 0.3: Stable
        # 0.3 <= CV < 0.6: Moderate
        # CV >= 0.6: Unstable
        feature_stats['stability'] = pd.cut(
            feature_stats['cv'],
            bins=[0, 0.3, 0.6, np.inf],
            labels=['stable', 'moderate', 'unstable']
        )
        
        # Top N features
        top_features = feature_stats.head(top_n)
        
        # Detect drift (linear trend in importance)
        drifting = []
        for feature in top_features['feature']:
            feature_data = importance_df[importance_df['feature'] == feature].sort_values('fold_id')
            
            if len(feature_data) >= 3:
                # Simple linear regression: y = ax + b
                x = feature_data['fold_id'].values
                y = feature_data['importance'].values
                
                slope = np.polyfit(x, y, 1)[0]
                
                # Drift if |slope| > 0.01 per fold
                if abs(slope) > 0.01:
                    direction = 'increasing' if slope > 0 else 'decreasing'
                    drifting.append({
                        'feature': feature,
                        'slope': float(slope),
                        'direction': direction,
                        'avg_importance': float(feature_data['importance'].mean())
                    })
        
        return {
            "total_features": len(feature_stats),
            "folds_analyzed": len(df),
            "top_features": top_features.to_dict('records'),
            "stable_features": feature_stats[feature_stats['stability'] == 'stable']['feature'].tolist()[:10],
            "unstable_features": feature_stats[feature_stats['stability'] == 'unstable']['feature'].tolist(),
            "drifting_features": drifting,
            "summary": {
                "stable_count": (feature_stats['stability'] == 'stable').sum(),
                "moderate_count": (feature_stats['stability'] == 'moderate').sum(),
                "unstable_count": (feature_stats['stability'] == 'unstable').sum(),
            }
        }
    
    def plot_top_features(self, top_n: int = 10, output_file: str = "outputs/feature_stability_plot.png"):
        """
        Plot feature importance trends across folds (requires matplotlib).
        
        Args:
            top_n: Number of top features to plot
            output_file: Output PNG file path
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️  matplotlib not installed. Run: pip install matplotlib")
            return
        
        df = self.load_history()
        if df.empty:
            print("⚠️  No data to plot")
            return
        
        # Get top N features by average importance
        analysis = self.analyze_stability(top_n=top_n)
        top_features = [f['feature'] for f in analysis['top_features'][:top_n]]
        
        # Prepare data
        importance_records = []
        for _, row in df.iterrows():
            fold_id = row['fold_id']
            for feature in top_features:
                importance = row['feature_importance'].get(feature, 0)
                importance_records.append({
                    'fold_id': fold_id,
                    'feature': feature,
                    'importance': importance
                })
        
        importance_df = pd.DataFrame(importance_records)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for feature in top_features:
            feature_data = importance_df[importance_df['feature'] == feature]
            ax.plot(feature_data['fold_id'], feature_data['importance'], marker='o', label=feature)
        
        ax.set_xlabel('Fold ID')
        ax.set_ylabel('Feature Importance')
        ax.set_title(f'Top {top_n} Feature Importance Stability Across Folds')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Plot saved to {output_file}")


def get_model_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
    """
    Extract feature importance from LightGBM model.
    
    Args:
        model: Trained LightGBM model
        feature_names: List of feature names
        
    Returns:
        Dict of {feature_name: importance_score}
    """
    try:
        import lightgbm as lgb
        
        if isinstance(model, lgb.Booster):
            importance = model.feature_importance(importance_type='gain')
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            raise ValueError("Model does not have feature_importance method")
        
        # Normalize to sum to 1.0
        importance = importance / importance.sum()
        
        return dict(zip(feature_names, importance))
    
    except Exception as e:
        print(f"⚠️  Error extracting feature importance: {e}")
        return {}
