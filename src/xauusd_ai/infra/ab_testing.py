"""
A/B Testing Framework for XAUUSD AI Bot

Enables controlled experiments comparing different trading strategies or parameters
(e.g., static vs dynamic slippage) with statistical rigor.

Key features:
- Deterministic treatment assignment (signal_id -> control/treatment)
- JSONL logging for append-only, crash-safe persistence
- Statistical analysis (t-test, confidence intervals, Cohen's d effect size)
- Thread-safe concurrent logging

Usage:
    ab_manager = ABTestManager(log_file="outputs/ab_test_results.jsonl", seed=42)
    
    # Assign treatment
    treatment = ab_manager.assign_treatment("signal_001")
    
    # Apply treatment-specific logic
    if treatment == "control":
        slippage = calculate_static_slippage()
    else:
        slippage = calculate_dynamic_slippage()
    
    # Log outcome
    ab_manager.log_result("signal_001", treatment, {
        "entry_price": 2650.50,
        "exit_price": 2655.00,
        "pnl": 4.50,
        "slippage_rr": slippage
    })
    
    # Analyze results
    analysis = ab_manager.analyze()
    print(f"P-value: {analysis['p_value']}")
"""

import json
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timezone
from typing import Literal, Dict, Any
import numpy as np
from scipy import stats


TreatmentType = Literal["control", "treatment"]


class ABTestManager:
    """
    Manages A/B testing experiments for trading strategies.
    
    Thread-safe manager for assigning treatments, logging outcomes, and
    analyzing statistical significance of experiment results.
    """
    
    def __init__(self, log_file: str, seed: int = 42):
        """
        Initialize A/B test manager.
        
        Args:
            log_file: Path to JSONL log file for storing results
            seed: Random seed for reproducible treatment assignment
        """
        self.log_file = Path(log_file)
        self.seed = seed
        self._lock = threading.Lock()
        
        # Create parent directory if needed
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def assign_treatment(self, signal_id: str) -> TreatmentType:
        """
        Assign treatment to a signal deterministically.
        
        Uses MD5 hash of signal_id + seed to ensure:
        1. Same signal_id always gets same treatment
        2. ~50/50 split across many signals
        3. Reproducible with same seed
        
        Args:
            signal_id: Unique identifier for the trading signal
        
        Returns:
            "control" or "treatment"
        """
        # Hash signal_id with seed for deterministic assignment
        hash_input = f"{signal_id}_{self.seed}".encode("utf-8")
        hash_digest = hashlib.md5(hash_input).hexdigest()
        
        # Use first byte of hash to determine treatment (0-127 = control, 128-255 = treatment)
        first_byte = int(hash_digest[:2], 16)
        
        if first_byte < 128:
            return "control"
        else:
            return "treatment"
    
    def log_result(
        self,
        signal_id: str,
        treatment: TreatmentType,
        outcome: Dict[str, Any]
    ) -> None:
        """
        Log experiment result to JSONL file.
        
        Thread-safe append operation. Creates file if it doesn't exist.
        
        Args:
            signal_id: Unique signal identifier
            treatment: "control" or "treatment"
            outcome: Dictionary with experiment outcome (must include "pnl")
        """
        entry = {
            "signal_id": signal_id,
            "treatment": treatment,
            "outcome": outcome,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Thread-safe file append
        with self._lock:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics from logged results.
        
        Returns:
            Dictionary with counts and aggregated PnL per treatment group
        """
        if not self.log_file.exists():
            return {
                "total_signals": 0,
                "control_count": 0,
                "treatment_count": 0,
                "control_total_pnl": 0.0,
                "treatment_total_pnl": 0.0
            }
        
        control_pnls = []
        treatment_pnls = []
        
        with open(self.log_file, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                pnl = entry["outcome"].get("pnl", 0.0)
                
                if entry["treatment"] == "control":
                    control_pnls.append(pnl)
                else:
                    treatment_pnls.append(pnl)
        
        return {
            "total_signals": len(control_pnls) + len(treatment_pnls),
            "control_count": len(control_pnls),
            "treatment_count": len(treatment_pnls),
            "control_total_pnl": sum(control_pnls),
            "treatment_total_pnl": sum(treatment_pnls),
            "control_mean_pnl": np.mean(control_pnls) if control_pnls else 0.0,
            "treatment_mean_pnl": np.mean(treatment_pnls) if treatment_pnls else 0.0
        }
    
    def analyze(self) -> Dict[str, Any]:
        """
        Perform statistical analysis on experiment results.
        
        Calculates:
        - Two-sample t-test p-value
        - 95% confidence interval for difference in means
        - Cohen's d effect size
        
        Requires >= 30 samples per group for valid statistical inference.
        
        Returns:
            Dictionary with statistical analysis results
        """
        if not self.log_file.exists():
            return {
                "error": "No data logged yet"
            }
        
        # Load all PnL values per treatment
        control_pnls = []
        treatment_pnls = []
        
        with open(self.log_file, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                pnl = entry["outcome"].get("pnl", 0.0)
                
                if entry["treatment"] == "control":
                    control_pnls.append(pnl)
                else:
                    treatment_pnls.append(pnl)
        
        # Check for sufficient data
        if len(control_pnls) < 30 or len(treatment_pnls) < 30:
            return {
                "error": "Insufficient data (need >= 30 per group)",
                "control_count": len(control_pnls),
                "treatment_count": len(treatment_pnls)
            }
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(treatment_pnls, control_pnls)
        
        # Cohen's d effect size
        control_mean = np.mean(control_pnls)
        treatment_mean = np.mean(treatment_pnls)
        control_std = np.std(control_pnls, ddof=1)
        treatment_std = np.std(treatment_pnls, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(
            ((len(control_pnls) - 1) * control_std**2 +
             (len(treatment_pnls) - 1) * treatment_std**2) /
            (len(control_pnls) + len(treatment_pnls) - 2)
        )
        
        # Handle zero variance case (all values identical)
        # Use small epsilon to avoid division by zero
        if pooled_std < 1e-10:
            # If both groups have zero variance, effect size is undefined or infinite
            # Return large value if means differ, else 0
            cohens_d = 999.0 if abs(treatment_mean - control_mean) > 1e-6 else 0.0
        else:
            cohens_d = (treatment_mean - control_mean) / pooled_std
        
        # 95% confidence interval for difference in means
        se_diff = np.sqrt(
            np.var(control_pnls, ddof=1) / len(control_pnls) +
            np.var(treatment_pnls, ddof=1) / len(treatment_pnls)
        )
        mean_diff = treatment_mean - control_mean
        ci_margin = 1.96 * se_diff  # z-score for 95% CI
        
        return {
            "p_value": float(p_value),
            "t_statistic": float(t_stat),
            "effect_size": float(cohens_d),
            "control_mean": float(control_mean),
            "treatment_mean": float(treatment_mean),
            "mean_difference": float(mean_diff),
            "confidence_interval": {
                "lower": float(mean_diff - ci_margin),
                "upper": float(mean_diff + ci_margin)
            },
            "control_count": len(control_pnls),
            "treatment_count": len(treatment_pnls),
            "interpretation": self._interpret_results(p_value, cohens_d, mean_diff)
        }
    
    @staticmethod
    def _interpret_results(p_value: float, cohens_d: float, mean_diff: float) -> str:
        """
        Provide human-readable interpretation of statistical results.
        
        Args:
            p_value: Two-tailed t-test p-value
            cohens_d: Cohen's d effect size
            mean_diff: Difference in means (treatment - control)
        
        Returns:
            Interpretation string
        """
        # Significance
        if p_value < 0.01:
            sig = "highly significant (p < 0.01)"
        elif p_value < 0.05:
            sig = "significant (p < 0.05)"
        else:
            sig = "not significant (p >= 0.05)"
        
        # Effect size (Cohen's d thresholds: 0.2=small, 0.5=medium, 0.8=large)
        if abs(cohens_d) < 0.2:
            effect = "negligible effect"
        elif abs(cohens_d) < 0.5:
            effect = "small effect"
        elif abs(cohens_d) < 0.8:
            effect = "medium effect"
        else:
            effect = "large effect"
        
        # Direction
        direction = "outperforms" if mean_diff > 0 else "underperforms"
        
        return f"Treatment {direction} control ({sig}, {effect}, Cohen's d={cohens_d:.2f})"
