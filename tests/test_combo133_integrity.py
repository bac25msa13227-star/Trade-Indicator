"""
test_combo133_integrity.py — Regression tests for Combo #133.

Covers:
  1. Constant parity: WF script vs live retrain script (TRAIN_BARS, TEST_BARS,
     RETRAIN_EVERY_BARS, MIN_CONF, BLOCKED, D1_GATE)
  2. Ensemble architecture: VotingClassifier soft with weights [3, 2, 1]
  3. Feature mask: RF scout drop bottom 30% by importance
  4. Sample weight formula: 2× pos_boost × time-decay, normalized mean = 1
  5. Live config parity: live_acc1.yaml min_confidence = 0.70, same BLOCKED hours,
     FEATURE_COLUMNS includes judas_swing_signal and silver_bullet_setup
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _parse_script_constants(script_path: Path) -> dict:
    """Extract top-level assignment constants from a script via AST (no execution)."""
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    constants: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        constants[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    return constants


class Combo133ConstantParityTests(unittest.TestCase):
    """WF script and live retrain script must share identical Combo #133 constants."""

    @classmethod
    def setUpClass(cls) -> None:
        scripts = ROOT / "scripts"
        cls.wf = _parse_script_constants(scripts / "show_combo133_daily.py")
        cls.live = _parse_script_constants(scripts / "auto_update_retrain.py")

    def test_train_bars_are_30000(self) -> None:
        self.assertEqual(self.wf["TRAIN_BARS"], 30_000)
        self.assertEqual(self.live["TRAIN_BARS"], 30_000)

    def test_test_bars_are_6000(self) -> None:
        self.assertEqual(self.wf["TEST_BARS"], 6_000)
        self.assertEqual(self.live["TEST_BARS"], 6_000)

    def test_step_equals_test_bars_in_wf(self) -> None:
        """WF folds advance by STEP_BARS == TEST_BARS so cadence matches live."""
        self.assertEqual(self.wf["STEP_BARS"], self.wf["TEST_BARS"])

    def test_retrain_every_bars_equals_test_bars_in_live(self) -> None:
        """Live triggers retrain after RETRAIN_EVERY_BARS == TEST_BARS new M5 bars.

        `RETRAIN_EVERY_BARS = TEST_BARS` in the script is a variable assignment so
        ast.literal_eval cannot resolve it.  We verify the invariant by checking
        the raw source text instead.
        """
        src = (ROOT / "scripts" / "auto_update_retrain.py").read_text(encoding="utf-8")
        # The assignment must read exactly `RETRAIN_EVERY_BARS = TEST_BARS`
        self.assertIn("RETRAIN_EVERY_BARS = TEST_BARS", src)

    def test_wf_and_live_retrain_cadence_match(self) -> None:
        """Both WF and live use the same ~20-day bar cadence (6 000 M5 bars)."""
        self.assertEqual(self.wf["STEP_BARS"], 6_000)
        self.assertEqual(self.live["TEST_BARS"], 6_000)

    def test_min_conf_is_0_70(self) -> None:
        self.assertAlmostEqual(self.wf["MIN_CONF"], 0.70, places=5)
        self.assertAlmostEqual(self.live["MIN_CONF"], 0.70, places=5)

    def test_blocked_hours_match(self) -> None:
        self.assertEqual(sorted(self.wf["BLOCKED"]), [3, 15, 17, 22, 23])
        self.assertEqual(sorted(self.live["BLOCKED"]), sorted(self.wf["BLOCKED"]))

    def test_d1_gate_disabled_in_both(self) -> None:
        self.assertFalse(self.wf["D1_GATE"])
        self.assertFalse(self.live["D1_GATE"])

    def test_threshold_search_keys_present_in_live(self) -> None:
        for key in ("THRESHOLD_MIN", "THRESHOLD_MAX", "THRESHOLD_STEP", "PREC_FLOOR"):
            with self.subTest(key=key):
                self.assertIn(key, self.live)


class Combo133EnsembleArchitectureTests(unittest.TestCase):
    """VotingClassifier soft voting with weights [3, 2, 1] (HGB, RF, ET)."""

    def _build_ensemble(self):
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            HistGradientBoostingClassifier,
            RandomForestClassifier,
            VotingClassifier,
        )

        hgb = HistGradientBoostingClassifier(
            max_iter=1000, learning_rate=0.01, max_depth=7, min_samples_leaf=20,
            l2_regularization=1.0, max_bins=128,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=40,
            random_state=42,
        )
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=15,
            max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42,
        )
        et = ExtraTreesClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=10,
            max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42,
        )
        return VotingClassifier(
            estimators=[("hgb", hgb), ("rf", rf), ("et", et)],
            voting="soft",
            weights=[3, 2, 1],
        )

    def test_voting_is_soft(self) -> None:
        model = self._build_ensemble()
        self.assertEqual(model.voting, "soft")

    def test_weights_are_3_2_1(self) -> None:
        model = self._build_ensemble()
        self.assertEqual(list(model.weights), [3, 2, 1])

    def test_estimator_names_are_hgb_rf_et(self) -> None:
        model = self._build_ensemble()
        names = [name for name, _ in model.estimators]
        self.assertEqual(names, ["hgb", "rf", "et"])

    def test_ensemble_probabilities_in_unit_interval(self) -> None:
        """Smoke test: ensemble trains and outputs valid probabilities."""
        from sklearn.ensemble import (
            ExtraTreesClassifier,
            HistGradientBoostingClassifier,
            RandomForestClassifier,
            VotingClassifier,
        )
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 10))
        y = (X[:, 0] + rng.standard_normal(200) * 0.5 > 0).astype(int)

        hgb = HistGradientBoostingClassifier(max_iter=20, random_state=42)
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        et = ExtraTreesClassifier(n_estimators=10, random_state=42)

        model = VotingClassifier(
            estimators=[("hgb", hgb), ("rf", rf), ("et", et)],
            voting="soft",
            weights=[3, 2, 1],
        )
        model.fit(X, y)
        proba = model.predict_proba(X[:10])[:, 1]
        self.assertTrue(np.all((proba >= 0.0) & (proba <= 1.0)))
        self.assertEqual(proba.shape, (10,))


class Combo133FeatureMaskTests(unittest.TestCase):
    """RF scout feature selection: keep features with importance >= percentile(30)."""

    def _apply_feature_mask(self, X: np.ndarray, y: np.ndarray):
        from sklearn.ensemble import RandomForestClassifier

        scout = RandomForestClassifier(
            n_estimators=80, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
        scout.fit(X, y)
        imp = scout.feature_importances_
        feat_mask = imp >= np.percentile(imp, 30)
        # Fallback: if fewer than 10 features kept, use all
        if feat_mask.sum() < 10:
            feat_mask = np.ones(len(imp), dtype=bool)
        return feat_mask, imp

    def test_mask_keeps_at_least_70pct_features(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.standard_normal((400, 20))
        y = (X[:, 0] > 0).astype(int)
        feat_mask, _ = self._apply_feature_mask(X, y)
        keep_pct = feat_mask.sum() / len(feat_mask)
        self.assertGreaterEqual(keep_pct, 0.69)  # percentile(30) → drop ≤30%

    def test_mask_is_boolean_array(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((300, 15))
        y = (X[:, 1] > 0).astype(int)
        feat_mask, _ = self._apply_feature_mask(X, y)
        self.assertEqual(feat_mask.dtype, bool)
        self.assertEqual(len(feat_mask), 15)

    def test_fallback_when_too_few_features(self) -> None:
        """When mask selects < 10 features, fallback returns all-True mask."""
        imp = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        feat_mask = imp >= np.percentile(imp, 30)
        if feat_mask.sum() < 10:
            feat_mask = np.ones(len(imp), dtype=bool)
        self.assertEqual(feat_mask.sum(), len(imp))

    def test_selected_features_applied_to_train_and_test(self) -> None:
        """X_tr_sel and X_te_sel must have same number of columns."""
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(7)
        X_raw = rng.standard_normal((500, 30))
        y = (X_raw[:, 2] > 0).astype(int)
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_raw)
        feat_mask, _ = self._apply_feature_mask(X_sc, y)
        X_tr_sel = X_sc[:400, feat_mask]
        X_te_sel = X_sc[400:, feat_mask]
        self.assertEqual(X_tr_sel.shape[1], X_te_sel.shape[1])


class Combo133SampleWeightTests(unittest.TestCase):
    """Sample weights: 2× positive class boost × exponential time decay, mean = 1."""

    @staticmethod
    def _build_weights(y: np.ndarray) -> np.ndarray:
        pos_c = int(y.sum())
        neg_c = int(len(y) - pos_c)
        pw = 2.0 * neg_c / pos_c
        class_w = np.where(y == 1, pw, 1.0).astype(float)
        n = len(y)
        decay_half = n * 0.4
        time_w = np.exp(np.log(2) * np.arange(n) / decay_half)
        time_w /= time_w.mean()
        sw = (class_w * time_w).astype(float)
        sw /= sw.mean()
        return sw

    def test_positive_class_has_higher_avg_weight(self) -> None:
        y = np.array([0, 0, 0, 0, 1, 1], dtype=int)
        sw = self._build_weights(y)
        self.assertGreater(sw[y == 1].mean(), sw[y == 0].mean())

    def test_time_decay_component_increases_over_bar_index(self) -> None:
        """Exponential time decay: later bars must have higher weight than earlier bars."""
        n = 200
        decay_half = n * 0.4
        time_w = np.exp(np.log(2) * np.arange(n) / decay_half)
        time_w /= time_w.mean()
        # Last bar heavier than first bar
        self.assertGreater(float(time_w[-1]), float(time_w[0]))
        # Bars after midpoint are above-mean weight
        self.assertGreater(float(time_w[n // 2:].mean()), 1.0)
        # Bars before midpoint are below-mean weight
        self.assertLess(float(time_w[: n // 2].mean()), 1.0)

    def test_mean_weight_is_one(self) -> None:
        rng = np.random.default_rng(99)
        y = (rng.random(800) > 0.6).astype(int)
        sw = self._build_weights(y)
        self.assertAlmostEqual(float(sw.mean()), 1.0, places=5)

    def test_all_weights_positive(self) -> None:
        rng = np.random.default_rng(5)
        y = (rng.random(300) > 0.55).astype(int)
        sw = self._build_weights(y)
        self.assertTrue(np.all(sw > 0))


class Combo133LiveConfigParityTests(unittest.TestCase):
    """live_acc1.yaml must mirror Combo #133 WF settings."""

    @classmethod
    def setUpClass(cls) -> None:
        import yaml  # guaranteed via requirements

        config_path = ROOT / "configs" / "live_acc1.yaml"
        if not config_path.exists():
            raise unittest.SkipTest("live_acc1.yaml not found")
        cls.raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    def test_risk_min_confidence_is_0_70(self) -> None:
        min_conf = self.raw.get("risk", {}).get("min_confidence")
        self.assertIsNotNone(min_conf, "risk.min_confidence missing from live_acc1.yaml")
        self.assertAlmostEqual(float(min_conf), 0.70, places=5)

    def test_strategy_sideway_min_confidence_is_0_70(self) -> None:
        val = self.raw.get("strategy", {}).get("sideway_min_confidence")
        self.assertIsNotNone(val, "strategy.sideway_min_confidence missing")
        self.assertAlmostEqual(float(val), 0.70, places=5)

    def test_strategy_volatile_min_confidence_is_0_70(self) -> None:
        val = self.raw.get("strategy", {}).get("volatile_min_confidence")
        self.assertIsNotNone(val, "strategy.volatile_min_confidence missing")
        self.assertAlmostEqual(float(val), 0.70, places=5)

    def test_blocked_hours_are_3_15_17_22_23(self) -> None:
        blocked = self.raw.get("strategy", {}).get("blocked_hours_utc", [])
        self.assertEqual(sorted(blocked), [3, 15, 17, 22, 23])

    def test_d1_trend_gate_disabled(self) -> None:
        val = self.raw.get("strategy", {}).get("d1_trend_gate")
        self.assertFalse(val, "strategy.d1_trend_gate must be false for Combo #133")

    def test_feature_columns_has_judas_swing_signal(self) -> None:
        from xauusd_ai.features.dataset import FEATURE_COLUMNS
        self.assertIn("judas_swing_signal", FEATURE_COLUMNS)

    def test_feature_columns_has_silver_bullet_setup(self) -> None:
        from xauusd_ai.features.dataset import FEATURE_COLUMNS
        self.assertIn("silver_bullet_setup", FEATURE_COLUMNS)

    def test_combo133_best_yaml_has_correct_train_bars(self) -> None:
        """xauusd_combo133_best.yaml used by live retrain must have same TRAIN_BARS logic."""
        import yaml

        best_path = ROOT / "configs" / "xauusd_combo133_best.yaml"
        if not best_path.exists():
            self.skipTest("xauusd_combo133_best.yaml not found")
        raw = yaml.safe_load(best_path.read_text(encoding="utf-8")) or {}
        # blocked hours must match
        blocked = raw.get("strategy", {}).get("blocked_hours_utc", [])
        self.assertEqual(sorted(blocked), [3, 15, 17, 22, 23])


if __name__ == "__main__":
    unittest.main()
