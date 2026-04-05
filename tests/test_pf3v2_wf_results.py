"""
Tests verifying the stored Walk-Forward search artifacts for V2 PF3 configs.

Sources verified:
  - outputs/wf_pf3v2_acc1_best_feasible.json   (V2 search results)
  - outputs/wf_pf3v2_acc2_best_feasible.json
  - configs/benchmarks/acc1_pf3v2_net815k_dd2891.yaml
  - configs/benchmarks/acc2_pf3v2_net63k_dd1864.yaml
  - configs/benchmarks/wf_expanded_profiles_20260619.json

Covers:
  1. WF most_trades metrics exact (PF, net, DD, trades, WR, balances, folds)
  2. WF most_trades gross profit/loss exact
  3. WF baseline metrics stored correctly per account
  4. total_feasible count stored in JSON
  5. YAML config risk params match search candidate (cross-validate)
  6. YAML config strategy params match search candidate (cross-validate)
  7. wf_expanded_profiles has v2 profiles with correct expected_wf and candidate fields
  8. Requirements met: PF>=3, Net_v2 > Net_old, DD_v2 < DD_old (both accounts)
"""

from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from xauusd_ai.config import load_settings

_REPO = Path(__file__).resolve().parents[1]

_ACC1_FEASIBLE_JSON = _REPO / "outputs" / "wf_pf3v2_acc1_best_feasible.json"
_ACC2_FEASIBLE_JSON = _REPO / "outputs" / "wf_pf3v2_acc2_best_feasible.json"
_PROFILES_JSON = _REPO / "configs" / "benchmarks" / "wf_expanded_profiles_20260619.json"
_ACC1_CFG = _REPO / "configs" / "benchmarks" / "acc1_pf3v2_net815k_dd2891.yaml"
_ACC2_CFG = _REPO / "configs" / "benchmarks" / "acc2_pf3v2_net63k_dd1864.yaml"

# Old benchmark values (used for requirements tests)
_ACC1_OLD_NET = 127_313.86
_ACC1_OLD_DD_PCT = 32.15
_ACC2_OLD_NET = 35_714.67
_ACC2_OLD_DD_PCT = 22.81
_PF_MIN = 3.0


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _approx(a: float, b: float, rel: float = 1e-3) -> bool:
    """True if |a - b| <= rel * max(|a|, |b|, 1)."""
    return abs(a - b) <= rel * max(abs(a), abs(b), 1.0)


# ---------------------------------------------------------------------------
# 1. ACC1 WF most_trades exact metrics
# ---------------------------------------------------------------------------

class TestAcc1WFResultMostTrades(unittest.TestCase):
    """Exact WF walk-forward metrics stored in wf_pf3v2_acc1_best_feasible.json."""

    @classmethod
    def setUpClass(cls):
        cls.d = _load_json(_ACC1_FEASIBLE_JSON)["most_trades"]

    def test_artifact_file_exists(self):
        self.assertTrue(_ACC1_FEASIBLE_JSON.exists(), "ACC1 WF result JSON missing")

    def test_feasible_flag_is_true(self):
        self.assertTrue(self.d["feasible"])

    def test_total_folds(self):
        self.assertEqual(self.d["folds"], 8)

    def test_starting_balance(self):
        self.assertAlmostEqual(self.d["starting_balance"], 200.0, places=2)

    def test_profit_factor_global(self):
        self.assertAlmostEqual(self.d["profit_factor_global"], 3.0442, places=4)

    def test_sum_net_profit(self):
        self.assertAlmostEqual(self.d["sum_net_profit"], 815_620.77, places=2)

    def test_ending_balance(self):
        self.assertAlmostEqual(self.d["ending_balance"], 815_820.77, places=2)

    def test_global_max_drawdown_pct(self):
        self.assertAlmostEqual(self.d["global_max_drawdown_pct"], 28.91, places=2)

    def test_total_trades(self):
        self.assertEqual(self.d["total_trades"], 322)

    def test_win_rate(self):
        self.assertAlmostEqual(self.d["win_rate"], 0.7609, places=4)

    def test_sum_gross_profit(self):
        self.assertAlmostEqual(self.d["sum_gross_profit"], 1_214_604.32, places=2)

    def test_sum_gross_loss(self):
        self.assertAlmostEqual(self.d["sum_gross_loss"], 398_983.56, places=2)

    def test_pf_equals_gross_profit_over_gross_loss(self):
        # profit_factor_global == sum_gross_profit / sum_gross_loss
        expected_pf = self.d["sum_gross_profit"] / self.d["sum_gross_loss"]
        self.assertAlmostEqual(self.d["profit_factor_global"], expected_pf, places=3)

    def test_net_profit_equals_ending_minus_starting(self):
        diff = self.d["ending_balance"] - self.d["starting_balance"]
        self.assertAlmostEqual(self.d["sum_net_profit"], diff, places=2)

    def test_compound_cap_50(self):
        self.assertAlmostEqual(self.d["compound_cap"], 50.0, places=2)

    def test_candidate_threshold(self):
        self.assertAlmostEqual(self.d["threshold"], 0.86, places=4)

    def test_candidate_risk_per_trade(self):
        self.assertAlmostEqual(self.d["risk_per_trade"], 0.07, places=4)

    def test_candidate_take_profit_rr(self):
        self.assertAlmostEqual(self.d["take_profit_rr"], 7.0, places=4)

    def test_candidate_template(self):
        self.assertEqual(self.d["template"], "minimal")

    def test_candidate_silver_bullet_disabled(self):
        self.assertFalse(self.d["silver_bullet_enabled"])

    def test_candidate_adx_gate_enabled(self):
        self.assertTrue(self.d["adx_gate_enabled"])

    def test_candidate_adx_min_trend(self):
        self.assertAlmostEqual(self.d["adx_min_trend"], 12.0, places=2)

    def test_candidate_consecutive_loss_pause_count(self):
        self.assertEqual(self.d["consecutive_loss_pause_count"], 3)

    def test_candidate_consecutive_loss_cooldown_bars(self):
        self.assertEqual(self.d["consecutive_loss_cooldown_bars"], 8)

    def test_candidate_anti_martingale_factor(self):
        self.assertAlmostEqual(self.d["anti_martingale_factor"], 0.6, places=4)

    def test_candidate_max_open_positions(self):
        self.assertEqual(self.d["max_open_positions"], 3)

    def test_candidate_stop_loss_atr_multiple(self):
        self.assertAlmostEqual(self.d["stop_loss_atr_multiple"], 1.2, places=4)

    def test_candidate_daily_loss_limit_pct(self):
        self.assertAlmostEqual(self.d["daily_loss_limit_pct"], 0.03, places=4)

    def test_candidate_partial_tp_disabled(self):
        self.assertFalse(self.d["partial_tp_enabled"])

    def test_candidate_require_trend_alignment_false(self):
        self.assertFalse(self.d["require_trend_alignment"])

    def test_candidate_risk_min_conf(self):
        self.assertAlmostEqual(self.d["risk_min_conf"], 0.85, places=4)

    def test_candidate_sideway_min_conf(self):
        self.assertAlmostEqual(self.d["sideway_min_conf"], 0.8, places=4)

    def test_candidate_volatile_min_conf(self):
        self.assertAlmostEqual(self.d["volatile_min_conf"], 0.75, places=4)

    def test_candidate_sideway_risk_multiplier(self):
        self.assertAlmostEqual(self.d["sideway_risk_multiplier"], 0.2, places=4)

    def test_candidate_strong_volatility_risk_multiplier(self):
        self.assertAlmostEqual(self.d["strong_volatility_risk_multiplier"], 0.8, places=4)

    def test_candidate_spread_cost_rr(self):
        self.assertAlmostEqual(self.d["spread_cost_rr"], 0.01, places=4)

    def test_candidate_slippage_rr(self):
        self.assertAlmostEqual(self.d["slippage_rr"], 0.01, places=4)

    def test_candidate_commission_rr_zero(self):
        self.assertAlmostEqual(self.d["commission_rr"], 0.0, places=4)

    def test_candidate_min_strategy_score(self):
        self.assertAlmostEqual(self.d["min_strategy_score"], 0.15, places=4)


# ---------------------------------------------------------------------------
# 2. ACC2 WF most_trades exact metrics
# ---------------------------------------------------------------------------

class TestAcc2WFResultMostTrades(unittest.TestCase):
    """Exact WF walk-forward metrics stored in wf_pf3v2_acc2_best_feasible.json."""

    @classmethod
    def setUpClass(cls):
        cls.d = _load_json(_ACC2_FEASIBLE_JSON)["most_trades"]

    def test_artifact_file_exists(self):
        self.assertTrue(_ACC2_FEASIBLE_JSON.exists(), "ACC2 WF result JSON missing")

    def test_feasible_flag_is_true(self):
        self.assertTrue(self.d["feasible"])

    def test_total_folds(self):
        self.assertEqual(self.d["folds"], 8)

    def test_starting_balance(self):
        self.assertAlmostEqual(self.d["starting_balance"], 200.0, places=2)

    def test_profit_factor_global(self):
        self.assertAlmostEqual(self.d["profit_factor_global"], 3.9679, places=4)

    def test_sum_net_profit(self):
        self.assertAlmostEqual(self.d["sum_net_profit"], 63_164.18, places=2)

    def test_ending_balance(self):
        self.assertAlmostEqual(self.d["ending_balance"], 63_364.18, places=2)

    def test_global_max_drawdown_pct(self):
        self.assertAlmostEqual(self.d["global_max_drawdown_pct"], 18.64, places=2)

    def test_total_trades(self):
        self.assertEqual(self.d["total_trades"], 182)

    def test_win_rate(self):
        self.assertAlmostEqual(self.d["win_rate"], 0.8187, places=4)

    def test_sum_gross_profit(self):
        self.assertAlmostEqual(self.d["sum_gross_profit"], 84_446.38, places=2)

    def test_sum_gross_loss(self):
        self.assertAlmostEqual(self.d["sum_gross_loss"], 21_282.21, places=2)

    def test_pf_equals_gross_profit_over_gross_loss(self):
        expected_pf = self.d["sum_gross_profit"] / self.d["sum_gross_loss"]
        self.assertAlmostEqual(self.d["profit_factor_global"], expected_pf, places=3)

    def test_net_profit_equals_ending_minus_starting(self):
        diff = self.d["ending_balance"] - self.d["starting_balance"]
        self.assertAlmostEqual(self.d["sum_net_profit"], diff, places=2)

    def test_compound_cap_50(self):
        self.assertAlmostEqual(self.d["compound_cap"], 50.0, places=2)

    def test_candidate_threshold(self):
        self.assertAlmostEqual(self.d["threshold"], 0.8, places=4)

    def test_candidate_risk_per_trade(self):
        self.assertAlmostEqual(self.d["risk_per_trade"], 0.06, places=4)

    def test_candidate_take_profit_rr(self):
        self.assertAlmostEqual(self.d["take_profit_rr"], 10.0, places=4)

    def test_candidate_template(self):
        self.assertEqual(self.d["template"], "focus_hours")

    def test_candidate_silver_bullet_enabled(self):
        self.assertTrue(self.d["silver_bullet_enabled"])

    def test_candidate_adx_gate_disabled(self):
        self.assertFalse(self.d["adx_gate_enabled"])

    def test_candidate_adx_min_trend(self):
        self.assertAlmostEqual(self.d["adx_min_trend"], 22.0, places=2)

    def test_candidate_consecutive_loss_pause_count(self):
        self.assertEqual(self.d["consecutive_loss_pause_count"], 3)

    def test_candidate_consecutive_loss_cooldown_bars(self):
        self.assertEqual(self.d["consecutive_loss_cooldown_bars"], 12)

    def test_candidate_anti_martingale_factor(self):
        self.assertAlmostEqual(self.d["anti_martingale_factor"], 0.7, places=4)

    def test_candidate_max_open_positions(self):
        self.assertEqual(self.d["max_open_positions"], 2)

    def test_candidate_stop_loss_atr_multiple(self):
        self.assertAlmostEqual(self.d["stop_loss_atr_multiple"], 1.2, places=4)

    def test_candidate_daily_loss_limit_pct(self):
        self.assertAlmostEqual(self.d["daily_loss_limit_pct"], 0.015, places=4)

    def test_candidate_partial_tp_disabled(self):
        self.assertFalse(self.d["partial_tp_enabled"])

    def test_candidate_require_trend_alignment_false(self):
        self.assertFalse(self.d["require_trend_alignment"])

    def test_candidate_risk_min_conf(self):
        self.assertAlmostEqual(self.d["risk_min_conf"], 0.75, places=4)

    def test_candidate_sideway_min_conf(self):
        self.assertAlmostEqual(self.d["sideway_min_conf"], 0.85, places=4)

    def test_candidate_volatile_min_conf(self):
        self.assertAlmostEqual(self.d["volatile_min_conf"], 0.75, places=4)

    def test_candidate_sideway_risk_multiplier(self):
        self.assertAlmostEqual(self.d["sideway_risk_multiplier"], 0.2, places=4)

    def test_candidate_strong_volatility_risk_multiplier(self):
        self.assertAlmostEqual(self.d["strong_volatility_risk_multiplier"], 1.0, places=4)

    def test_candidate_spread_cost_rr(self):
        self.assertAlmostEqual(self.d["spread_cost_rr"], 0.01, places=4)

    def test_candidate_slippage_rr(self):
        self.assertAlmostEqual(self.d["slippage_rr"], 0.01, places=4)

    def test_candidate_commission_rr_zero(self):
        self.assertAlmostEqual(self.d["commission_rr"], 0.0, places=4)

    def test_candidate_min_strategy_score(self):
        self.assertAlmostEqual(self.d["min_strategy_score"], 0.15, places=4)


# ---------------------------------------------------------------------------
# 3. Baseline metrics stored in each JSON
# ---------------------------------------------------------------------------

class TestAcc1WFBaseline(unittest.TestCase):
    """ACC1 WF baseline (old config) metrics stored in best_feasible JSON."""

    @classmethod
    def setUpClass(cls):
        cls.b = _load_json(_ACC1_FEASIBLE_JSON)["baseline"]

    def test_baseline_feasible(self):
        self.assertTrue(self.b["feasible"])

    def test_baseline_profit_factor(self):
        self.assertAlmostEqual(self.b["profit_factor_global"], 2.162, places=3)

    def test_baseline_net_profit(self):
        self.assertAlmostEqual(self.b["sum_net_profit"], 210_330.63, places=2)

    def test_baseline_drawdown(self):
        self.assertAlmostEqual(self.b["global_max_drawdown_pct"], 21.84, places=2)

    def test_baseline_total_trades(self):
        self.assertEqual(self.b["total_trades"], 617)

    def test_baseline_win_rate(self):
        self.assertAlmostEqual(self.b["win_rate"], 0.7034, places=4)

    def test_baseline_folds(self):
        self.assertEqual(self.b["folds"], 8)


class TestAcc2WFBaseline(unittest.TestCase):
    """ACC2 WF baseline (old config) metrics stored in best_feasible JSON."""

    @classmethod
    def setUpClass(cls):
        cls.b = _load_json(_ACC2_FEASIBLE_JSON)["baseline"]

    def test_baseline_feasible(self):
        self.assertTrue(self.b["feasible"])

    def test_baseline_profit_factor(self):
        self.assertAlmostEqual(self.b["profit_factor_global"], 1.808, places=3)

    def test_baseline_net_profit(self):
        self.assertAlmostEqual(self.b["sum_net_profit"], 2_423.8, places=2)

    def test_baseline_drawdown(self):
        self.assertAlmostEqual(self.b["global_max_drawdown_pct"], 14.24, places=2)

    def test_baseline_total_trades(self):
        self.assertEqual(self.b["total_trades"], 300)

    def test_baseline_win_rate(self):
        self.assertAlmostEqual(self.b["win_rate"], 0.7, places=4)

    def test_baseline_folds(self):
        self.assertEqual(self.b["folds"], 8)


# ---------------------------------------------------------------------------
# 4. Total feasible count
# ---------------------------------------------------------------------------

class TestWFFeasibleCounts(unittest.TestCase):
    """V2 search returned the documented number of feasible candidates."""

    def test_acc1_total_feasible(self):
        d = _load_json(_ACC1_FEASIBLE_JSON)
        self.assertEqual(d["total_feasible"], 1730)

    def test_acc2_total_feasible(self):
        d = _load_json(_ACC2_FEASIBLE_JSON)
        self.assertEqual(d["total_feasible"], 1649)


# ---------------------------------------------------------------------------
# 5. YAML config risk params match search candidate (cross-validate)
# ---------------------------------------------------------------------------

class TestAcc1ConfigMatchesCandidate(unittest.TestCase):
    """Loaded YAML risk params must exactly match the stored candidate."""

    @classmethod
    def setUpClass(cls):
        cls.s = load_settings(_ACC1_CFG)
        cls.c = _load_json(_PROFILES_JSON)["profiles"]["acc1_pf3v2_net815k_dd2891"]["candidate"]
        cls.r = cls.s.risk
        cls.st = cls.s.strategy

    # --- risk ---
    def test_risk_per_trade(self):
        self.assertAlmostEqual(self.r.risk_per_trade, self.c["risk_per_trade"], places=4)

    def test_compound_cap(self):
        self.assertAlmostEqual(self.r.compound_cap, self.c["compound_cap"], places=4)

    def test_take_profit_rr(self):
        self.assertAlmostEqual(self.r.take_profit_rr, self.c["take_profit_rr"], places=4)

    def test_stop_loss_atr_multiple(self):
        self.assertAlmostEqual(self.r.stop_loss_atr_multiple, self.c["stop_loss_atr_multiple"], places=4)

    def test_max_open_positions(self):
        self.assertEqual(self.r.max_open_positions, self.c["max_open_positions"])

    def test_min_confidence(self):
        self.assertAlmostEqual(self.r.min_confidence, self.c["risk_min_conf"], places=4)

    def test_consecutive_loss_pause_count(self):
        self.assertEqual(self.r.consecutive_loss_pause_count, self.c["consecutive_loss_pause_count"])

    def test_consecutive_loss_cooldown_bars(self):
        self.assertEqual(self.r.consecutive_loss_cooldown_bars, self.c["consecutive_loss_cooldown_bars"])

    def test_anti_martingale_factor(self):
        self.assertAlmostEqual(self.r.anti_martingale_factor, self.c["anti_martingale_factor"], places=4)

    def test_sideway_risk_multiplier(self):
        self.assertAlmostEqual(self.r.sideway_risk_multiplier, self.c["sideway_risk_multiplier"], places=4)

    def test_strong_volatility_risk_multiplier(self):
        self.assertAlmostEqual(self.r.strong_volatility_risk_multiplier, self.c["strong_volatility_risk_multiplier"], places=4)

    def test_partial_tp_enabled(self):
        self.assertEqual(self.r.partial_tp_enabled, self.c["partial_tp_enabled"])

    def test_daily_loss_limit_pct(self):
        self.assertAlmostEqual(self.r.daily_loss_limit_pct, self.c["daily_loss_limit_pct"], places=4)

    def test_spread_cost_rr(self):
        self.assertAlmostEqual(self.r.spread_cost_rr, self.c["spread_cost_rr"], places=4)

    def test_slippage_rr(self):
        self.assertAlmostEqual(self.r.slippage_rr, self.c["slippage_rr"], places=4)

    def test_commission_rr(self):
        self.assertAlmostEqual(self.r.commission_rr, self.c["commission_rr"], places=4)

    # --- strategy ---
    def test_signal_threshold(self):
        self.assertAlmostEqual(self.st.signal_threshold, self.c["threshold"], places=4)

    def test_silver_bullet_enabled(self):
        self.assertEqual(self.st.silver_bullet_enabled, self.c["silver_bullet_enabled"])

    def test_adx_gate_enabled(self):
        self.assertEqual(self.st.adx_gate_enabled, self.c["adx_gate_enabled"])

    def test_adx_min_trend(self):
        self.assertAlmostEqual(self.st.adx_min_trend, self.c["adx_min_trend"], places=2)

    def test_min_strategy_score(self):
        self.assertAlmostEqual(self.st.min_strategy_score, self.c["min_strategy_score"], places=4)

    def test_require_trend_alignment(self):
        self.assertEqual(self.st.require_trend_alignment, self.c["require_trend_alignment"])

    def test_sideway_min_confidence(self):
        self.assertAlmostEqual(self.st.sideway_min_confidence, self.c["sideway_min_conf"], places=4)

    def test_volatile_min_confidence(self):
        self.assertAlmostEqual(self.st.volatile_min_confidence, self.c["volatile_min_conf"], places=4)


class TestAcc2ConfigMatchesCandidate(unittest.TestCase):
    """Loaded YAML risk params must exactly match the stored candidate."""

    @classmethod
    def setUpClass(cls):
        cls.s = load_settings(_ACC2_CFG)
        cls.c = _load_json(_PROFILES_JSON)["profiles"]["acc2_pf3v2_net63k_dd1864"]["candidate"]
        cls.r = cls.s.risk
        cls.st = cls.s.strategy

    # --- risk ---
    def test_risk_per_trade(self):
        self.assertAlmostEqual(self.r.risk_per_trade, self.c["risk_per_trade"], places=4)

    def test_compound_cap(self):
        self.assertAlmostEqual(self.r.compound_cap, self.c["compound_cap"], places=4)

    def test_take_profit_rr(self):
        self.assertAlmostEqual(self.r.take_profit_rr, self.c["take_profit_rr"], places=4)

    def test_stop_loss_atr_multiple(self):
        self.assertAlmostEqual(self.r.stop_loss_atr_multiple, self.c["stop_loss_atr_multiple"], places=4)

    def test_max_open_positions(self):
        self.assertEqual(self.r.max_open_positions, self.c["max_open_positions"])

    def test_min_confidence(self):
        self.assertAlmostEqual(self.r.min_confidence, self.c["risk_min_conf"], places=4)

    def test_consecutive_loss_pause_count(self):
        self.assertEqual(self.r.consecutive_loss_pause_count, self.c["consecutive_loss_pause_count"])

    def test_consecutive_loss_cooldown_bars(self):
        self.assertEqual(self.r.consecutive_loss_cooldown_bars, self.c["consecutive_loss_cooldown_bars"])

    def test_anti_martingale_factor(self):
        self.assertAlmostEqual(self.r.anti_martingale_factor, self.c["anti_martingale_factor"], places=4)

    def test_sideway_risk_multiplier(self):
        self.assertAlmostEqual(self.r.sideway_risk_multiplier, self.c["sideway_risk_multiplier"], places=4)

    def test_strong_volatility_risk_multiplier(self):
        self.assertAlmostEqual(self.r.strong_volatility_risk_multiplier, self.c["strong_volatility_risk_multiplier"], places=4)

    def test_partial_tp_enabled(self):
        self.assertEqual(self.r.partial_tp_enabled, self.c["partial_tp_enabled"])

    def test_daily_loss_limit_pct(self):
        self.assertAlmostEqual(self.r.daily_loss_limit_pct, self.c["daily_loss_limit_pct"], places=4)

    def test_spread_cost_rr(self):
        self.assertAlmostEqual(self.r.spread_cost_rr, self.c["spread_cost_rr"], places=4)

    def test_slippage_rr(self):
        self.assertAlmostEqual(self.r.slippage_rr, self.c["slippage_rr"], places=4)

    def test_commission_rr(self):
        self.assertAlmostEqual(self.r.commission_rr, self.c["commission_rr"], places=4)

    # --- strategy ---
    def test_signal_threshold(self):
        self.assertAlmostEqual(self.st.signal_threshold, self.c["threshold"], places=4)

    def test_silver_bullet_enabled(self):
        self.assertEqual(self.st.silver_bullet_enabled, self.c["silver_bullet_enabled"])

    def test_adx_gate_enabled(self):
        self.assertEqual(self.st.adx_gate_enabled, self.c["adx_gate_enabled"])

    def test_adx_min_trend(self):
        self.assertAlmostEqual(self.st.adx_min_trend, self.c["adx_min_trend"], places=2)

    def test_min_strategy_score(self):
        self.assertAlmostEqual(self.st.min_strategy_score, self.c["min_strategy_score"], places=4)

    def test_require_trend_alignment(self):
        self.assertEqual(self.st.require_trend_alignment, self.c["require_trend_alignment"])

    def test_sideway_min_confidence(self):
        self.assertAlmostEqual(self.st.sideway_min_confidence, self.c["sideway_min_conf"], places=4)

    def test_volatile_min_confidence(self):
        self.assertAlmostEqual(self.st.volatile_min_confidence, self.c["volatile_min_conf"], places=4)


# ---------------------------------------------------------------------------
# 6. wf_expanded_profiles updated with v2 entries
# ---------------------------------------------------------------------------

class TestExpandedProfilesV2Entries(unittest.TestCase):
    """wf_expanded_profiles_20260619.json must have correct v2 profile entries."""

    @classmethod
    def setUpClass(cls):
        cls.profiles = _load_json(_PROFILES_JSON)["profiles"]

    def test_acc1_v2_profile_exists(self):
        self.assertIn("acc1_pf3v2_net815k_dd2891", self.profiles)

    def test_acc2_v2_profile_exists(self):
        self.assertIn("acc2_pf3v2_net63k_dd1864", self.profiles)

    def test_acc1_config_path(self):
        p = self.profiles["acc1_pf3v2_net815k_dd2891"]
        self.assertEqual(p["config_path"], "configs/benchmarks/acc1_pf3v2_net815k_dd2891.yaml")

    def test_acc2_config_path(self):
        p = self.profiles["acc2_pf3v2_net63k_dd1864"]
        self.assertEqual(p["config_path"], "configs/benchmarks/acc2_pf3v2_net63k_dd1864.yaml")

    def test_acc1_expected_wf_net(self):
        wf = self.profiles["acc1_pf3v2_net815k_dd2891"]["expected_wf"]
        self.assertAlmostEqual(wf["sum_net_profit"], 815_620.77, places=2)

    def test_acc1_expected_wf_dd(self):
        wf = self.profiles["acc1_pf3v2_net815k_dd2891"]["expected_wf"]
        self.assertAlmostEqual(wf["global_max_drawdown_pct"], 28.91, places=2)

    def test_acc1_expected_wf_pf(self):
        wf = self.profiles["acc1_pf3v2_net815k_dd2891"]["expected_wf"]
        self.assertAlmostEqual(wf["profit_factor_global"], 3.0442, places=4)

    def test_acc1_expected_wf_trades(self):
        wf = self.profiles["acc1_pf3v2_net815k_dd2891"]["expected_wf"]
        self.assertEqual(wf["total_trades"], 322)

    def test_acc1_expected_wf_win_rate(self):
        wf = self.profiles["acc1_pf3v2_net815k_dd2891"]["expected_wf"]
        self.assertAlmostEqual(wf["win_rate"], 0.7609, places=4)

    def test_acc1_expected_wf_ending_balance(self):
        wf = self.profiles["acc1_pf3v2_net815k_dd2891"]["expected_wf"]
        self.assertAlmostEqual(wf["ending_balance"], 815_820.77, places=2)

    def test_acc2_expected_wf_net(self):
        wf = self.profiles["acc2_pf3v2_net63k_dd1864"]["expected_wf"]
        self.assertAlmostEqual(wf["sum_net_profit"], 63_164.18, places=2)

    def test_acc2_expected_wf_dd(self):
        wf = self.profiles["acc2_pf3v2_net63k_dd1864"]["expected_wf"]
        self.assertAlmostEqual(wf["global_max_drawdown_pct"], 18.64, places=2)

    def test_acc2_expected_wf_pf(self):
        wf = self.profiles["acc2_pf3v2_net63k_dd1864"]["expected_wf"]
        self.assertAlmostEqual(wf["profit_factor_global"], 3.9679, places=4)

    def test_acc2_expected_wf_trades(self):
        wf = self.profiles["acc2_pf3v2_net63k_dd1864"]["expected_wf"]
        self.assertEqual(wf["total_trades"], 182)

    def test_acc2_expected_wf_win_rate(self):
        wf = self.profiles["acc2_pf3v2_net63k_dd1864"]["expected_wf"]
        self.assertAlmostEqual(wf["win_rate"], 0.8187, places=4)

    def test_acc2_expected_wf_ending_balance(self):
        wf = self.profiles["acc2_pf3v2_net63k_dd1864"]["expected_wf"]
        self.assertAlmostEqual(wf["ending_balance"], 63_364.18, places=2)

    def test_acc1_feasible_total_stored_in_profile(self):
        p = self.profiles["acc1_pf3v2_net815k_dd2891"]
        self.assertEqual(p["search_feasible_total"], 1730)

    def test_acc2_feasible_total_stored_in_profile(self):
        p = self.profiles["acc2_pf3v2_net63k_dd1864"]
        self.assertEqual(p["search_feasible_total"], 1649)

    def test_acc1_model_path_uses_expand_model(self):
        p = self.profiles["acc1_pf3v2_net815k_dd2891"]
        self.assertIn("acc1_expand", p["model_path"])

    def test_acc2_model_path_uses_expand_model(self):
        p = self.profiles["acc2_pf3v2_net63k_dd1864"]
        self.assertIn("acc2_expand", p["model_path"])

    def test_profiles_updated_at_utc(self):
        meta = _load_json(_PROFILES_JSON)
        self.assertEqual(meta["updated_at_utc"], "2026-07-09T00:00:00Z")

    def test_total_profiles_count(self):
        # 2 expand + 2 pf3v1 + 2 pf3v2 = 6
        self.assertEqual(len(self.profiles), 6)


# ---------------------------------------------------------------------------
# 7. Requirements met: PF>=3, Net_v2 > Net_old, DD_v2 < DD_old (both accs)
# ---------------------------------------------------------------------------

class TestV2RequirementsMet(unittest.TestCase):
    """Verify all three improvement requirements hold simultaneously for both accounts."""

    @classmethod
    def setUpClass(cls):
        cls.acc1 = _load_json(_ACC1_FEASIBLE_JSON)["most_trades"]
        cls.acc2 = _load_json(_ACC2_FEASIBLE_JSON)["most_trades"]

    # ACC1
    def test_acc1_pf_gte_3(self):
        self.assertGreaterEqual(self.acc1["profit_factor_global"], _PF_MIN,
                                f"ACC1 PF={self.acc1['profit_factor_global']:.4f} < {_PF_MIN}")

    def test_acc1_net_exceeds_old_benchmark(self):
        self.assertGreater(self.acc1["sum_net_profit"], _ACC1_OLD_NET,
                           f"ACC1 net={self.acc1['sum_net_profit']:.2f} not > old={_ACC1_OLD_NET}")

    def test_acc1_dd_below_old_benchmark(self):
        self.assertLess(self.acc1["global_max_drawdown_pct"], _ACC1_OLD_DD_PCT,
                        f"ACC1 DD={self.acc1['global_max_drawdown_pct']:.2f}% not < old={_ACC1_OLD_DD_PCT}%")

    def test_acc1_all_three_requirements_simultaneously(self):
        pf_ok = self.acc1["profit_factor_global"] >= _PF_MIN
        net_ok = self.acc1["sum_net_profit"] > _ACC1_OLD_NET
        dd_ok = self.acc1["global_max_drawdown_pct"] < _ACC1_OLD_DD_PCT
        self.assertTrue(pf_ok and net_ok and dd_ok,
                        f"ACC1 requirements not all met: PF={pf_ok}, NET={net_ok}, DD={dd_ok}")

    # ACC2
    def test_acc2_pf_gte_3(self):
        self.assertGreaterEqual(self.acc2["profit_factor_global"], _PF_MIN,
                                f"ACC2 PF={self.acc2['profit_factor_global']:.4f} < {_PF_MIN}")

    def test_acc2_net_exceeds_old_benchmark(self):
        self.assertGreater(self.acc2["sum_net_profit"], _ACC2_OLD_NET,
                           f"ACC2 net={self.acc2['sum_net_profit']:.2f} not > old={_ACC2_OLD_NET}")

    def test_acc2_dd_below_old_benchmark(self):
        self.assertLess(self.acc2["global_max_drawdown_pct"], _ACC2_OLD_DD_PCT,
                        f"ACC2 DD={self.acc2['global_max_drawdown_pct']:.2f}% not < old={_ACC2_OLD_DD_PCT}%")

    def test_acc2_all_three_requirements_simultaneously(self):
        pf_ok = self.acc2["profit_factor_global"] >= _PF_MIN
        net_ok = self.acc2["sum_net_profit"] > _ACC2_OLD_NET
        dd_ok = self.acc2["global_max_drawdown_pct"] < _ACC2_OLD_DD_PCT
        self.assertTrue(pf_ok and net_ok and dd_ok,
                        f"ACC2 requirements not all met: PF={pf_ok}, NET={net_ok}, DD={dd_ok}")

    # Cross-account: ACC1 net much larger (compound advantage vs tight focus_hours)
    def test_acc1_net_larger_than_acc2(self):
        self.assertGreater(self.acc1["sum_net_profit"], self.acc2["sum_net_profit"])

    # ACC2 has higher PF (cleaner signals but fewer trades)
    def test_acc2_pf_higher_than_acc1(self):
        self.assertGreater(self.acc2["profit_factor_global"], self.acc1["profit_factor_global"])

    # ACC2 has lower DD (less aggressive risk)
    def test_acc2_dd_lower_than_acc1(self):
        self.assertLess(self.acc2["global_max_drawdown_pct"], self.acc1["global_max_drawdown_pct"])

    # ACC1 has more trades (minimal template = broader hours)
    def test_acc1_more_trades_than_acc2(self):
        self.assertGreater(self.acc1["total_trades"], self.acc2["total_trades"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
