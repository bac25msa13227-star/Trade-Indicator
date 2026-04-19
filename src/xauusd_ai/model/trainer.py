from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from xauusd_ai.config import Settings
from xauusd_ai.features.dataset import FEATURE_COLUMNS

try:
    from xauusd_ai.infra.mlflow_client import MLflowTracker
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


class ModelTrainer:
    def __init__(self, settings: Settings, mlflow_tracker=None) -> None:
        self.settings = settings
        self._mlflow: MLflowTracker | None = mlflow_tracker  # type: ignore[name-defined]
        self._use_ensemble = bool(getattr(settings.training, 'use_ensemble', False))
        self._feat_sel_drop = int(getattr(settings.training, 'feature_selection_drop_pct', 0))
        self.model = self._build_model()
        self._feature_mask = None
        self._calibrator = None
        self.scaler = StandardScaler()
        self.decision_threshold = settings.strategy.signal_threshold
        self.feature_columns: list[str] = list(FEATURE_COLUMNS)  # updated by load_artifacts for backward compat

    def _build_model(self):
        """Build model: WF-identical ensemble or single HGB."""
        if self._use_ensemble:
            _hgb = HistGradientBoostingClassifier(
                max_iter=1000, learning_rate=0.01, max_depth=7,
                min_samples_leaf=20, l2_regularization=1.0,
                max_bins=128, class_weight=None,
                early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=40, random_state=42,
            )
            _rf = RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_leaf=15,
                max_features="sqrt", class_weight="balanced",
                n_jobs=-1, random_state=42,
            )
            _et = ExtraTreesClassifier(
                n_estimators=200, max_depth=14, min_samples_leaf=10,
                max_features="sqrt", class_weight="balanced",
                n_jobs=-1, random_state=42,
            )
            return VotingClassifier(
                estimators=[("hgb", _hgb), ("rf", _rf), ("et", _et)],
                voting="soft",
                weights=[3, 2, 1],
            )
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=5,
            min_samples_leaf=20, l2_regularization=1.0,
            max_bins=63, class_weight=None,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=42,
        )

    def _aligned_feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Align any dataframe/row to model feature schema.

        Live runtime may load a model whose ``feature_columns`` are larger than
        current dataset features (e.g. migrated artifacts). For inference we
        keep bot running by creating missing columns filled with 0.0, then
        reordering columns exactly as model expects.
        """
        missing = [col for col in self.feature_columns if col not in frame.columns]
        if missing:
            logging.getLogger(__name__).warning(
                "ModelTrainer: missing %d feature columns at inference; fill 0.0: %s",
                len(missing),
                missing,
            )
        aligned = frame.copy()
        for col in missing:
            aligned[col] = 0.0
        return aligned[self.feature_columns]

    def train(self, dataset: pd.DataFrame, save_artifacts: bool = True) -> dict[str, float]:
        train_df = dataset[dataset["split"] == "train"]
        test_df = dataset[dataset["split"] == "test"]

        threshold = self.settings.strategy.signal_threshold
        if save_artifacts and self.settings.training.optimize_threshold and len(train_df) > 50:
            threshold = self._optimize_threshold(train_df)
        self.decision_threshold = threshold

        x_train = self.scaler.fit_transform(train_df[self.feature_columns])
        y_train = train_df["target"]
        x_test = self.scaler.transform(test_df[self.feature_columns])
        y_test = test_df["target"]

        # Dynamic sample weights: 2× positive boost + time-decay
        pos_count = int(y_train.sum())
        neg_count = int(len(y_train) - pos_count)
        if pos_count > 10 and neg_count > 10:
            pos_w = 2.0 * neg_count / pos_count
            class_w = np.where(y_train.values == 1, pos_w, 1.0).astype(float)
            # Time-decay: recent bars weighted higher (half-life at 40%)
            _n = len(y_train)
            _decay_half = _n * 0.4
            time_w = np.exp(np.log(2) * np.arange(_n) / _decay_half)
            time_w /= time_w.mean()
            sw = (class_w * time_w).astype(float)
            sw /= sw.mean()
        else:
            sw = None

        # Feature selection: RF-based drop bottom N% (matches WF pipeline)
        if self._feat_sel_drop > 0:
            _scout = RandomForestClassifier(
                n_estimators=80, max_depth=8, min_samples_leaf=20,
                class_weight="balanced", n_jobs=-1, random_state=42,
            )
            _scout.fit(x_train, y_train, sample_weight=sw)
            _imp = _scout.feature_importances_
            _imp_thr = np.percentile(_imp, self._feat_sel_drop)
            self._feature_mask = _imp >= _imp_thr
            if self._feature_mask.sum() < 10:
                self._feature_mask = None
        else:
            self._feature_mask = None
        x_train_sel = self._apply_feature_mask(x_train)
        x_test_sel = self._apply_feature_mask(x_test)

        self.model = self._build_model()  # fresh model each train
        self.model.fit(x_train_sel, y_train, sample_weight=sw)

        # Probability calibration via isotonic regression on validation holdout
        _val_size = max(int(len(x_train_sel) * 0.15), 50)
        if _val_size < len(x_train_sel):
            _cal_x = x_train_sel[-_val_size:]
            _cal_y = y_train.values[-_val_size:]
            try:
                self._calibrator = CalibratedClassifierCV(
                    self.model, method="isotonic", cv="prefit",
                )
                self._calibrator.fit(_cal_x, _cal_y)
                probabilities = self._calibrator.predict_proba(x_test_sel)[:, 1]
            except Exception:
                self._calibrator = None
                probabilities = self.model.predict_proba(x_test_sel)[:, 1]
        else:
            self._calibrator = None
            probabilities = self.model.predict_proba(x_test_sel)[:, 1]

        predictions = (probabilities >= self.decision_threshold).astype(int)

        metrics = {
            "train_rows": float(len(train_df)),
            "test_rows": float(len(test_df)),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test": float(y_test.mean()),
            "selected_threshold": float(self.decision_threshold),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() > 1 else 0.5,
        }
        self._last_metrics = metrics
        if save_artifacts:
            self._last_roc_auc = metrics.get("roc_auc", 0.0)
            self._save_artifacts()
            self._mlflow_log_training(metrics)
        return metrics

    def _optimize_threshold(self, train_df: pd.DataFrame) -> float:
        n = len(train_df)
        if n < 100:
            return self.settings.strategy.signal_threshold

        candidates = np.arange(
            self.settings.training.threshold_min,
            self.settings.training.threshold_max + self.settings.training.threshold_step,
            self.settings.training.threshold_step,
        )
        prec_floor = self.settings.training.min_precision_floor

        # Use last 30K rows for speed — still captures recent market regime
        n_max = min(n, 30000)
        sub_df_all = train_df.iloc[-n_max:]
        n2 = len(sub_df_all)

        # 1-fold: train on first 70%, validate on last 30%
        split_idx = int(n2 * 0.70)
        sub_df = sub_df_all.iloc[:split_idx]
        val_df = sub_df_all.iloc[split_idx:]
        if val_df.empty or sub_df.empty:
            return self.settings.strategy.signal_threshold

        local_scaler = StandardScaler()
        local_model = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.1, max_depth=4,
            min_samples_leaf=30, class_weight=None,
            early_stopping=False, random_state=42,
        )
        x_sub = local_scaler.fit_transform(sub_df[self.feature_columns])
        y_sub = sub_df["target"]
        x_val = local_scaler.transform(val_df[self.feature_columns])
        y_val = val_df["target"]
        _pos_c = int(y_sub.sum())
        _neg_c = int(len(y_sub) - _pos_c)
        if _pos_c > 5 and _neg_c > 5:
            _pw = 2.0 * _neg_c / _pos_c
            _sw = np.where(y_sub.values == 1, _pw, 1.0).astype(float)
            _sw /= _sw.mean()
        else:
            _sw = None
        local_model.fit(x_sub, y_sub, sample_weight=_sw)
        probabilities = local_model.predict_proba(x_val)[:, 1]

        best_thr = float(self.settings.training.threshold_min)
        best_score = -float("inf")
        safe_thr = float(self.settings.training.threshold_max)
        safe_prec = -1.0
        base_friction = (
            float(getattr(self.settings.risk, "spread_cost_rr", 0.10))
            + float(getattr(self.settings.risk, "slippage_rr", 0.05))
            + float(getattr(self.settings.risk, "commission_rr", 0.02))
        )
        for candidate in candidates:
            preds = (probabilities >= candidate).astype(int)
            trade_count = int(preds.sum())
            if trade_count < 8:
                continue
            precision = precision_score(y_val, preds, zero_division=0)
            recall = recall_score(y_val, preds, zero_division=0)
            if recall < 0.05:
                continue
            if precision > safe_prec:
                safe_prec = precision
                safe_thr = float(candidate)
            selected = val_df.loc[preds == 1]
            if selected.empty:
                continue
            session_mult = pd.to_numeric(selected.get("session_spread_mult", 1.0), errors="coerce").fillna(1.0)
            realized_rr = pd.to_numeric(selected.get("realized_rr", 0.0), errors="coerce").fillna(0.0)
            net_rr = realized_rr - (base_friction + (session_mult - 1.0) * float(getattr(self.settings.risk, "spread_cost_rr", 0.10)))
            gross_profit = float(net_rr[net_rr > 0].sum())
            gross_loss = float(-net_rr[net_rr < 0].sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (2.0 if gross_profit > 0 else 0.0)
            avg_net_rr = float(net_rr.mean())
            positive_rr_rate = float((net_rr > 0).mean())
            trade_density = trade_count / max(len(val_df), 1)
            if precision < prec_floor and (avg_net_rr <= 0 or profit_factor < 1.05):
                continue
            score = (
                avg_net_rr * np.sqrt(trade_count)
                + max(profit_factor - 1.0, -1.0) * 0.35
                + precision * 0.20
                + positive_rr_rate * 0.15
                - max(0.0, 0.015 - trade_density) * 4.0
            )
            if score > best_score:
                best_score = score
                best_thr = float(candidate)
        if best_score == -float("inf"):
            best_thr = safe_thr

        return best_thr

    def _mlflow_log_training(self, metrics: dict) -> None:
        """Log training params + metrics to MLflow if tracker is available."""
        if self._mlflow is None:
            return
        import datetime as _dt
        _run_name = f"train_{_dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        try:
            with self._mlflow.start_run(run_name=_run_name, tags={"source": "live_runner"}):
                params: dict = {
                    "signal_threshold": self.decision_threshold,
                    "feature_count": len(self.feature_columns),
                    "use_ensemble": self._use_ensemble,
                    "feature_selection_drop_pct": self._feat_sel_drop,
                }
                if hasattr(self.model, "max_iter"):
                    params["max_iter"] = self.model.max_iter
                    params["learning_rate"] = self.model.learning_rate
                    params["max_depth"] = self.model.max_depth
                self._mlflow.log_params(params)
                self._mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
                # Upload artifacts to MinIO via MLflow
                model_path = Path(self.settings.app.model_path)
                scaler_path = Path(self.settings.app.scaler_path)
                for artifact in [model_path, scaler_path]:
                    if artifact.exists():
                        self._mlflow.log_artifact(artifact, artifact_path="artifacts")
                cal_path = model_path.with_suffix(".cal.pkl")
                if cal_path.exists():
                    self._mlflow.log_artifact(cal_path, artifact_path="artifacts")
        except Exception as exc:  # noqa: BLE001
            import logging
            logging.getLogger(__name__).warning("MLflow log_training failed: %s", exc)

    @staticmethod
    def _atomic_write_pickle(obj: object, dest: Path) -> None:
        """Write pickle to a temp file in the same dir, then replace dest."""
        import tempfile, os
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(obj, fh)
            # os.replace atomically overwrites dest even if held open for reading
            os.replace(tmp, str(dest))
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _save_artifacts(self) -> None:
        model_path = Path(self.settings.app.model_path)
        scaler_path = Path(self.settings.app.scaler_path)
        meta_path = Path(self.settings.app.model_meta_path)
        self._atomic_write_pickle(self.model, model_path)
        self._atomic_write_pickle(self.scaler, scaler_path)
        # Save calibrator alongside model
        _cal_path = model_path.with_suffix(".cal.pkl")
        if hasattr(self, "_calibrator") and self._calibrator is not None:
            self._atomic_write_pickle(self._calibrator, _cal_path)
        elif _cal_path.exists():
            _cal_path.unlink(missing_ok=True)
        meta_dict: dict = {
            "decision_threshold": self.decision_threshold,
            "feature_columns": self.feature_columns,
        }
        if self._feature_mask is not None:
            meta_dict["feature_mask"] = self._feature_mask.tolist()
        if hasattr(self, "_last_roc_auc"):
            meta_dict["roc_auc"] = round(float(self._last_roc_auc), 6)
        if hasattr(self, "_last_metrics") and self._last_metrics:
            for k in ("precision", "recall", "f1", "accuracy",
                      "train_rows", "test_rows",
                      "positive_rate_train", "positive_rate_test"):
                if k in self._last_metrics:
                    meta_dict[k] = self._last_metrics[k]
            # roc_auc fallback: if _last_roc_auc wasn't set (save_artifacts=False path)
            if "roc_auc" not in meta_dict and "roc_auc" in self._last_metrics:
                meta_dict["roc_auc"] = round(float(self._last_metrics["roc_auc"]), 6)
        meta_path.write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")

    def load_artifacts(self) -> bool:
        model_path = Path(self.settings.app.model_path)
        scaler_path = Path(self.settings.app.scaler_path)
        meta_path = Path(self.settings.app.model_meta_path)
        if not model_path.exists():
            return False
        try:
            with open(str(model_path), "rb") as file_handle:
                self.model = pickle.load(file_handle)
        except (AttributeError, ModuleNotFoundError, ImportError, Exception) as _pkl_err:
            logging.getLogger(__name__).warning(
                "load_artifacts: failed to unpickle model (%s). "
                "Likely sklearn version mismatch — will retrain.",
                _pkl_err,
            )
            return False
        # Load scaler — failure here is non-fatal: tree-based models don't require scaling.
        # An empty/corrupt scaler file (e.g. 0-byte placeholder from git) will not trigger retrain.
        if scaler_path.exists():
            try:
                with open(str(scaler_path), "rb") as file_handle:
                    _loaded_scaler = pickle.load(file_handle)
                # Verify the scaler was actually fitted (has mean_ attribute)
                if hasattr(_loaded_scaler, "mean_") or hasattr(_loaded_scaler, "func"):
                    self.scaler = _loaded_scaler
                else:
                    raise ValueError("scaler not fitted")
            except Exception as _scaler_err:
                from sklearn.preprocessing import FunctionTransformer
                self.scaler = FunctionTransformer()  # identity — tree models don't need scaling
                logging.getLogger(__name__).warning(
                    "load_artifacts: scaler invalid/empty (%s). "
                    "Using identity transform (OK for tree-based models).",
                    _scaler_err,
                )
        # Load calibrator if available
        _cal_path = model_path.with_suffix(".cal.pkl")
        if _cal_path.exists():
            try:
                with open(str(_cal_path), "rb") as fh:
                    self._calibrator = pickle.load(fh)
            except Exception:
                self._calibrator = None
        else:
            self._calibrator = None
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            self.decision_threshold = float(metadata.get("decision_threshold", self.settings.strategy.signal_threshold))
            if "feature_mask" in metadata:
                self._feature_mask = np.array(metadata["feature_mask"], dtype=bool)
            else:
                self._feature_mask = None
            if "feature_columns" in metadata:
                self.feature_columns = list(metadata["feature_columns"])
            elif hasattr(self.scaler, "n_features_in_") and self.scaler.n_features_in_ != len(self.feature_columns):
                # Old model trained with fewer features — slice FEATURE_COLUMNS to match scaler
                self.feature_columns = list(FEATURE_COLUMNS[:self.scaler.n_features_in_])
        return True

    def _apply_feature_mask(self, x: np.ndarray) -> np.ndarray:
        if self._feature_mask is not None and x.shape[1] == len(self._feature_mask):
            return x[:, self._feature_mask]
        return x

    def predict_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        frame = dataset.copy()
        feat_frame = self._aligned_feature_frame(frame)
        x_scaled = self.scaler.transform(feat_frame)
        x_sel = self._apply_feature_mask(x_scaled)
        if hasattr(self, "_calibrator") and self._calibrator is not None:
            probabilities = self._calibrator.predict_proba(x_sel)[:, 1]
        else:
            probabilities = self.model.predict_proba(x_sel)[:, 1]
        frame["probability"] = probabilities
        frame["prediction"] = (probabilities >= self.decision_threshold).astype(int)
        return frame

    def score_live_row(self, live_frame: pd.DataFrame) -> dict[str, float]:
        latest = self._aligned_feature_frame(live_frame.iloc[[-1]])
        x_scaled = self.scaler.transform(latest)
        x_sel = self._apply_feature_mask(x_scaled)
        if hasattr(self, "_calibrator") and self._calibrator is not None:
            probability = float(self._calibrator.predict_proba(x_sel)[:, 1][0])
        else:
            probability = float(self.model.predict_proba(x_sel)[:, 1][0])
        prediction = int(probability >= self.decision_threshold)
        margin = probability - self.decision_threshold
        return {"probability": probability, "prediction": prediction, "margin": margin}

    def train_with_loss_weights(
        self,
        dataset: pd.DataFrame,
        loss_patterns: list[dict],
        weight_factor: float = 2.5,
        save_artifacts: bool = True,
    ) -> dict[str, float]:
        """
        Retrain với sample_weight tăng cho các hàng tương tự pattern lệnh thua.
        Loss patterns: list[dict] với các key từ FEATURE_COLUMNS.
        weight_factor: mức độ upweight (2.5 = các pattern thua nặng gấp 2.5×).

        Cơ chế: tìm các hàng trong training set có volatility_regime + rsi_bucket + side_bias
        giống pattern thua → upweight → model học cẩn thận hơn ở hoàn cảnh đó.
        """
        train_df = dataset[dataset["split"] == "train"].copy()
        test_df = dataset[dataset["split"] == "test"]

        if train_df.empty or not loss_patterns:
            return self.train(dataset)

        # --- Xây dựng sample_weight ---
        weights = np.ones(len(train_df), dtype=float)

        for pattern in loss_patterns:
            regime = pattern.get("volatility_regime", -1)
            rsi_p = pattern.get("rsi", 50.0)
            score_p = pattern.get("strategy_score", 0.0)

            # Tìm hàng có cùng regime VÀ RSI trong vùng ±10 VÀ strategy_score cùng dấu
            regime_match = (train_df["volatility_regime"] == regime).values
            rsi_match = (train_df["rsi"].between(rsi_p - 10, rsi_p + 10)).values
            score_sign_match = (np.sign(train_df["strategy_score"].values) == np.sign(score_p))

            similar_mask = regime_match & rsi_match & score_sign_match
            weights[similar_mask] *= weight_factor

        # Normalize để tổng weight không thay đổi tỷ lệ
        weights = weights / weights.mean()

        threshold = self.settings.strategy.signal_threshold
        if self.settings.training.optimize_threshold and len(train_df) > 50:
            threshold = self._optimize_threshold(train_df)
        self.decision_threshold = threshold

        x_train = self.scaler.fit_transform(train_df[self.feature_columns])
        y_train = train_df["target"]
        x_test = self.scaler.transform(test_df[self.feature_columns])
        y_test = test_df["target"]

        # Feature selection: RF-based drop bottom N% (matches WF pipeline)
        if self._feat_sel_drop > 0:
            _scout = RandomForestClassifier(
                n_estimators=80, max_depth=8, min_samples_leaf=20,
                class_weight="balanced", n_jobs=-1, random_state=42,
            )
            _scout.fit(x_train, y_train, sample_weight=weights)
            _imp = _scout.feature_importances_
            _imp_thr = np.percentile(_imp, self._feat_sel_drop)
            self._feature_mask = _imp >= _imp_thr
            if self._feature_mask.sum() < 10:
                self._feature_mask = None
        else:
            self._feature_mask = None
        x_train_sel = self._apply_feature_mask(x_train)
        x_test_sel = self._apply_feature_mask(x_test)

        # Fit with sample_weight
        self.model = self._build_model()
        self.model.fit(x_train_sel, y_train, sample_weight=weights)

        probabilities = self.model.predict_proba(x_test_sel)[:, 1]
        predictions = (probabilities >= self.decision_threshold).astype(int)

        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        metrics = {
            "train_rows": float(len(train_df)),
            "test_rows": float(len(test_df)),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test": float(y_test.mean()),
            "selected_threshold": float(self.decision_threshold),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() > 1 else 0.5,
            "loss_weighted": True,
            "loss_patterns_count": len(loss_patterns),
            "upweighted_samples": int(weights[weights > 1.0].sum()),
        }
        # Always update in-memory state so _save_artifacts() has fresh metrics
        self._last_metrics = metrics
        self._last_roc_auc = metrics.get("roc_auc", 0.0)
        if save_artifacts:
            self._save_artifacts()
        return metrics
