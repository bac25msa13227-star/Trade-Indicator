from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from xauusd_ai.config import Settings
from xauusd_ai.features.dataset import FEATURE_COLUMNS


class ModelTrainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            early_stopping=False,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.decision_threshold = settings.strategy.signal_threshold

    def train(self, dataset: pd.DataFrame) -> dict[str, float]:
        train_df = dataset[dataset["split"] == "train"]
        test_df = dataset[dataset["split"] == "test"]

        threshold = self.settings.strategy.signal_threshold
        if self.settings.training.optimize_threshold and len(train_df) > 50:
            threshold = self._optimize_threshold(train_df)
        self.decision_threshold = threshold

        x_train = self.scaler.fit_transform(train_df[FEATURE_COLUMNS])
        y_train = train_df["target"]
        x_test = self.scaler.transform(test_df[FEATURE_COLUMNS])
        y_test = test_df["target"]

        self.model.fit(x_train, y_train)
        probabilities = self.model.predict_proba(x_test)[:, 1]
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
        self._save_artifacts()
        return metrics

    def _optimize_threshold(self, train_df: pd.DataFrame) -> float:
        split_index = int(len(train_df) * (1 - self.settings.training.validation_split))
        subtrain_df = train_df.iloc[:split_index]
        validation_df = train_df.iloc[split_index:]
        if validation_df.empty or subtrain_df.empty:
            return self.settings.strategy.signal_threshold

        local_scaler = StandardScaler()
        local_model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_depth=5,
            min_samples_leaf=20, class_weight="balanced",
            early_stopping=False, random_state=42,
        )
        x_subtrain = local_scaler.fit_transform(subtrain_df[FEATURE_COLUMNS])
        y_subtrain = subtrain_df["target"]
        x_validation = local_scaler.transform(validation_df[FEATURE_COLUMNS])
        y_validation = validation_df["target"]
        local_model.fit(x_subtrain, y_subtrain)
        probabilities = local_model.predict_proba(x_validation)[:, 1]

        candidates = np.arange(
            self.settings.training.threshold_min,
            self.settings.training.threshold_max + self.settings.training.threshold_step,
            self.settings.training.threshold_step,
        )
        best_threshold = self.settings.strategy.signal_threshold
        best_f1 = 0.0
        for candidate in candidates:
            predictions = (probabilities >= candidate).astype(int)
            precision = precision_score(y_validation, predictions, zero_division=0)
            if precision < self.settings.training.min_precision_floor:
                continue
            f1_val = f1_score(y_validation, predictions, zero_division=0)
            if f1_val > best_f1:
                best_f1 = f1_val
                best_threshold = float(candidate)

        return best_threshold

    def _save_artifacts(self) -> None:
        model_path = Path(self.settings.app.model_path)
        scaler_path = Path(self.settings.app.scaler_path)
        meta_path = Path(self.settings.app.model_meta_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with model_path.open("wb") as file_handle:
            pickle.dump(self.model, file_handle)
        with scaler_path.open("wb") as file_handle:
            pickle.dump(self.scaler, file_handle)
        meta_path.write_text(json.dumps({"decision_threshold": self.decision_threshold}, indent=2), encoding="utf-8")

    def load_artifacts(self) -> bool:
        model_path = Path(self.settings.app.model_path)
        scaler_path = Path(self.settings.app.scaler_path)
        meta_path = Path(self.settings.app.model_meta_path)
        if not model_path.exists() or not scaler_path.exists():
            return False
        with model_path.open("rb") as file_handle:
            self.model = pickle.load(file_handle)
        with scaler_path.open("rb") as file_handle:
            self.scaler = pickle.load(file_handle)
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            self.decision_threshold = float(metadata.get("decision_threshold", self.settings.strategy.signal_threshold))
        return True

    def predict_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        frame = dataset.copy()
        probabilities = self.model.predict_proba(self.scaler.transform(frame[FEATURE_COLUMNS]))[:, 1]
        frame["probability"] = probabilities
        frame["prediction"] = (probabilities >= self.decision_threshold).astype(int)
        return frame

    def score_live_row(self, live_frame: pd.DataFrame) -> dict[str, float]:
        latest = live_frame.iloc[[-1]][FEATURE_COLUMNS]
        probability = float(self.model.predict_proba(self.scaler.transform(latest))[:, 1][0])
        prediction = int(probability >= self.decision_threshold)
        return {"probability": probability, "prediction": prediction}

    def train_with_loss_weights(
        self,
        dataset: pd.DataFrame,
        loss_patterns: list[dict],
        weight_factor: float = 2.5,
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

        x_train = self.scaler.fit_transform(train_df[FEATURE_COLUMNS])
        y_train = train_df["target"]
        x_test = self.scaler.transform(test_df[FEATURE_COLUMNS])
        y_test = test_df["target"]

        # Fit với sample_weight
        self.model.fit(x_train, y_train, sample_weight=weights)

        probabilities = self.model.predict_proba(x_test)[:, 1]
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
        self._save_artifacts()
        return metrics
