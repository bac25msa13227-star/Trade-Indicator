from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from xauusd_ai.config import Settings
from xauusd_ai.features.dataset import FEATURE_COLUMNS


class ModelTrainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = LogisticRegression(max_iter=2000, class_weight="balanced")
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
        local_model = LogisticRegression(max_iter=2000, class_weight="balanced")
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
        best_score = (-1.0, -1.0)
        for candidate in candidates:
            predictions = (probabilities >= candidate).astype(int)
            precision = precision_score(y_validation, predictions, zero_division=0)
            recall = recall_score(y_validation, predictions, zero_division=0)
            if precision < self.settings.training.min_precision_floor:
                continue
            score = (recall, precision)
            if score > best_score:
                best_score = score
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
