from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_DIM = 2048

DEFAULT_THRESHOLD = 0.50

DEFAULT_C = 0.10

DEFAULT_RANDOM_STATE = 42


class SecureVisionClassifier:
    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        c: float = DEFAULT_C,
        random_state: int = DEFAULT_RANDOM_STATE,
    ):
        if not 0 < threshold < 1:
            raise ValueError(
                "threshold must be between 0 and 1"
            )

        if c <= 0:
            raise ValueError(
                "C must be positive"
            )

        self.threshold = float(
            threshold
        )

        self.c = float(
            c
        )

        self.random_state = int(
            random_state
        )

        self.pipeline = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=self.c,
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=(
                            self.random_state
                        ),
                    ),
                ),
            ]
        )

        self.is_fitted = False


    @staticmethod
    def validate_features(
        features: np.ndarray,
    ) -> np.ndarray:
        features = np.asarray(
            features,
            dtype=np.float32,
        )

        if features.ndim != 2:
            raise ValueError(
                "Expected feature matrix "
                "[N, 2048]. "
                f"Received: {features.shape}"
            )

        if (
            features.shape[1]
            != FEATURE_DIM
        ):
            raise ValueError(
                "Unexpected feature dimension. "
                f"Expected {FEATURE_DIM}, "
                f"received {features.shape[1]}"
            )

        if not np.isfinite(
            features
        ).all():
            raise ValueError(
                "Features contain NaN or Inf."
            )

        return features


    @staticmethod
    def validate_labels(
        labels: np.ndarray,
    ) -> np.ndarray:
        labels = np.asarray(
            labels,
            dtype=np.int64,
        )

        if labels.ndim != 1:
            raise ValueError(
                "Labels must be 1-dimensional."
            )

        unique_labels = set(
            np.unique(
                labels
            ).tolist()
        )

        if not unique_labels.issubset(
            {0, 1}
        ):
            raise ValueError(
                f"Unexpected labels: "
                f"{unique_labels}"
            )

        if len(
            unique_labels
        ) < 2:
            raise ValueError(
                "Training requires "
                "both classes 0 and 1."
            )

        return labels


    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weight: (
            np.ndarray
            | None
        ) = None,
    ) -> "SecureVisionClassifier":
        features = (
            self.validate_features(
                features
            )
        )

        labels = (
            self.validate_labels(
                labels
            )
        )

        if (
            len(features)
            != len(labels)
        ):
            raise ValueError(
                "Feature and label counts "
                "do not match."
            )

        fit_kwargs = {}

        if sample_weight is not None:
            sample_weight = (
                np.asarray(
                    sample_weight,
                    dtype=np.float64,
                )
            )

            if (
                len(sample_weight)
                != len(labels)
            ):
                raise ValueError(
                    "sample_weight length "
                    "does not match labels."
                )

            if np.any(
                sample_weight <= 0
            ):
                raise ValueError(
                    "All sample weights "
                    "must be positive."
                )

            fit_kwargs[
                "classifier__sample_weight"
            ] = sample_weight

        self.pipeline.fit(
            features,
            labels,
            **fit_kwargs,
        )

        self.is_fitted = True

        return self


    def predict_proba(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(
                "Classifier is not fitted."
            )

        features = (
            self.validate_features(
                features
            )
        )

        probabilities = (
            self.pipeline.predict_proba(
                features
            )
        )

        classifier = (
            self.pipeline.named_steps[
                "classifier"
            ]
        )

        classes = (
            classifier.classes_
        )

        positive_indices = (
            np.where(
                classes == 1
            )[0]
        )

        if len(
            positive_indices
        ) != 1:
            raise RuntimeError(
                "Could not locate "
                "positive class."
            )

        positive_index = int(
            positive_indices[0]
        )

        return (
            probabilities[
                :,
                positive_index,
            ]
            .astype(
                np.float64
            )
        )


    def predict(
        self,
        features: np.ndarray,
        threshold: (
            float
            | None
        ) = None,
    ) -> np.ndarray:
        if threshold is None:
            threshold = (
                self.threshold
            )

        if not 0 < threshold < 1:
            raise ValueError(
                "threshold must be "
                "between 0 and 1"
            )

        probabilities = (
            self.predict_proba(
                features
            )
        )

        return (
            probabilities
            >= threshold
        ).astype(
            np.int64
        )


    def save(
        self,
        path: str | Path,
        metadata: (
            dict[str, Any]
            | None
        ) = None,
    ) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                "Cannot save an "
                "unfitted classifier."
            )

        path = Path(
            path
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "pipeline": (
                self.pipeline
            ),

            "threshold": (
                self.threshold
            ),

            "feature_dim": (
                FEATURE_DIM
            ),

            "c": (
                self.c
            ),

            "random_state": (
                self.random_state
            ),

            "metadata": (
                metadata or {}
            ),
        }

        joblib.dump(
            payload,
            path,
        )


    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "SecureVisionClassifier":
        path = Path(
            path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Classifier not found: "
                f"{path}"
            )

        payload = joblib.load(
            path
        )

        required = [
            "pipeline",
            "threshold",
            "feature_dim",
        ]

        missing = [
            key
            for key in required
            if key not in payload
        ]

        if missing:
            raise RuntimeError(
                "Invalid classifier file. "
                f"Missing: {missing}"
            )

        if (
            int(
                payload[
                    "feature_dim"
                ]
            )
            != FEATURE_DIM
        ):
            raise RuntimeError(
                "Classifier feature "
                "dimension mismatch."
            )

        instance = cls(
            threshold=float(
                payload[
                    "threshold"
                ]
            ),
            c=float(
                payload.get(
                    "c",
                    DEFAULT_C,
                )
            ),
            random_state=int(
                payload.get(
                    "random_state",
                    DEFAULT_RANDOM_STATE,
                )
            ),
        )

        instance.pipeline = (
            payload[
                "pipeline"
            ]
        )

        instance.is_fitted = True

        return instance