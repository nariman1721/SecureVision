from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.action_recognition.classifier import (
    DEFAULT_THRESHOLD,
    SecureVisionClassifier,
)

from training.metrics import (
    aggregate_video_predictions,
    compute_binary_metrics,
)


# =========================================================
# PATHS
# =========================================================

AI_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

FEATURES_DIR = (
    AI_ROOT
    / "data"
    / "processed"
    / "features"
)

MODELS_DIR = (
    AI_ROOT
    / "models"
)

REPORTS_DIR = (
    AI_ROOT
    / "reports"
)

TRAIN_FEATURES_FILE = (
    FEATURES_DIR
    / "train_x3d_features.npz"
)

MODEL_FILE = (
    MODELS_DIR
    / "securevision_logreg.joblib"
)

MODEL_METADATA_FILE = (
    MODELS_DIR
    / "securevision_logreg.json"
)

TRAIN_PREDICTIONS_FILE = (
    REPORTS_DIR
    / "train_classifier_predictions.csv"
)

TRAIN_VIDEO_PREDICTIONS_FILE = (
    REPORTS_DIR
    / "train_video_predictions.csv"
)

TRAIN_SUMMARY_FILE = (
    REPORTS_DIR
    / "classifier_training_summary.json"
)


# =========================================================
# DATA
# =========================================================

def load_features():
    if not TRAIN_FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Missing features:\n"
            f"{TRAIN_FEATURES_FILE}\n\n"
            "Run Phase 5B first."
        )

    data = np.load(
        TRAIN_FEATURES_FILE,
        allow_pickle=False,
    )

    required = [
        "X",
        "y",
        "clip_ids",
        "content_ids",
        "video_ids",
        "clip_types",
    ]

    missing = [
        key
        for key in required
        if key not in data.files
    ]

    if missing:
        raise ValueError(
            f"Missing arrays: "
            f"{missing}"
        )

    return {
        key: data[key]
        for key in required
    }


# =========================================================
# VIDEO-BALANCED SAMPLE WEIGHTS
# =========================================================

def build_video_balanced_weights(
    content_ids,
) -> np.ndarray:
    content_ids = (
        np.asarray(
            content_ids
        )
        .astype(str)
    )

    counts = (
        pd.Series(
            content_ids
        )
        .value_counts()
    )

    weights = np.array(
        [
            1.0
            / float(
                counts[
                    content_id
                ]
            )
            for content_id
            in content_ids
        ],
        dtype=np.float64,
    )

    # Normalize mean weight to 1.
    weights = (
        weights
        * (
            len(weights)
            / weights.sum()
        )
    )

    return weights


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 6 "
        "Binary Classifier Training"
    )
    print("=" * 72)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_features()

    X = (
        data["X"]
        .astype(
            np.float32
        )
    )

    y = (
        data["y"]
        .astype(
            np.int64
        )
    )

    clip_ids = (
        data[
            "clip_ids"
        ]
        .astype(str)
    )

    content_ids = (
        data[
            "content_ids"
        ]
        .astype(str)
    )

    video_ids = (
        data[
            "video_ids"
        ]
        .astype(str)
    )

    clip_types = (
        data[
            "clip_types"
        ]
        .astype(str)
    )

    print()
    print("TRAINING DATA")
    print("-" * 72)

    print(
        "X shape:",
        X.shape,
    )

    print(
        "y shape:",
        y.shape,
    )

    print(
        "Positive:",
        int(
            (y == 1).sum()
        ),
    )

    print(
        "Negative:",
        int(
            (y == 0).sum()
        ),
    )

    print(
        "Unique videos:",
        len(
            np.unique(
                content_ids
            )
        ),
    )

    if not np.isfinite(
        X
    ).all():
        raise RuntimeError(
            "Training features "
            "contain NaN or Inf."
        )

    # -----------------------------------------------------
    # Sample weighting
    # -----------------------------------------------------

    sample_weights = (
        build_video_balanced_weights(
            content_ids
        )
    )

    print()
    print("SAMPLE WEIGHTS")
    print("-" * 72)

    print(
        "Strategy:",
        "inverse clips per content_id",
    )

    print(
        "Mean:",
        round(
            float(
                sample_weights.mean()
            ),
            6,
        ),
    )

    print(
        "Min:",
        round(
            float(
                sample_weights.min()
            ),
            6,
        ),
    )

    print(
        "Max:",
        round(
            float(
                sample_weights.max()
            ),
            6,
        ),
    )

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    classifier = (
        SecureVisionClassifier(
            threshold=(
                DEFAULT_THRESHOLD
            ),
            c=0.10,
            random_state=42,
        )
    )

    print()
    print("TRAINING MODEL")
    print("-" * 72)

    print(
        "Model:",
        "StandardScaler + LogisticRegression",
    )

    print(
        "C:",
        classifier.c,
    )

    print(
        "class_weight:",
        "balanced",
    )

    print(
        "threshold:",
        classifier.threshold,
    )

    classifier.fit(
        X,
        y,
        sample_weight=(
            sample_weights
        ),
    )

    # -----------------------------------------------------
    # Training predictions
    # -----------------------------------------------------

    probabilities = (
        classifier.predict_proba(
            X
        )
    )

    predictions = (
        classifier.predict(
            X
        )
    )

    clip_metrics = (
        compute_binary_metrics(
            y,
            probabilities,
            threshold=(
                classifier.threshold
            ),
        )
    )

    video_predictions = (
        aggregate_video_predictions(
            y_true=y,
            probabilities=(
                probabilities
            ),
            content_ids=(
                content_ids
            ),
            video_ids=(
                video_ids
            ),
            threshold=(
                classifier.threshold
            ),
        )
    )

    video_metrics = (
        compute_binary_metrics(
            video_predictions[
                "true_label"
            ].to_numpy(),
            video_predictions[
                "probability"
            ].to_numpy(),
            threshold=(
                classifier.threshold
            ),
        )
    )

    # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------

    clip_predictions = (
        pd.DataFrame(
            {
                "clip_id": (
                    clip_ids
                ),

                "content_id": (
                    content_ids
                ),

                "video_id": (
                    video_ids
                ),

                "clip_type": (
                    clip_types
                ),

                "true_label": (
                    y
                ),

                "probability": (
                    probabilities
                ),

                "prediction": (
                    predictions
                ),
            }
        )
    )

    clip_predictions.to_csv(
        TRAIN_PREDICTIONS_FILE,
        index=False,
    )

    video_predictions.to_csv(
        TRAIN_VIDEO_PREDICTIONS_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    model_metadata = {
        "model_type": (
            "logistic_regression"
        ),

        "feature_extractor": (
            "x3d_xs_kinetics400"
        ),

        "feature_dim": 2048,

        "threshold": (
            classifier.threshold
        ),

        "regularization_c": (
            classifier.c
        ),

        "class_weight": (
            "balanced"
        ),

        "sample_weight_strategy": (
            "inverse_clip_count_per_content_id"
        ),

        "train_clips": int(
            len(y)
        ),

        "train_unique_videos": int(
            len(
                np.unique(
                    content_ids
                )
            )
        ),
    }

    classifier.save(
        MODEL_FILE,
        metadata=(
            model_metadata
        ),
    )

    MODEL_METADATA_FILE.write_text(
        json.dumps(
            model_metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary = {
        "model": (
            model_metadata
        ),

        "training_clip_metrics": (
            clip_metrics
        ),

        "training_video_metrics": (
            video_metrics
        ),

        "sample_weights": {
            "mean": float(
                sample_weights.mean()
            ),

            "min": float(
                sample_weights.min()
            ),

            "max": float(
                sample_weights.max()
            ),
        },
    }

    TRAIN_SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Console output
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print("TRAIN CLIP METRICS")
    print("=" * 72)

    for key in [
        "accuracy",
        "balanced_accuracy",
        "precision_violation",
        "recall_violation",
        "f1_violation",
        "f2_violation",
        "roc_auc",
        "pr_auc",
    ]:
        print(
            f"{key:24}",
            round(
                float(
                    clip_metrics[
                        key
                    ]
                ),
                4,
            ),
        )

    print(
        "confusion_matrix:",
        clip_metrics[
            "confusion_matrix"
        ],
    )

    print()
    print("=" * 72)
    print("TRAIN VIDEO METRICS")
    print("=" * 72)

    for key in [
        "accuracy",
        "balanced_accuracy",
        "precision_violation",
        "recall_violation",
        "f1_violation",
        "f2_violation",
        "roc_auc",
        "pr_auc",
    ]:
        print(
            f"{key:24}",
            round(
                float(
                    video_metrics[
                        key
                    ]
                ),
                4,
            ),
        )

    print(
        "confusion_matrix:",
        video_metrics[
            "confusion_matrix"
        ],
    )

    print()
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    print(MODEL_FILE)
    print(MODEL_METADATA_FILE)
    print(TRAIN_SUMMARY_FILE)
    print(TRAIN_PREDICTIONS_FILE)
    print(TRAIN_VIDEO_PREDICTIONS_FILE)

    print()
    print(
        "✅ SecureVision baseline "
        "classifier trained."
    )


if __name__ == "__main__":
    main()