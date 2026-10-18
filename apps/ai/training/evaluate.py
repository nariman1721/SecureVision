from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.action_recognition.classifier import (
    SecureVisionClassifier,
)

from training.metrics import (
    aggregate_video_predictions,
    build_threshold_sweep,
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

VALIDATION_FEATURES_FILE = (
    FEATURES_DIR
    / "validation_x3d_features.npz"
)

MODEL_FILE = (
    MODELS_DIR
    / "securevision_logreg.joblib"
)

CLIP_PREDICTIONS_FILE = (
    REPORTS_DIR
    / "validation_clip_predictions.csv"
)

VIDEO_PREDICTIONS_FILE = (
    REPORTS_DIR
    / "validation_video_predictions.csv"
)

THRESHOLD_SWEEP_FILE = (
    REPORTS_DIR
    / "validation_threshold_sweep.csv"
)

EVALUATION_FILE = (
    REPORTS_DIR
    / "classifier_evaluation.json"
)


# =========================================================
# DATA
# =========================================================

def load_validation_features():
    if not VALIDATION_FEATURES_FILE.exists():
        raise FileNotFoundError(
            VALIDATION_FEATURES_FILE
        )

    data = np.load(
        VALIDATION_FEATURES_FILE,
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
            f"Missing arrays: {missing}"
        )

    return {
        key: data[key]
        for key in required
    }


# =========================================================
# CLIP TYPE BREAKDOWN
# =========================================================

def build_clip_type_breakdown(
    dataframe: pd.DataFrame,
) -> dict:
    result = {}

    for clip_type, group in (
        dataframe.groupby(
            "clip_type"
        )
    ):
        count = int(
            len(group)
        )

        predicted_positive = int(
            group[
                "prediction"
            ]
            .eq(1)
            .sum()
        )

        result[
            str(
                clip_type
            )
        ] = {
            "count": (
                count
            ),

            "mean_probability": float(
                group[
                    "probability"
                ].mean()
            ),

            "predicted_positive": (
                predicted_positive
            ),

            "positive_rate": float(
                predicted_positive
                / count
                if count
                else 0.0
            ),
        }

    return result


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 6 "
        "Validation Evaluation"
    )
    print("=" * 72)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing classifier:\n"
            f"{MODEL_FILE}\n\n"
            "Run training.train first."
        )

    data = (
        load_validation_features()
    )

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
    print("VALIDATION DATA")
    print("-" * 72)

    print(
        "X shape:",
        X.shape,
    )

    print(
        "Positive clips:",
        int(
            (y == 1).sum()
        ),
    )

    print(
        "Negative clips:",
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

    # -----------------------------------------------------
    # Classifier
    # -----------------------------------------------------

    classifier = (
        SecureVisionClassifier.load(
            MODEL_FILE
        )
    )

    threshold = (
        classifier.threshold
    )

    print()
    print("CLASSIFIER")
    print("-" * 72)

    print(
        "Model:",
        MODEL_FILE,
    )

    print(
        "Threshold:",
        threshold,
    )

    # -----------------------------------------------------
    # Clip predictions
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
        CLIP_PREDICTIONS_FILE,
        index=False,
    )

    clip_metrics = (
        compute_binary_metrics(
            y,
            probabilities,
            threshold=(
                threshold
            ),
        )
    )

    # -----------------------------------------------------
    # Video aggregation
    # -----------------------------------------------------

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
                threshold
            ),
        )
    )

    video_predictions = (
        video_predictions
        .sort_values(
            "video_id"
        )
        .reset_index(
            drop=True
        )
    )

    video_predictions.to_csv(
        VIDEO_PREDICTIONS_FILE,
        index=False,
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
                threshold
            ),
        )
    )

    # -----------------------------------------------------
    # Threshold diagnostic
    # -----------------------------------------------------

    thresholds = (
        np.arange(
            0.20,
            0.81,
            0.05,
        )
    )

    threshold_sweep = (
        build_threshold_sweep(
            y_true=(
                video_predictions[
                    "true_label"
                ].to_numpy()
            ),
            probabilities=(
                video_predictions[
                    "probability"
                ].to_numpy()
            ),
            thresholds=(
                thresholds
            ),
        )
    )

    threshold_sweep.to_csv(
        THRESHOLD_SWEEP_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Clip type breakdown
    # -----------------------------------------------------

    clip_type_breakdown = (
        build_clip_type_breakdown(
            clip_predictions
        )
    )

    # -----------------------------------------------------
    # Save evaluation
    # -----------------------------------------------------

    evaluation = {
        "classifier": {
            "model_file": (
                str(
                    MODEL_FILE
                )
            ),

            "threshold": (
                float(
                    threshold
                )
            ),

            "feature_dim": (
                int(
                    X.shape[1]
                )
            ),
        },

        "validation": {
            "clips": int(
                len(y)
            ),

            "unique_videos": int(
                len(
                    video_predictions
                )
            ),

            "positive_clips": int(
                (y == 1).sum()
            ),

            "negative_clips": int(
                (y == 0).sum()
            ),
        },

        "clip_level_metrics": (
            clip_metrics
        ),

        "video_level_metrics": (
            video_metrics
        ),

        "clip_type_breakdown": (
            clip_type_breakdown
        ),

        "notes": {
            "video_aggregation": (
                "maximum violation "
                "probability across clips"
            ),

            "threshold_sweep_is_diagnostic": (
                True
            ),

            "validation_size_warning": (
                "Only 8 unique validation "
                "videos; metrics have high variance."
            ),
        },
    }

    EVALUATION_FILE.write_text(
        json.dumps(
            evaluation,
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
    print("CLIP-LEVEL METRICS")
    print("=" * 72)

    for key in [
        "accuracy",
        "balanced_accuracy",
        "precision_violation",
        "recall_violation",
        "f1_violation",
        "f2_violation",
        "specificity",
        "roc_auc",
        "pr_auc",
    ]:
        value = (
            clip_metrics[
                key
            ]
        )

        print(
            f"{key:24}",
            (
                round(
                    float(value),
                    4,
                )
                if value is not None
                else None
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
    print("VIDEO-LEVEL METRICS")
    print("=" * 72)

    for key in [
        "accuracy",
        "balanced_accuracy",
        "precision_violation",
        "recall_violation",
        "f1_violation",
        "f2_violation",
        "specificity",
        "roc_auc",
        "pr_auc",
    ]:
        value = (
            video_metrics[
                key
            ]
        )

        print(
            f"{key:24}",
            (
                round(
                    float(value),
                    4,
                )
                if value is not None
                else None
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
    print("VIDEO PREDICTIONS")
    print("=" * 72)

    print(
        video_predictions.to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("CLIP TYPE BREAKDOWN")
    print("=" * 72)

    for (
        clip_type,
        values,
    ) in (
        clip_type_breakdown.items()
    ):
        print()
        print(
            clip_type
        )

        for key, value in (
            values.items()
        ):
            print(
                f"  {key:20} {value}"
            )

    print()
    print("=" * 72)
    print("VIDEO THRESHOLD SWEEP")
    print("=" * 72)

    print(
        threshold_sweep.round(
            4
        ).to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    print(CLIP_PREDICTIONS_FILE)
    print(VIDEO_PREDICTIONS_FILE)
    print(THRESHOLD_SWEEP_FILE)
    print(EVALUATION_FILE)

    print()
    print(
        "✅ Validation evaluation complete."
    )


if __name__ == "__main__":
    main()