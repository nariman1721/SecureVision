from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

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

VALIDATION_FEATURES_FILE = (
    FEATURES_DIR
    / "validation_x3d_multiwindow_features.npz"
)

FINAL_MODEL_FILE = (
    MODELS_DIR
    / "securevision_final_classifier.joblib"
)

CLIP_PREDICTIONS_FILE = (
    REPORTS_DIR
    / "final_validation_clip_predictions.csv"
)

VIDEO_PREDICTIONS_FILE = (
    REPORTS_DIR
    / "final_validation_video_predictions.csv"
)

EVALUATION_FILE = (
    REPORTS_DIR
    / "final_validation_evaluation.json"
)


# =========================================================
# LOAD MODEL
# =========================================================

def load_final_model():
    if not FINAL_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Final model not found:\n"
            f"{FINAL_MODEL_FILE}"
        )

    payload = joblib.load(
        FINAL_MODEL_FILE
    )

    required = [
        "pipeline",
        "threshold",
        "feature_strategy",
        "input_feature_dim",
        "pca_components",
        "C",
        "weight_strategy",
        "x3d_windows",
        "x3d_pooling",
    ]

    missing = [
        key
        for key in required
        if key not in payload
    ]

    if missing:
        raise RuntimeError(
            f"Model payload missing: "
            f"{missing}"
        )

    return payload


# =========================================================
# LOAD VALIDATION FEATURES
# =========================================================

def load_validation_data():
    if not VALIDATION_FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Validation features not found:\n"
            f"{VALIDATION_FEATURES_FILE}"
        )

    data = np.load(
        VALIDATION_FEATURES_FILE,
        allow_pickle=False,
    )

    required = [
        "X_mean",
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
        raise RuntimeError(
            f"Validation feature file "
            f"is missing: {missing}"
        )

    return data


# =========================================================
# POSITIVE PROBABILITY
# =========================================================

def get_positive_probability(
    pipeline,
    X: np.ndarray,
) -> np.ndarray:
    probabilities = (
        pipeline.predict_proba(
            X
        )
    )

    classifier = (
        pipeline.named_steps[
            "classifier"
        ]
    )

    positive_indices = np.where(
        classifier.classes_
        == 1
    )[0]

    if len(
        positive_indices
    ) != 1:
        raise RuntimeError(
            "Could not locate "
            "positive class 1."
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
        result[
            str(
                clip_type
            )
        ] = {
            "count": int(
                len(group)
            ),

            "mean_probability": float(
                group[
                    "probability"
                ].mean()
            ),

            "predicted_positive": int(
                group[
                    "prediction"
                ]
                .eq(1)
                .sum()
            ),
        }

    return result


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 7E "
        "Final Development Validation"
    )
    print("=" * 72)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Load frozen model
    # -----------------------------------------------------

    payload = load_final_model()

    pipeline = (
        payload[
            "pipeline"
        ]
    )

    threshold = float(
        payload[
            "threshold"
        ]
    )

    feature_strategy = str(
        payload[
            "feature_strategy"
        ]
    )

    pca_components = int(
        payload[
            "pca_components"
        ]
    )

    c_value = float(
        payload[
            "C"
        ]
    )

    weight_strategy = str(
        payload[
            "weight_strategy"
        ]
    )

    print()
    print("FROZEN MODEL")
    print("-" * 72)

    print(
        "Feature strategy:",
        feature_strategy,
    )

    print(
        "X3D windows:",
        payload[
            "x3d_windows"
        ],
    )

    print(
        "Pooling:",
        payload[
            "x3d_pooling"
        ],
    )

    print(
        "PCA components:",
        pca_components,
    )

    print(
        "C:",
        c_value,
    )

    print(
        "Weight strategy:",
        weight_strategy,
    )

    print(
        "Threshold:",
        threshold,
    )

    # -----------------------------------------------------
    # Guard against accidental config changes
    # -----------------------------------------------------

    if feature_strategy != "multi_mean":
        raise RuntimeError(
            "Expected feature strategy "
            "'multi_mean'."
        )

    if pca_components != 16:
        raise RuntimeError(
            "Expected PCA=16."
        )

    if not np.isclose(
        c_value,
        0.0003,
    ):
        raise RuntimeError(
            "Expected C=0.0003."
        )

    if weight_strategy != "video_only":
        raise RuntimeError(
            "Expected video_only "
            "weight strategy."
        )

    if not np.isclose(
        threshold,
        0.50,
    ):
        raise RuntimeError(
            "Expected threshold=0.50."
        )

    # -----------------------------------------------------
    # Validation data
    # -----------------------------------------------------

    data = load_validation_data()

    X = (
        data[
            "X_mean"
        ]
        .astype(
            np.float32
        )
    )

    y = (
        data[
            "y"
        ]
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

    if X.shape != (
        len(y),
        2048,
    ):
        raise RuntimeError(
            f"Unexpected validation "
            f"feature shape: {X.shape}"
        )

    if not np.isfinite(
        X
    ).all():
        raise RuntimeError(
            "Validation features "
            "contain NaN or Inf."
        )

    print()
    print("DEVELOPMENT VALIDATION DATA")
    print("-" * 72)

    print(
        "X shape:",
        X.shape,
    )

    print(
        "Clips:",
        len(y),
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
    # IMPORTANT:
    # NO TRAINING / NO FIT HERE
    # -----------------------------------------------------

    print()
    print("INFERENCE")
    print("-" * 72)

    print(
        "Calling pipeline.fit(): False"
    )

    print(
        "Changing PCA: False"
    )

    print(
        "Changing C: False"
    )

    print(
        "Changing threshold: False"
    )

    probabilities = (
        get_positive_probability(
            pipeline,
            X,
        )
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int64
    )

    # -----------------------------------------------------
    # Clip metrics
    # -----------------------------------------------------

    clip_metrics = (
        compute_binary_metrics(
            y_true=y,
            probabilities=(
                probabilities
            ),
            threshold=(
                threshold
            ),
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

                "correct": (
                    predictions
                    == y
                ),
            }
        )
    )

    clip_predictions.to_csv(
        CLIP_PREDICTIONS_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Video-level aggregation
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

    video_predictions[
        "correct"
    ] = (
        video_predictions[
            "prediction"
        ]
        == video_predictions[
            "true_label"
        ]
    )

    video_predictions[
        "error_type"
    ] = "correct"

    video_predictions.loc[
        (
            video_predictions[
                "true_label"
            ].eq(0)
            &
            video_predictions[
                "prediction"
            ].eq(1)
        ),
        "error_type",
    ] = "false_positive"

    video_predictions.loc[
        (
            video_predictions[
                "true_label"
            ].eq(1)
            &
            video_predictions[
                "prediction"
            ].eq(0)
        ),
        "error_type",
    ] = "false_negative"

    video_predictions = (
        video_predictions
        .sort_values(
            by=[
                "video_id"
            ]
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
            threshold=(
                threshold
            ),
        )
    )

    clip_type_breakdown = (
        build_clip_type_breakdown(
            clip_predictions
        )
    )

    # -----------------------------------------------------
    # Evaluation JSON
    # -----------------------------------------------------

    evaluation = {
        "evaluation_type": (
            "reserved development validation"
        ),

        "important_note": (
            "These videos were not used in "
            "pipeline.fit(), but their earlier "
            "baseline results were inspected "
            "during development. Competition "
            "hidden videos remain the true "
            "unseen final test."
        ),

        "model": {
            "feature_strategy": (
                feature_strategy
            ),

            "x3d_windows": (
                payload[
                    "x3d_windows"
                ]
            ),

            "pooling": (
                payload[
                    "x3d_pooling"
                ]
            ),

            "pca_components": (
                pca_components
            ),

            "C": (
                c_value
            ),

            "weight_strategy": (
                weight_strategy
            ),

            "threshold": (
                threshold
            ),
        },

        "validation": {
            "clips": int(
                len(y)
            ),

            "positive_clips": int(
                (y == 1).sum()
            ),

            "negative_clips": int(
                (y == 0).sum()
            ),

            "unique_videos": int(
                len(
                    video_predictions
                )
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

        "inference_guards": {
            "pipeline_fit_called": False,
            "pca_changed": False,
            "C_changed": False,
            "threshold_changed": False,
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
        video_predictions[
            [
                "video_id",
                "true_label",
                "probability",
                "prediction",
                "correct",
                "error_type",
            ]
        ]
        .to_string(
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
                f"  {key:20} "
                f"{value}"
            )

    print()
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    print(
        CLIP_PREDICTIONS_FILE
    )

    print(
        VIDEO_PREDICTIONS_FILE
    )

    print(
        EVALUATION_FILE
    )

    print()
    print(
        "✅ Final candidate evaluated."
    )

    print(
        "✅ No model fitting performed."
    )

    print(
        "✅ Frozen threshold preserved."
    )


if __name__ == "__main__":
    main()