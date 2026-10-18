from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    / "train_x3d_multiwindow_features.npz"
)

BEST_CONFIG_FILE = (
    REPORTS_DIR
    / "feature_strategy_best_config.json"
)

FINAL_MODEL_FILE = (
    MODELS_DIR
    / "securevision_final_classifier.joblib"
)

FINAL_METADATA_FILE = (
    MODELS_DIR
    / "securevision_final_classifier.json"
)

TRAIN_CLIP_PREDICTIONS_FILE = (
    REPORTS_DIR
    / "final_model_train_clip_predictions.csv"
)

TRAIN_VIDEO_PREDICTIONS_FILE = (
    REPORTS_DIR
    / "final_model_train_video_predictions.csv"
)

TRAIN_SUMMARY_FILE = (
    REPORTS_DIR
    / "final_model_training_summary.json"
)


# =========================================================
# HASH
# =========================================================

def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


# =========================================================
# CONFIG
# =========================================================

def load_best_config() -> dict:
    if not BEST_CONFIG_FILE.exists():
        raise FileNotFoundError(
            f"Missing Phase 7C config:\n"
            f"{BEST_CONFIG_FILE}"
        )

    config = json.loads(
        BEST_CONFIG_FILE.read_text(
            encoding="utf-8"
        )
    )

    required = [
        "feature_strategy",
        "pca_components",
        "C",
        "weight_strategy",
        "threshold",
    ]

    missing = [
        key
        for key in required
        if key not in config
    ]

    if missing:
        raise ValueError(
            f"Best config is missing: "
            f"{missing}"
        )

    return config


# =========================================================
# FEATURES
# =========================================================

def load_training_data():
    if not TRAIN_FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Missing features:\n"
            f"{TRAIN_FEATURES_FILE}"
        )

    data = np.load(
        TRAIN_FEATURES_FILE,
        allow_pickle=False,
    )

    required = [
        "X_mean",
        "X_max",
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
            f"Missing feature arrays: "
            f"{missing}"
        )

    return data


def select_features(
    data,
    feature_strategy: str,
) -> np.ndarray:
    if feature_strategy == "multi_mean":
        X = data[
            "X_mean"
        ]

    elif feature_strategy == "multi_max":
        X = data[
            "X_max"
        ]

    else:
        raise ValueError(
            "Final training expects "
            "a multi-window feature strategy. "
            f"Received: {feature_strategy}"
        )

    X = X.astype(
        np.float32
    )

    if X.ndim != 2:
        raise ValueError(
            f"Expected 2D feature matrix. "
            f"Received: {X.shape}"
        )

    if X.shape[1] != 2048:
        raise ValueError(
            f"Expected 2048 features. "
            f"Received: {X.shape[1]}"
        )

    if not np.isfinite(
        X
    ).all():
        raise ValueError(
            "Training features contain "
            "NaN or Inf."
        )

    return X


# =========================================================
# VIDEO SAMPLE WEIGHTS
# =========================================================

def build_video_balanced_weights(
    content_ids: np.ndarray,
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

    # Normalize so mean sample weight = 1.
    weights *= (
        len(weights)
        / weights.sum()
    )

    return weights


# =========================================================
# MODEL
# =========================================================

def build_final_pipeline(
    pca_components: int,
    c_value: float,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),

            (
                "pca",
                PCA(
                    n_components=int(
                        pca_components
                    ),
                    svd_solver="full",
                ),
            ),

            (
                "classifier",
                LogisticRegression(
                    C=float(
                        c_value
                    ),
                    solver="liblinear",
                    class_weight=None,
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def positive_probability(
    pipeline: Pipeline,
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
            "Positive class 1 not found."
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
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 7D "
        "Final Candidate Training"
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

    config = load_best_config()

    feature_strategy = str(
        config[
            "feature_strategy"
        ]
    )

    pca_components = int(
        config[
            "pca_components"
        ]
    )

    c_value = float(
        config[
            "C"
        ]
    )

    weight_strategy = str(
        config[
            "weight_strategy"
        ]
    )

    threshold = float(
        config[
            "threshold"
        ]
    )

    print()
    print("SELECTED CONFIGURATION")
    print("-" * 72)

    print(
        "Feature strategy:",
        feature_strategy,
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

    print(
        "Holdout validation used:",
        config.get(
            "holdout_validation_used",
            None,
        ),
    )

    if feature_strategy != "multi_mean":
        raise RuntimeError(
            "Expected Phase 7C winner "
            "'multi_mean'. "
            f"Received: {feature_strategy}"
        )

    if pca_components != 16:
        raise RuntimeError(
            "Expected PCA=16 from "
            "Phase 7C."
        )

    if not np.isclose(
        c_value,
        0.0003,
    ):
        raise RuntimeError(
            "Expected C=0.0003 from "
            "Phase 7C."
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
    # Load data
    # -----------------------------------------------------

    data = load_training_data()

    X = select_features(
        data,
        feature_strategy=(
            feature_strategy
        ),
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

    if len(X) != len(y):
        raise RuntimeError(
            "Feature/label count mismatch."
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
    # Sample weights
    # -----------------------------------------------------

    sample_weights = (
        build_video_balanced_weights(
            content_ids
        )
    )

    print()
    print("VIDEO SAMPLE WEIGHTS")
    print("-" * 72)

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
    # Build + fit
    # -----------------------------------------------------

    pipeline = build_final_pipeline(
        pca_components=(
            pca_components
        ),
        c_value=(
            c_value
        ),
    )

    print()
    print("TRAINING")
    print("-" * 72)

    print(
        "Pipeline:",
        "StandardScaler → PCA → LogisticRegression",
    )

    pipeline.fit(
        X,
        y,
        classifier__sample_weight=(
            sample_weights
        ),
    )

    # -----------------------------------------------------
    # PCA sanity
    # -----------------------------------------------------

    pca = (
        pipeline.named_steps[
            "pca"
        ]
    )

    explained_variance = float(
        pca.explained_variance_ratio_.sum()
    )

    print()
    print("PCA")
    print("-" * 72)

    print(
        "Components:",
        pca.n_components_,
    )

    print(
        "Explained variance ratio:",
        round(
            explained_variance,
            6,
        ),
    )

    # -----------------------------------------------------
    # Training predictions
    # -----------------------------------------------------

    probabilities = (
        positive_probability(
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

    # -----------------------------------------------------
    # Predictions CSV
    # -----------------------------------------------------

    clip_predictions = pd.DataFrame(
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

    clip_predictions.to_csv(
        TRAIN_CLIP_PREDICTIONS_FILE,
        index=False,
    )

    video_predictions.to_csv(
        TRAIN_VIDEO_PREDICTIONS_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Save model
    # -----------------------------------------------------

    feature_file_sha256 = (
        sha256_file(
            TRAIN_FEATURES_FILE
        )
    )

    payload = {
        "pipeline": (
            pipeline
        ),

        "threshold": (
            threshold
        ),

        "feature_strategy": (
            feature_strategy
        ),

        "input_feature_dim": 2048,

        "pca_components": (
            pca_components
        ),

        "C": (
            c_value
        ),

        "weight_strategy": (
            weight_strategy
        ),

        "x3d_windows": 3,

        "x3d_pooling": (
            "mean"
        ),

        "random_state": 42,

        "metadata": {
            "source_config": str(
                BEST_CONFIG_FILE
            ),

            "training_feature_file": str(
                TRAIN_FEATURES_FILE
            ),

            "training_feature_sha256": (
                feature_file_sha256
            ),

            "training_clips": int(
                len(y)
            ),

            "training_unique_videos": int(
                len(
                    np.unique(
                        content_ids
                    )
                )
            ),
        },
    }

    joblib.dump(
        payload,
        FINAL_MODEL_FILE,
    )

    final_model_sha256 = (
        sha256_file(
            FINAL_MODEL_FILE
        )
    )

    # -----------------------------------------------------
    # Reload test
    # -----------------------------------------------------

    reloaded = joblib.load(
        FINAL_MODEL_FILE
    )

    reloaded_pipeline = (
        reloaded[
            "pipeline"
        ]
    )

    reloaded_probabilities = (
        positive_probability(
            reloaded_pipeline,
            X,
        )
    )

    max_probability_difference = float(
        np.max(
            np.abs(
                probabilities
                - reloaded_probabilities
            )
        )
    )

    reload_consistent = bool(
        np.allclose(
            probabilities,
            reloaded_probabilities,
            atol=1e-12,
            rtol=1e-12,
        )
    )

    if not reload_consistent:
        raise RuntimeError(
            "Reloaded model predictions "
            "do not match original model."
        )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    model_metadata = {
        "model_name": (
            "SecureVision Final Candidate"
        ),

        "model_type": (
            "StandardScaler + PCA + "
            "LogisticRegression"
        ),

        "feature_strategy": (
            feature_strategy
        ),

        "x3d_windows": 3,

        "x3d_pooling": (
            "mean"
        ),

        "input_feature_dim": 2048,

        "pca_components": (
            pca_components
        ),

        "pca_explained_variance_ratio": (
            explained_variance
        ),

        "C": (
            c_value
        ),

        "class_weight": None,

        "sample_weight_strategy": (
            "inverse_clip_count_per_content_id"
        ),

        "threshold": (
            threshold
        ),

        "training_clips": int(
            len(y)
        ),

        "training_positive_clips": int(
            (y == 1).sum()
        ),

        "training_negative_clips": int(
            (y == 0).sum()
        ),

        "training_unique_videos": int(
            len(
                np.unique(
                    content_ids
                )
            )
        ),

        "training_feature_file": str(
            TRAIN_FEATURES_FILE
        ),

        "training_feature_sha256": (
            feature_file_sha256
        ),

        "model_file": str(
            FINAL_MODEL_FILE
        ),

        "model_sha256": (
            final_model_sha256
        ),

        "reload_consistent": (
            reload_consistent
        ),

        "max_reload_probability_difference": (
            max_probability_difference
        ),

        "holdout_validation_used_for_training": (
            False
        ),
    }

    FINAL_METADATA_FILE.write_text(
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

        "important_note": (
            "Training metrics are sanity checks only. "
            "Generalization must be measured on the "
            "untouched holdout set in Phase 7E."
        ),
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
    # Console
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
        value = (
            clip_metrics[
                key
            ]
        )

        print(
            f"{key:24}",
            (
                round(
                    float(
                        value
                    ),
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
        value = (
            video_metrics[
                key
            ]
        )

        print(
            f"{key:24}",
            (
                round(
                    float(
                        value
                    ),
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
    print("MODEL SAVE / RELOAD CHECK")
    print("=" * 72)

    print(
        "Model:",
        FINAL_MODEL_FILE,
    )

    print(
        "SHA256:",
        final_model_sha256,
    )

    print(
        "Reload consistent:",
        reload_consistent,
    )

    print(
        "Max probability difference:",
        max_probability_difference,
    )

    print()
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    print(
        FINAL_MODEL_FILE
    )

    print(
        FINAL_METADATA_FILE
    )

    print(
        TRAIN_SUMMARY_FILE
    )

    print(
        TRAIN_CLIP_PREDICTIONS_FILE
    )

    print(
        TRAIN_VIDEO_PREDICTIONS_FILE
    )

    print()
    print(
        "Holdout validation used "
        "for training: False"
    )

    print()
    print(
        "✅ Final candidate trained "
        "on all 32 training videos."
    )

    print(
        "✅ Model serialization "
        "verified."
    )

    print(
        "✅ Ready for Phase 7E "
        "untouched holdout evaluation."
    )


if __name__ == "__main__":
    main()