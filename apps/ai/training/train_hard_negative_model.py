from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.model.loader import load_model
from app.model.multiwindow import (
    extract_multiwindow_feature_set,
    prepare_multiwindow_batch_from_manifest,
)

from training.diagnose_temporal_signal import (
    load_train_manifest,
)


AI_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

PROCESSED_DIR = (
    AI_ROOT
    / "data"
    / "processed"
)

MODELS_DIR = (
    AI_ROOT
    / "models"
)

REPORTS_DIR = (
    AI_ROOT
    / "reports"
)

HARDNEG_FEATURES_FILE = (
    PROCESSED_DIR
    / "hard_negative_multiwindow_features.npz"
)

BASE_FEATURE_CACHE = (
    PROCESSED_DIR
    / "hardneg_base_train_features.npz"
)

MODEL_FILE = (
    MODELS_DIR
    / "securevision_hardneg_classifier.joblib"
)

MODEL_JSON_FILE = (
    MODELS_DIR
    / "securevision_hardneg_classifier.json"
)

TRAINING_REPORT_FILE = (
    REPORTS_DIR
    / "hardneg_model_training_summary.json"
)


PCA_COMPONENTS = 16
LOGREG_C = 0.0003
THRESHOLD = 0.50

# Gives the mined examples additional influence
# without changing the model architecture.
HARD_NEGATIVE_BOOST = 2.0


def file_sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:
        while True:
            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def extract_base_features(
    manifest: pd.DataFrame,
):
    print()
    print("EXTRACTING ORIGINAL TRAIN FEATURES")
    print("-" * 72)

    model, device = (
        load_model(
            pretrained=True
        )
    )

    print(
        "Device:",
        device,
    )

    features = []

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(
            manifest
        ),
        desc="base train features",
    ):
        batch, _ = (
            prepare_multiwindow_batch_from_manifest(
                row=row.to_dict(),
                ai_root=AI_ROOT,
                device=device,
                num_windows=3,
            )
        )

        feature_set = (
            extract_multiwindow_feature_set(
                model,
                batch,
            )
        )

        feature = (
            feature_set[
                "mean"
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
            .reshape(
                -1
            )
        )

        if feature.shape != (
            2048,
        ):
            raise RuntimeError(
                f"Invalid feature shape: "
                f"{feature.shape}"
            )

        features.append(
            feature
        )

    X = np.stack(
        features,
        axis=0,
    )

    y = (
        manifest[
            "label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    np.savez_compressed(
        BASE_FEATURE_CACHE,
        X=X,
        y=y,

        clip_ids=(
            manifest[
                "clip_id"
            ]
            .astype(str)
            .to_numpy()
        ),

        content_ids=(
            manifest[
                "content_id"
            ]
            .astype(str)
            .to_numpy()
        ),

        video_ids=(
            manifest[
                "canonical_video_id"
            ]
            .astype(str)
            .to_numpy()
        ),

        clip_types=(
            manifest[
                "clip_type"
            ]
            .astype(str)
            .to_numpy()
        ),
    )

    return (
        X,
        y,
        manifest[
            "content_id"
        ]
        .astype(str)
        .to_numpy(),
        manifest[
            "clip_type"
        ]
        .astype(str)
        .to_numpy(),
    )


def load_base_features(
    manifest: pd.DataFrame,
):
    if not BASE_FEATURE_CACHE.exists():
        return (
            extract_base_features(
                manifest
            )
        )

    print()
    print(
        "Using cached original "
        "train features:"
    )

    print(
        BASE_FEATURE_CACHE
    )

    data = np.load(
        BASE_FEATURE_CACHE,
        allow_pickle=True,
    )

    X = (
        data[
            "X"
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

    content_ids = (
        data[
            "content_ids"
        ]
        .astype(str)
    )

    clip_types = (
        data[
            "clip_types"
        ]
        .astype(str)
    )

    if len(X) != len(
        manifest
    ):
        raise RuntimeError(
            "Base feature cache size does "
            "not match train manifest. "
            "Delete the cache and rerun."
        )

    return (
        X,
        y,
        content_ids,
        clip_types,
    )


def build_sample_weights(
    content_ids: np.ndarray,
    clip_types: np.ndarray,
) -> np.ndarray:
    dataframe = pd.DataFrame(
        {
            "content_id": (
                content_ids
            ),

            "clip_type": (
                clip_types
            ),
        }
    )

    counts = (
        dataframe[
            "content_id"
        ]
        .value_counts()
    )

    weights = np.asarray(
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

    hard_mask = (
        clip_types
        == "hard_negative_normal"
    )

    weights[
        hard_mask
    ] *= (
        HARD_NEGATIVE_BOOST
    )

    weights /= (
        weights.mean()
    )

    return weights


def metric_summary(
    y_true,
    probability,
):
    prediction = (
        probability
        >= THRESHOLD
    ).astype(
        np.int64
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            prediction,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
    )

    specificity = (
        tn
        / (
            tn
            + fp
        )
        if (
            tn
            + fp
        )
        else 0.0
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                prediction,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                prediction,
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),

        "specificity": float(
            specificity
        ),

        "f1": float(
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                probability,
            )
        ),

        "confusion_matrix": {
            "tn": int(
                tn
            ),
            "fp": int(
                fp
            ),
            "fn": int(
                fn
            ),
            "tp": int(
                tp
            ),
        },
    }


def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 8G-B "
        "Hard Negative Model Training"
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

    if not HARDNEG_FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n"
            f"{HARDNEG_FEATURES_FILE}\n"
            f"Run mine_hard_negatives first."
        )

    manifest = (
        load_train_manifest()
        .reset_index(
            drop=True
        )
    )

    (
        X_base,
        y_base,
        content_base,
        type_base,
    ) = load_base_features(
        manifest
    )

    hard_data = np.load(
        HARDNEG_FEATURES_FILE,
        allow_pickle=True,
    )

    X_hard = (
        hard_data[
            "X"
        ]
        .astype(
            np.float32
        )
    )

    y_hard = (
        hard_data[
            "y"
        ]
        .astype(
            np.int64
        )
    )

    content_hard = (
        hard_data[
            "content_ids"
        ]
        .astype(str)
    )

    type_hard = np.asarray(
        [
            "hard_negative_normal"
            for _ in range(
                len(
                    X_hard
                )
            )
        ],
        dtype=str,
    )

    X = np.concatenate(
        [
            X_base,
            X_hard,
        ],
        axis=0,
    )

    y = np.concatenate(
        [
            y_base,
            y_hard,
        ]
    )

    content_ids = np.concatenate(
        [
            content_base,
            content_hard,
        ]
    )

    clip_types = np.concatenate(
        [
            type_base,
            type_hard,
        ]
    )

    if not np.isfinite(
        X
    ).all():
        raise RuntimeError(
            "Non-finite features detected."
        )

    print()
    print("AUGMENTED DATASET")
    print("-" * 72)

    print(
        "Original clips:",
        len(
            X_base
        ),
    )

    print(
        "Hard negatives:",
        len(
            X_hard
        ),
    )

    print(
        "Total clips:",
        len(
            X
        ),
    )

    print(
        "Positive:",
        int(
            (
                y == 1
            ).sum()
        ),
    )

    print(
        "Negative:",
        int(
            (
                y == 0
            ).sum()
        ),
    )

    print(
        "Feature shape:",
        X.shape,
    )

    sample_weights = (
        build_sample_weights(
            content_ids,
            clip_types,
        )
    )

    print()
    print("SAMPLE WEIGHTS")
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

    print(
        "Hard-negative boost:",
        HARD_NEGATIVE_BOOST,
    )

    pipeline = Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),

            (
                "pca",
                PCA(
                    n_components=(
                        PCA_COMPONENTS
                    ),
                    svd_solver="full",
                ),
            ),

            (
                "classifier",
                LogisticRegression(
                    C=(
                        LOGREG_C
                    ),
                    solver="liblinear",
                    class_weight=None,
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )

    print()
    print("TRAINING")
    print("-" * 72)

    pipeline.fit(
        X,
        y,
        classifier__sample_weight=(
            sample_weights
        ),
    )

    probability = (
        pipeline.predict_proba(
            X
        )[:, 1]
    )

    metrics = (
        metric_summary(
            y,
            probability,
        )
    )

    explained_variance = float(
        pipeline.named_steps[
            "pca"
        ]
        .explained_variance_ratio_
        .sum()
    )

    payload = {
        "pipeline": (
            pipeline
        ),

        "threshold": (
            THRESHOLD
        ),

        "feature_strategy": (
            "multi_mean"
        ),

        "x3d_windows": 3,

        "x3d_pooling": (
            "mean"
        ),

        "training_strategy": (
            "hard_negative_augmented"
        ),

        "hard_negative_boost": (
            HARD_NEGATIVE_BOOST
        ),
    }

    joblib.dump(
        payload,
        MODEL_FILE,
    )

    metadata = {
        "model_name": (
            "SecureVision Hard Negative Classifier"
        ),

        "feature_strategy": (
            "multi_mean"
        ),

        "pca_components": (
            PCA_COMPONENTS
        ),

        "logistic_regression_c": (
            LOGREG_C
        ),

        "threshold": (
            THRESHOLD
        ),

        "hard_negative_boost": (
            HARD_NEGATIVE_BOOST
        ),

        "original_clips": int(
            len(
                X_base
            )
        ),

        "hard_negative_clips": int(
            len(
                X_hard
            )
        ),

        "total_clips": int(
            len(
                X
            )
        ),

        "positive_clips": int(
            (
                y == 1
            ).sum()
        ),

        "negative_clips": int(
            (
                y == 0
            ).sum()
        ),

        "pca_explained_variance": (
            explained_variance
        ),

        "development_validation_used": (
            False
        ),

        "training_metrics": (
            metrics
        ),

        "model_sha256": (
            file_sha256(
                MODEL_FILE
            )
        ),
    }

    MODEL_JSON_FILE.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    TRAINING_REPORT_FILE.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("TRAINING METRICS")
    print("=" * 72)

    for key, value in (
        metrics.items()
    ):
        print(
            key,
            ":",
            value,
        )

    print()
    print(
        "PCA explained variance:",
        round(
            explained_variance,
            6,
        ),
    )

    print()
    print("FILES CREATED")
    print("-" * 72)

    print(
        MODEL_FILE
    )

    print(
        MODEL_JSON_FILE
    )

    print(
        TRAINING_REPORT_FILE
    )

    print()
    print(
        "✅ Hard-negative classifier trained."
    )

    print(
        "✅ Development validation was not used."
    )


if __name__ == "__main__":
    main()