from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training.metrics import (
    aggregate_video_predictions,
    build_threshold_sweep,
    compute_binary_metrics,
)


# =========================================================
# CONFIG
# =========================================================

RANDOM_STATE = 42

PCA_OPTIONS = [
    8,
    16,
    32,
]

C_VALUES = [
    0.0003,
    0.001,
    0.003,
    0.01,
]

WEIGHT_STRATEGIES = {
    "none": {
        "class_weight": None,
        "video_sample_weight": False,
    },

    "video_only": {
        "class_weight": None,
        "video_sample_weight": True,
    },

    "balanced_only": {
        "class_weight": "balanced",
        "video_sample_weight": False,
    },
}

THRESHOLDS = np.round(
    np.arange(
        0.10,
        0.91,
        0.05,
    ),
    2,
)

MIN_RECALL = 0.80


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

REPORTS_DIR = (
    AI_ROOT
    / "reports"
)

SINGLE_FEATURES_FILE = (
    FEATURES_DIR
    / "train_x3d_features.npz"
)

MULTI_FEATURES_FILE = (
    FEATURES_DIR
    / "train_x3d_multiwindow_features.npz"
)

FOLD_ASSIGNMENTS_FILE = (
    REPORTS_DIR
    / "cv_fold_assignments.csv"
)

RESULTS_FILE = (
    REPORTS_DIR
    / "feature_strategy_cv_results.csv"
)

FOLD_RESULTS_FILE = (
    REPORTS_DIR
    / "feature_strategy_fold_results.csv"
)

BEST_CONFIG_FILE = (
    REPORTS_DIR
    / "feature_strategy_best_config.json"
)

BEST_OOF_VIDEO_FILE = (
    REPORTS_DIR
    / "feature_strategy_best_oof_video_predictions.csv"
)

BEST_THRESHOLD_SWEEP_FILE = (
    REPORTS_DIR
    / "feature_strategy_best_threshold_sweep.csv"
)


# =========================================================
# FEATURE LOADING
# =========================================================

def load_npz(
    path: Path,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing feature file:\n{path}"
        )

    return np.load(
        path,
        allow_pickle=False,
    )


def validate_feature_alignment(
    single,
    multi,
):
    required_common = [
        "y",
        "clip_ids",
        "content_ids",
        "video_ids",
        "clip_types",
    ]

    for key in required_common:
        if key not in single.files:
            raise ValueError(
                f"Single feature file "
                f"is missing: {key}"
            )

        if key not in multi.files:
            raise ValueError(
                f"Multi-window feature file "
                f"is missing: {key}"
            )

        left = single[key]
        right = multi[key]

        if left.shape != right.shape:
            raise RuntimeError(
                f"Feature alignment mismatch "
                f"for {key}: "
                f"{left.shape} != "
                f"{right.shape}"
            )

        if not np.array_equal(
            left,
            right,
        ):
            raise RuntimeError(
                f"Feature alignment mismatch "
                f"for {key}."
            )


def load_feature_sets():
    single = load_npz(
        SINGLE_FEATURES_FILE
    )

    multi = load_npz(
        MULTI_FEATURES_FILE
    )

    validate_feature_alignment(
        single,
        multi,
    )

    required_multi = [
        "X_mean",
        "X_max",
    ]

    for key in required_multi:
        if key not in multi.files:
            raise ValueError(
                f"Multi-window file "
                f"is missing: {key}"
            )

    feature_sets = {
        "single": (
            single[
                "X"
            ]
            .astype(
                np.float32
            )
        ),

        "multi_mean": (
            multi[
                "X_mean"
            ]
            .astype(
                np.float32
            )
        ),

        "multi_max": (
            multi[
                "X_max"
            ]
            .astype(
                np.float32
            )
        ),
    }

    metadata = {
        "y": (
            single[
                "y"
            ]
            .astype(
                np.int64
            )
        ),

        "clip_ids": (
            single[
                "clip_ids"
            ]
            .astype(str)
        ),

        "content_ids": (
            single[
                "content_ids"
            ]
            .astype(str)
        ),

        "video_ids": (
            single[
                "video_ids"
            ]
            .astype(str)
        ),

        "clip_types": (
            single[
                "clip_types"
            ]
            .astype(str)
        ),
    }

    for (
        name,
        features,
    ) in feature_sets.items():
        if features.shape != (
            len(
                metadata[
                    "y"
                ]
            ),
            2048,
        ):
            raise RuntimeError(
                f"Unexpected shape for "
                f"{name}: "
                f"{features.shape}"
            )

        if not np.isfinite(
            features
        ).all():
            raise RuntimeError(
                f"{name} contains "
                f"NaN or Inf."
            )

    return (
        feature_sets,
        metadata,
    )


# =========================================================
# FOLDS
# =========================================================

def load_fold_assignments():
    if not FOLD_ASSIGNMENTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing Phase 7A fold file:\n"
            f"{FOLD_ASSIGNMENTS_FILE}\n\n"
            "Run training.cross_validate first."
        )

    assignments = pd.read_csv(
        FOLD_ASSIGNMENTS_FILE
    )

    required = [
        "content_id",
        "fold",
    ]

    missing = [
        column
        for column in required
        if column not in assignments.columns
    ]

    if missing:
        raise ValueError(
            f"Fold assignments are "
            f"missing columns: {missing}"
        )

    if assignments[
        "content_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate content_id "
            "inside fold assignments."
        )

    return assignments


def validate_fold_coverage(
    assignments: pd.DataFrame,
    content_ids: np.ndarray,
):
    feature_groups = set(
        content_ids.astype(str)
    )

    assignment_groups = set(
        assignments[
            "content_id"
        ].astype(str)
    )

    if (
        feature_groups
        != assignment_groups
    ):
        raise RuntimeError(
            "Fold assignment content IDs "
            "do not match feature contents."
        )


# =========================================================
# SAMPLE WEIGHTS
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

    weights *= (
        len(weights)
        / weights.sum()
    )

    return weights


# =========================================================
# MODEL
# =========================================================

def build_pipeline(
    pca_components: int,
    c_value: float,
    class_weight,
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
                    class_weight=(
                        class_weight
                    ),
                    max_iter=5000,
                    random_state=(
                        RANDOM_STATE
                    ),
                ),
            ),
        ]
    )


def get_positive_probability(
    pipeline: Pipeline,
    features: np.ndarray,
) -> np.ndarray:
    probabilities = (
        pipeline.predict_proba(
            features
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
            "Positive class not found."
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
# THRESHOLD SELECTION
# =========================================================

def select_operating_threshold(
    sweep: pd.DataFrame,
) -> pd.Series:
    """
    SecureVision should prioritize not missing
    violations, but among high-recall thresholds
    prefer fewer false positives.
    """

    eligible = sweep[
        sweep[
            "recall"
        ]
        >= MIN_RECALL
    ].copy()

    if eligible.empty:
        eligible = (
            sweep.copy()
        )

    eligible = (
        eligible
        .sort_values(
            by=[
                "balanced_accuracy",
                "specificity",
                "f2",
                "precision",
                "recall",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return (
        eligible.iloc[0]
    )


# =========================================================
# ONE CONFIGURATION
# =========================================================

def evaluate_configuration(
    features: np.ndarray,
    y: np.ndarray,
    content_ids: np.ndarray,
    video_ids: np.ndarray,
    assignments: pd.DataFrame,
    feature_strategy: str,
    pca_components: int,
    c_value: float,
    weight_strategy: str,
    config_id: str,
):
    strategy = (
        WEIGHT_STRATEGIES[
            weight_strategy
        ]
    )

    class_weight = (
        strategy[
            "class_weight"
        ]
    )

    use_video_weights = bool(
        strategy[
            "video_sample_weight"
        ]
    )

    folds = sorted(
        assignments[
            "fold"
        ]
        .unique()
        .tolist()
    )

    oof_probabilities = np.full(
        len(y),
        np.nan,
        dtype=np.float64,
    )

    fold_rows = []

    for fold in folds:
        validation_groups = set(
            assignments.loc[
                assignments[
                    "fold"
                ].eq(fold),
                "content_id",
            ]
            .astype(str)
            .tolist()
        )

        validation_mask = np.array(
            [
                content_id
                in validation_groups
                for content_id
                in content_ids
            ],
            dtype=bool,
        )

        train_mask = (
            ~validation_mask
        )

        train_groups = set(
            content_ids[
                train_mask
            ]
        )

        validation_groups_actual = set(
            content_ids[
                validation_mask
            ]
        )

        overlap = (
            train_groups
            & validation_groups_actual
        )

        if overlap:
            raise RuntimeError(
                "Grouped CV leakage detected."
            )

        X_train = (
            features[
                train_mask
            ]
        )

        y_train = (
            y[
                train_mask
            ]
        )

        X_validation = (
            features[
                validation_mask
            ]
        )

        y_validation = (
            y[
                validation_mask
            ]
        )

        pipeline = build_pipeline(
            pca_components=(
                pca_components
            ),
            c_value=(
                c_value
            ),
            class_weight=(
                class_weight
            ),
        )

        fit_kwargs = {}

        if use_video_weights:
            sample_weights = (
                build_video_balanced_weights(
                    content_ids[
                        train_mask
                    ]
                )
            )

            fit_kwargs[
                "classifier__sample_weight"
            ] = sample_weights

        pipeline.fit(
            X_train,
            y_train,
            **fit_kwargs,
        )

        probabilities = (
            get_positive_probability(
                pipeline,
                X_validation,
            )
        )

        oof_probabilities[
            validation_mask
        ] = probabilities

        video_predictions = (
            aggregate_video_predictions(
                y_true=(
                    y_validation
                ),
                probabilities=(
                    probabilities
                ),
                content_ids=(
                    content_ids[
                        validation_mask
                    ]
                ),
                video_ids=(
                    video_ids[
                        validation_mask
                    ]
                ),
                threshold=0.50,
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
                threshold=0.50,
            )
        )

        fold_rows.append(
            {
                "config_id": (
                    config_id
                ),

                "feature_strategy": (
                    feature_strategy
                ),

                "fold": int(
                    fold
                ),

                "pca_components": int(
                    pca_components
                ),

                "C": float(
                    c_value
                ),

                "weight_strategy": (
                    weight_strategy
                ),

                "train_clips": int(
                    train_mask.sum()
                ),

                "validation_clips": int(
                    validation_mask.sum()
                ),

                "train_videos": int(
                    len(
                        train_groups
                    )
                ),

                "validation_videos": int(
                    len(
                        validation_groups_actual
                    )
                ),

                "video_roc_auc": (
                    video_metrics[
                        "roc_auc"
                    ]
                ),

                "video_pr_auc": (
                    video_metrics[
                        "pr_auc"
                    ]
                ),

                "video_balanced_accuracy": (
                    video_metrics[
                        "balanced_accuracy"
                    ]
                ),

                "video_recall": (
                    video_metrics[
                        "recall_violation"
                    ]
                ),

                "video_specificity": (
                    video_metrics[
                        "specificity"
                    ]
                ),
            }
        )

    if not np.isfinite(
        oof_probabilities
    ).all():
        raise RuntimeError(
            "Not every clip received "
            "an OOF probability."
        )

    # -----------------------------------------------------
    # Aggregate all OOF predictions
    # -----------------------------------------------------

    oof_video_predictions = (
        aggregate_video_predictions(
            y_true=y,
            probabilities=(
                oof_probabilities
            ),
            content_ids=(
                content_ids
            ),
            video_ids=(
                video_ids
            ),
            threshold=0.50,
        )
    )

    video_metrics_050 = (
        compute_binary_metrics(
            oof_video_predictions[
                "true_label"
            ].to_numpy(),
            oof_video_predictions[
                "probability"
            ].to_numpy(),
            threshold=0.50,
        )
    )

    threshold_sweep = (
        build_threshold_sweep(
            y_true=(
                oof_video_predictions[
                    "true_label"
                ].to_numpy()
            ),
            probabilities=(
                oof_video_predictions[
                    "probability"
                ].to_numpy()
            ),
            thresholds=(
                THRESHOLDS
            ),
        )
    )

    best_threshold = (
        select_operating_threshold(
            threshold_sweep
        )
    )

    result = {
        "config_id": (
            config_id
        ),

        "feature_strategy": (
            feature_strategy
        ),

        "pca_components": int(
            pca_components
        ),

        "C": float(
            c_value
        ),

        "weight_strategy": (
            weight_strategy
        ),

        "video_roc_auc": (
            video_metrics_050[
                "roc_auc"
            ]
        ),

        "video_pr_auc": (
            video_metrics_050[
                "pr_auc"
            ]
        ),

        "balanced_accuracy_050": (
            video_metrics_050[
                "balanced_accuracy"
            ]
        ),

        "recall_050": (
            video_metrics_050[
                "recall_violation"
            ]
        ),

        "specificity_050": (
            video_metrics_050[
                "specificity"
            ]
        ),

        "best_threshold": float(
            best_threshold[
                "threshold"
            ]
        ),

        "best_balanced_accuracy": float(
            best_threshold[
                "balanced_accuracy"
            ]
        ),

        "best_precision": float(
            best_threshold[
                "precision"
            ]
        ),

        "best_recall": float(
            best_threshold[
                "recall"
            ]
        ),

        "best_specificity": float(
            best_threshold[
                "specificity"
            ]
        ),

        "best_f1": float(
            best_threshold[
                "f1"
            ]
        ),

        "best_f2": float(
            best_threshold[
                "f2"
            ]
        ),
    }

    return (
        result,
        fold_rows,
        oof_video_predictions,
        threshold_sweep,
    )


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 7C "
        "Feature Strategy Comparison"
    )
    print("=" * 72)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        feature_sets,
        metadata,
    ) = load_feature_sets()

    y = (
        metadata[
            "y"
        ]
    )

    content_ids = (
        metadata[
            "content_ids"
        ]
    )

    video_ids = (
        metadata[
            "video_ids"
        ]
    )

    assignments = (
        load_fold_assignments()
    )

    validate_fold_coverage(
        assignments,
        content_ids,
    )

    print()
    print("DATASET")
    print("-" * 72)

    print(
        "Clips:",
        len(y),
    )

    print(
        "Unique videos:",
        len(
            np.unique(
                content_ids
            )
        ),
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

    print()
    print("FEATURE SETS")
    print("-" * 72)

    for (
        name,
        features,
    ) in feature_sets.items():
        print(
            f"{name:15}",
            features.shape,
        )

    configurations = list(
        product(
            feature_sets.keys(),
            PCA_OPTIONS,
            C_VALUES,
            WEIGHT_STRATEGIES.keys(),
        )
    )

    print()
    print("SEARCH")
    print("-" * 72)

    print(
        "Feature strategies:",
        len(
            feature_sets
        ),
    )

    print(
        "PCA options:",
        len(
            PCA_OPTIONS
        ),
    )

    print(
        "C values:",
        len(
            C_VALUES
        ),
    )

    print(
        "Weight strategies:",
        len(
            WEIGHT_STRATEGIES
        ),
    )

    print(
        "Total configurations:",
        len(
            configurations
        ),
    )

    print(
        "Total fits:",
        len(
            configurations
        )
        * assignments[
            "fold"
        ].nunique(),
    )

    all_results = []

    all_fold_rows = []

    artifacts = {}

    for index, (
        feature_strategy,
        pca_components,
        c_value,
        weight_strategy,
    ) in enumerate(
        tqdm(
            configurations,
            desc="Feature CV",
        ),
        start=1,
    ):
        config_id = (
            f"fs_{index:03d}_"
            f"{feature_strategy}_"
            f"pca{pca_components}_"
            f"c{c_value}_"
            f"{weight_strategy}"
        )

        (
            result,
            fold_rows,
            oof_video_predictions,
            threshold_sweep,
        ) = evaluate_configuration(
            features=(
                feature_sets[
                    feature_strategy
                ]
            ),
            y=y,
            content_ids=(
                content_ids
            ),
            video_ids=(
                video_ids
            ),
            assignments=(
                assignments
            ),
            feature_strategy=(
                feature_strategy
            ),
            pca_components=(
                pca_components
            ),
            c_value=(
                c_value
            ),
            weight_strategy=(
                weight_strategy
            ),
            config_id=(
                config_id
            ),
        )

        all_results.append(
            result
        )

        all_fold_rows.extend(
            fold_rows
        )

        artifacts[
            config_id
        ] = {
            "oof_video_predictions": (
                oof_video_predictions
            ),

            "threshold_sweep": (
                threshold_sweep
            ),
        }

    # -----------------------------------------------------
    # Ranking
    # -----------------------------------------------------

    results = pd.DataFrame(
        all_results
    )

    results = (
        results
        .sort_values(
            by=[
                "best_balanced_accuracy",
                "best_specificity",
                "video_roc_auc",
                "video_pr_auc",
                "best_recall",
            ],
            ascending=[
                False,
                False,
                False,
                False,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    results[
        "rank"
    ] = np.arange(
        1,
        len(results) + 1,
    )

    results.to_csv(
        RESULTS_FILE,
        index=False,
    )

    pd.DataFrame(
        all_fold_rows
    ).to_csv(
        FOLD_RESULTS_FILE,
        index=False,
    )

    best = (
        results.iloc[0]
    )

    best_id = str(
        best[
            "config_id"
        ]
    )

    best_artifacts = (
        artifacts[
            best_id
        ]
    )

    best_artifacts[
        "oof_video_predictions"
    ].to_csv(
        BEST_OOF_VIDEO_FILE,
        index=False,
    )

    best_artifacts[
        "threshold_sweep"
    ].to_csv(
        BEST_THRESHOLD_SWEEP_FILE,
        index=False,
    )

    best_config = {
        "selection_rule": (
            "Require OOF video recall >= 0.80 "
            "when possible; rank by best "
            "balanced accuracy, specificity, "
            "ROC-AUC, PR-AUC, then recall."
        ),

        "holdout_validation_used": (
            False
        ),

        "config_id": (
            best_id
        ),

        "feature_strategy": str(
            best[
                "feature_strategy"
            ]
        ),

        "pca_components": int(
            best[
                "pca_components"
            ]
        ),

        "C": float(
            best[
                "C"
            ]
        ),

        "weight_strategy": str(
            best[
                "weight_strategy"
            ]
        ),

        "video_roc_auc": float(
            best[
                "video_roc_auc"
            ]
        ),

        "video_pr_auc": float(
            best[
                "video_pr_auc"
            ]
        ),

        "threshold": float(
            best[
                "best_threshold"
            ]
        ),

        "balanced_accuracy": float(
            best[
                "best_balanced_accuracy"
            ]
        ),

        "precision": float(
            best[
                "best_precision"
            ]
        ),

        "recall": float(
            best[
                "best_recall"
            ]
        ),

        "specificity": float(
            best[
                "best_specificity"
            ]
        ),

        "f1": float(
            best[
                "best_f1"
            ]
        ),

        "f2": float(
            best[
                "best_f2"
            ]
        ),
    }

    BEST_CONFIG_FILE.write_text(
        json.dumps(
            best_config,
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
    print("TOP 15 CONFIGURATIONS")
    print("=" * 72)

    columns = [
        "rank",
        "feature_strategy",
        "pca_components",
        "C",
        "weight_strategy",
        "video_roc_auc",
        "video_pr_auc",
        "best_threshold",
        "best_balanced_accuracy",
        "best_precision",
        "best_recall",
        "best_specificity",
        "best_f1",
        "best_f2",
    ]

    print(
        results[
            columns
        ]
        .head(15)
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("BEST CONFIGURATION")
    print("=" * 72)

    for (
        key,
        value,
    ) in (
        best_config.items()
    ):
        print(
            f"{key:30} {value}"
        )

    print()
    print("=" * 72)
    print("BEST RESULT BY FEATURE STRATEGY")
    print("=" * 72)

    per_strategy = (
        results
        .sort_values(
            "rank"
        )
        .groupby(
            "feature_strategy",
            as_index=False,
        )
        .first()
    )

    print(
        per_strategy[
            [
                "feature_strategy",
                "pca_components",
                "C",
                "weight_strategy",
                "video_roc_auc",
                "best_balanced_accuracy",
                "best_recall",
                "best_specificity",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("LEAKAGE CHECK")
    print("=" * 72)

    print(
        "Same grouped folds reused: True"
    )

    print(
        "Grouped by content_id: True"
    )

    print(
        "Holdout validation used: False"
    )

    print()
    print(
        "✅ Feature strategy comparison "
        "complete."
    )

    print()
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    print(
        RESULTS_FILE
    )

    print(
        FOLD_RESULTS_FILE
    )

    print(
        BEST_CONFIG_FILE
    )

    print(
        BEST_OOF_VIDEO_FILE
    )

    print(
        BEST_THRESHOLD_SWEEP_FILE
    )


if __name__ == "__main__":
    main()