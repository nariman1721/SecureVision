from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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

N_SPLITS = 4

PCA_OPTIONS = [
    None,
    8,
    16,
    32,
]

C_VALUES = [
    0.001,
    0.01,
    0.03,
    0.1,
]

WEIGHT_STRATEGIES = {
    "none": {
        "class_weight": None,
        "video_sample_weight": False,
    },

    "balanced_only": {
        "class_weight": "balanced",
        "video_sample_weight": False,
    },

    "video_only": {
        "class_weight": None,
        "video_sample_weight": True,
    },

    "balanced_and_video": {
        "class_weight": "balanced",
        "video_sample_weight": True,
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

TRAIN_FEATURES_FILE = (
    FEATURES_DIR
    / "train_x3d_features.npz"
)

CV_RESULTS_FILE = (
    REPORTS_DIR
    / "cv_results.csv"
)

CV_FOLD_RESULTS_FILE = (
    REPORTS_DIR
    / "cv_fold_results.csv"
)

CV_FOLD_ASSIGNMENTS_FILE = (
    REPORTS_DIR
    / "cv_fold_assignments.csv"
)

BEST_CONFIG_FILE = (
    REPORTS_DIR
    / "cv_best_config.json"
)

BEST_OOF_VIDEO_FILE = (
    REPORTS_DIR
    / "cv_best_oof_video_predictions.csv"
)

BEST_THRESHOLD_SWEEP_FILE = (
    REPORTS_DIR
    / "cv_best_threshold_sweep.csv"
)


# =========================================================
# LOAD FEATURES
# =========================================================

def load_features() -> dict:
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
# VIDEO TABLE
# =========================================================

def build_video_table(
    y: np.ndarray,
    content_ids: np.ndarray,
    video_ids: np.ndarray,
) -> pd.DataFrame:
    dataframe = pd.DataFrame(
        {
            "content_id": (
                content_ids.astype(str)
            ),

            "video_id": (
                video_ids.astype(str)
            ),

            "clip_label": (
                y.astype(
                    np.int64
                )
            ),
        }
    )

    video_table = (
        dataframe
        .groupby(
            "content_id",
            as_index=False,
        )
        .agg(
            video_id=(
                "video_id",
                "first",
            ),

            video_label=(
                "clip_label",
                "max",
            ),

            clip_count=(
                "clip_label",
                "size",
            ),
        )
    )

    return video_table


# =========================================================
# FOLD ASSIGNMENTS
# =========================================================

def create_grouped_folds(
    video_table: pd.DataFrame,
) -> pd.DataFrame:
    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    assignments = (
        video_table.copy()
    )

    assignments[
        "fold"
    ] = -1

    X_groups = np.zeros(
        (
            len(video_table),
            1,
        )
    )

    y_groups = (
        video_table[
            "video_label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    for fold, (
        _,
        validation_indices,
    ) in enumerate(
        splitter.split(
            X_groups,
            y_groups,
        ),
        start=1,
    ):
        assignments.loc[
            validation_indices,
            "fold",
        ] = fold

    if (
        assignments[
            "fold"
        ]
        .eq(-1)
        .any()
    ):
        raise RuntimeError(
            "Some videos were not "
            "assigned to a fold."
        )

    return assignments


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
    pca_components: (
        int
        | None
    ),
    c_value: float,
    class_weight,
) -> Pipeline:
    steps = [
        (
            "scaler",
            StandardScaler(),
        ),
    ]

    if pca_components is not None:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=(
                        pca_components
                    ),
                    svd_solver="full",
                ),
            )
        )

    steps.append(
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
        )
    )

    return Pipeline(
        steps
    )


# =========================================================
# POSITIVE PROBABILITY
# =========================================================

def get_positive_probability(
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

    positive_index = int(
        np.where(
            classifier.classes_
            == 1
        )[0][0]
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
# BEST OOF THRESHOLD
# =========================================================

def select_oof_threshold(
    sweep: pd.DataFrame,
) -> pd.Series:
    # Missing a violation is expensive.
    # Prefer thresholds that achieve at
    # least 80% video recall.

    eligible = sweep[
        sweep[
            "recall"
        ]
        >= 0.80
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
                "f2",
                "specificity",
                "recall",
            ],
            ascending=[
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

    return eligible.iloc[0]


# =========================================================
# ONE CONFIGURATION
# =========================================================

def evaluate_configuration(
    X: np.ndarray,
    y: np.ndarray,
    content_ids: np.ndarray,
    video_ids: np.ndarray,
    assignments: pd.DataFrame,
    pca_components: (
        int
        | None
    ),
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

    oof_probabilities = np.full(
        len(y),
        np.nan,
        dtype=np.float64,
    )

    fold_rows = []

    for fold in range(
        1,
        N_SPLITS + 1,
    ):
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

        train_mask = np.array(
            [
                content_id
                not in validation_groups
                for content_id
                in content_ids
            ],
            dtype=bool,
        )

        validation_mask = (
            ~train_mask
        )

        train_group_ids = set(
            content_ids[
                train_mask
            ]
        )

        validation_group_ids = set(
            content_ids[
                validation_mask
            ]
        )

        overlap = (
            train_group_ids
            & validation_group_ids
        )

        if overlap:
            raise RuntimeError(
                "CV leakage detected."
            )

        X_train = X[
            train_mask
        ]

        y_train = y[
            train_mask
        ]

        X_validation = X[
            validation_mask
        ]

        y_validation = y[
            validation_mask
        ]

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
            weights = (
                build_video_balanced_weights(
                    content_ids[
                        train_mask
                    ]
                )
            )

            fit_kwargs[
                "classifier__sample_weight"
            ] = weights

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

        clip_metrics = (
            compute_binary_metrics(
                y_validation,
                probabilities,
                threshold=0.50,
            )
        )

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

                "fold": (
                    fold
                ),

                "pca_components": (
                    "none"
                    if pca_components
                    is None
                    else int(
                        pca_components
                    )
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
                        train_group_ids
                    )
                ),

                "validation_videos": int(
                    len(
                        validation_group_ids
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

                "video_f1": (
                    video_metrics[
                        "f1_violation"
                    ]
                ),

                "video_f2": (
                    video_metrics[
                        "f2_violation"
                    ]
                ),

                "clip_balanced_accuracy": (
                    clip_metrics[
                        "balanced_accuracy"
                    ]
                ),

                "clip_recall": (
                    clip_metrics[
                        "recall_violation"
                    ]
                ),
            }
        )

    if not np.isfinite(
        oof_probabilities
    ).all():
        raise RuntimeError(
            "Not every clip received "
            "an OOF prediction."
        )

    # -----------------------------------------------------
    # Entire OOF dataset
    # -----------------------------------------------------

    oof_clip_metrics = (
        compute_binary_metrics(
            y,
            oof_probabilities,
            threshold=0.50,
        )
    )

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

    oof_video_metrics = (
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

    # -----------------------------------------------------
    # OOF threshold diagnostics
    # -----------------------------------------------------

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

    best_threshold_row = (
        select_oof_threshold(
            threshold_sweep
        )
    )

    result = {
        "config_id": (
            config_id
        ),

        "pca_components": (
            "none"
            if pca_components is None
            else int(
                pca_components
            )
        ),

        "C": float(
            c_value
        ),

        "weight_strategy": (
            weight_strategy
        ),

        "class_weight": (
            "none"
            if class_weight is None
            else str(
                class_weight
            )
        ),

        "video_sample_weight": (
            use_video_weights
        ),

        # Primary threshold-free
        # model-selection metrics.
        "video_roc_auc": (
            oof_video_metrics[
                "roc_auc"
            ]
        ),

        "video_pr_auc": (
            oof_video_metrics[
                "pr_auc"
            ]
        ),

        # Metrics at 0.50.
        "video_balanced_accuracy_050": (
            oof_video_metrics[
                "balanced_accuracy"
            ]
        ),

        "video_precision_050": (
            oof_video_metrics[
                "precision_violation"
            ]
        ),

        "video_recall_050": (
            oof_video_metrics[
                "recall_violation"
            ]
        ),

        "video_specificity_050": (
            oof_video_metrics[
                "specificity"
            ]
        ),

        "video_f1_050": (
            oof_video_metrics[
                "f1_violation"
            ]
        ),

        "video_f2_050": (
            oof_video_metrics[
                "f2_violation"
            ]
        ),

        "clip_balanced_accuracy_050": (
            oof_clip_metrics[
                "balanced_accuracy"
            ]
        ),

        "clip_recall_050": (
            oof_clip_metrics[
                "recall_violation"
            ]
        ),

        # Diagnostic threshold based
        # only on train OOF predictions.
        "best_oof_threshold": float(
            best_threshold_row[
                "threshold"
            ]
        ),

        "best_oof_balanced_accuracy": float(
            best_threshold_row[
                "balanced_accuracy"
            ]
        ),

        "best_oof_precision": float(
            best_threshold_row[
                "precision"
            ]
        ),

        "best_oof_recall": float(
            best_threshold_row[
                "recall"
            ]
        ),

        "best_oof_specificity": float(
            best_threshold_row[
                "specificity"
            ]
        ),

        "best_oof_f1": float(
            best_threshold_row[
                "f1"
            ]
        ),

        "best_oof_f2": float(
            best_threshold_row[
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
        "SecureVision — Phase 7A "
        "Grouped Cross-Validation"
    )
    print("=" * 72)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_features()

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

    video_ids = (
        data[
            "video_ids"
        ]
        .astype(str)
    )

    print()
    print("TRAIN FEATURES")
    print("-" * 72)

    print(
        "X:",
        X.shape,
    )

    print(
        "y:",
        y.shape,
    )

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

    if not np.isfinite(
        X
    ).all():
        raise RuntimeError(
            "Features contain "
            "NaN or Inf."
        )

    # -----------------------------------------------------
    # Video groups
    # -----------------------------------------------------

    video_table = (
        build_video_table(
            y=y,
            content_ids=(
                content_ids
            ),
            video_ids=(
                video_ids
            ),
        )
    )

    print()
    print("VIDEO-LEVEL TRAIN SET")
    print("-" * 72)

    print(
        "Videos:",
        len(
            video_table
        ),
    )

    print(
        "Violation videos:",
        int(
            video_table[
                "video_label"
            ].eq(1).sum()
        ),
    )

    print(
        "Normal videos:",
        int(
            video_table[
                "video_label"
            ].eq(0).sum()
        ),
    )

    assignments = (
        create_grouped_folds(
            video_table
        )
    )

    assignments.to_csv(
        CV_FOLD_ASSIGNMENTS_FILE,
        index=False,
    )

    print()
    print("FOLD DISTRIBUTION")
    print("-" * 72)

    fold_distribution = (
        assignments
        .groupby(
            "fold"
        )
        .agg(
            videos=(
                "content_id",
                "count",
            ),

            violation=(
                "video_label",
                "sum",
            ),
        )
    )

    fold_distribution[
        "normal"
    ] = (
        fold_distribution[
            "videos"
        ]
        - fold_distribution[
            "violation"
        ]
    )

    print(
        fold_distribution.to_string()
    )

    # -----------------------------------------------------
    # Configurations
    # -----------------------------------------------------

    configurations = list(
        product(
            PCA_OPTIONS,
            C_VALUES,
            WEIGHT_STRATEGIES.keys(),
        )
    )

    print()
    print("SEARCH")
    print("-" * 72)

    print(
        "Configurations:",
        len(
            configurations
        ),
    )

    print(
        "Folds per configuration:",
        N_SPLITS,
    )

    print(
        "Total model fits:",
        len(
            configurations
        )
        * N_SPLITS,
    )

    all_results = []

    all_fold_results = []

    config_artifacts = {}

    for index, (
        pca_components,
        c_value,
        weight_strategy,
    ) in enumerate(
        tqdm(
            configurations,
            desc="CV configs",
        ),
        start=1,
    ):
        pca_name = (
            "none"
            if pca_components is None
            else str(
                pca_components
            )
        )

        config_id = (
            f"cfg_{index:03d}"
            f"_pca_{pca_name}"
            f"_c_{c_value}"
            f"_{weight_strategy}"
        )

        (
            result,
            fold_rows,
            oof_video_predictions,
            threshold_sweep,
        ) = evaluate_configuration(
            X=X,
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

        all_fold_results.extend(
            fold_rows
        )

        config_artifacts[
            config_id
        ] = {
            "video_predictions": (
                oof_video_predictions
            ),

            "threshold_sweep": (
                threshold_sweep
            ),
        }

    # -----------------------------------------------------
    # Leaderboard
    # -----------------------------------------------------

    results = pd.DataFrame(
        all_results
    )

    results = (
        results
        .sort_values(
            by=[
                "video_roc_auc",
                "video_pr_auc",
                "video_balanced_accuracy_050",
                "video_recall_050",
            ],
            ascending=[
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
    ] = (
        np.arange(
            1,
            len(results) + 1,
        )
    )

    results.to_csv(
        CV_RESULTS_FILE,
        index=False,
    )

    fold_results = (
        pd.DataFrame(
            all_fold_results
        )
    )

    fold_results.to_csv(
        CV_FOLD_RESULTS_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Best configuration
    # -----------------------------------------------------

    best = (
        results.iloc[0]
    )

    best_config_id = str(
        best[
            "config_id"
        ]
    )

    artifacts = (
        config_artifacts[
            best_config_id
        ]
    )

    best_video_predictions = (
        artifacts[
            "video_predictions"
        ]
    )

    best_threshold_sweep = (
        artifacts[
            "threshold_sweep"
        ]
    )

    best_video_predictions.to_csv(
        BEST_OOF_VIDEO_FILE,
        index=False,
    )

    best_threshold_sweep.to_csv(
        BEST_THRESHOLD_SWEEP_FILE,
        index=False,
    )

    best_config = {
        "selection_rule": (
            "Highest OOF video ROC-AUC; "
            "then PR-AUC; then balanced "
            "accuracy at threshold 0.50."
        ),

        "validation_holdout_used": (
            False
        ),

        "config_id": (
            best_config_id
        ),

        "pca_components": (
            None
            if str(
                best[
                    "pca_components"
                ]
            )
            == "none"
            else int(
                best[
                    "pca_components"
                ]
            )
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

        "video_balanced_accuracy_050": float(
            best[
                "video_balanced_accuracy_050"
            ]
        ),

        "video_recall_050": float(
            best[
                "video_recall_050"
            ]
        ),

        "video_specificity_050": float(
            best[
                "video_specificity_050"
            ]
        ),

        "best_oof_threshold": float(
            best[
                "best_oof_threshold"
            ]
        ),

        "best_oof_balanced_accuracy": float(
            best[
                "best_oof_balanced_accuracy"
            ]
        ),

        "best_oof_precision": float(
            best[
                "best_oof_precision"
            ]
        ),

        "best_oof_recall": float(
            best[
                "best_oof_recall"
            ]
        ),

        "best_oof_specificity": float(
            best[
                "best_oof_specificity"
            ]
        ),

        "best_oof_f1": float(
            best[
                "best_oof_f1"
            ]
        ),

        "best_oof_f2": float(
            best[
                "best_oof_f2"
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
    # Console report
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 72)

    columns = [
        "rank",
        "config_id",
        "pca_components",
        "C",
        "weight_strategy",
        "video_roc_auc",
        "video_pr_auc",
        "video_balanced_accuracy_050",
        "video_recall_050",
        "video_specificity_050",
        "best_oof_threshold",
        "best_oof_balanced_accuracy",
        "best_oof_recall",
        "best_oof_specificity",
    ]

    print(
        results[
            columns
        ]
        .head(10)
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("BEST CONFIGURATION")
    print("=" * 72)

    for key, value in (
        best_config.items()
    ):
        print(
            f"{key:32} {value}"
        )

    print()
    print("=" * 72)
    print("LEAKAGE CHECK")
    print("=" * 72)

    print(
        "CV grouped by content_id: True"
    )

    print(
        "Validation holdout used "
        "during search: False"
    )

    print()
    print(
        "✅ Grouped CV search complete."
    )

    print()
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    print(
        CV_RESULTS_FILE
    )

    print(
        CV_FOLD_RESULTS_FILE
    )

    print(
        CV_FOLD_ASSIGNMENTS_FILE
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