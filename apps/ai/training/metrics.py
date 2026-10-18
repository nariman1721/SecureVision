from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    y_true,
    probabilities,
    threshold: float = 0.50,
) -> dict:
    y_true = np.asarray(
        y_true,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = (
        probabilities
        >= threshold
    ).astype(
        np.int64
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[
            0,
            1,
        ],
    )

    (
        true_negative,
        false_positive,
        false_negative,
        true_positive,
    ) = matrix.ravel()

    specificity_denominator = (
        true_negative
        + false_positive
    )

    specificity = (
        true_negative
        / specificity_denominator
        if specificity_denominator
        else 0.0
    )

    if (
        len(
            np.unique(
                y_true
            )
        )
        > 1
    ):
        roc_auc = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

        pr_auc = float(
            average_precision_score(
                y_true,
                probabilities,
            )
        )

    else:
        roc_auc = None
        pr_auc = None

    return {
        "threshold": float(
            threshold
        ),

        "samples": int(
            len(y_true)
        ),

        "actual_positive": int(
            (y_true == 1).sum()
        ),

        "actual_negative": int(
            (y_true == 0).sum()
        ),

        "predicted_positive": int(
            (predictions == 1).sum()
        ),

        "predicted_negative": int(
            (predictions == 0).sum()
        ),

        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),

        "precision_violation": float(
            precision_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),

        "recall_violation": float(
            recall_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),

        "f1_violation": float(
            f1_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),

        "f2_violation": float(
            fbeta_score(
                y_true,
                predictions,
                beta=2,
                pos_label=1,
                zero_division=0,
            )
        ),

        "specificity": float(
            specificity
        ),

        "roc_auc": (
            roc_auc
        ),

        "pr_auc": (
            pr_auc
        ),

        "confusion_matrix": {
            "tn": int(
                true_negative
            ),

            "fp": int(
                false_positive
            ),

            "fn": int(
                false_negative
            ),

            "tp": int(
                true_positive
            ),
        },
    }


def aggregate_video_predictions(
    y_true,
    probabilities,
    content_ids,
    video_ids,
    threshold: float = 0.50,
) -> pd.DataFrame:
    dataframe = pd.DataFrame(
        {
            "content_id": (
                np.asarray(
                    content_ids
                ).astype(str)
            ),

            "video_id": (
                np.asarray(
                    video_ids
                ).astype(str)
            ),

            "true_label": (
                np.asarray(
                    y_true,
                    dtype=np.int64,
                )
            ),

            "probability": (
                np.asarray(
                    probabilities,
                    dtype=np.float64,
                )
            ),
        }
    )

    videos = (
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

            true_label=(
                "true_label",
                "max",
            ),

            probability=(
                "probability",
                "max",
            ),

            clip_count=(
                "probability",
                "size",
            ),
        )
    )

    videos[
        "prediction"
    ] = (
        videos[
            "probability"
        ]
        >= threshold
    ).astype(
        np.int64
    )

    return videos


def build_threshold_sweep(
    y_true,
    probabilities,
    thresholds,
) -> pd.DataFrame:
    rows = []

    for threshold in thresholds:
        metrics = (
            compute_binary_metrics(
                y_true,
                probabilities,
                threshold=float(
                    threshold
                ),
            )
        )

        confusion = (
            metrics[
                "confusion_matrix"
            ]
        )

        rows.append(
            {
                "threshold": (
                    float(
                        threshold
                    )
                ),

                "accuracy": (
                    metrics[
                        "accuracy"
                    ]
                ),

                "balanced_accuracy": (
                    metrics[
                        "balanced_accuracy"
                    ]
                ),

                "precision": (
                    metrics[
                        "precision_violation"
                    ]
                ),

                "recall": (
                    metrics[
                        "recall_violation"
                    ]
                ),

                "f1": (
                    metrics[
                        "f1_violation"
                    ]
                ),

                "f2": (
                    metrics[
                        "f2_violation"
                    ]
                ),

                "specificity": (
                    metrics[
                        "specificity"
                    ]
                ),

                "tn": (
                    confusion["tn"]
                ),

                "fp": (
                    confusion["fp"]
                ),

                "fn": (
                    confusion["fn"]
                ),

                "tp": (
                    confusion["tp"]
                ),
            }
        )

    return pd.DataFrame(
        rows
    )