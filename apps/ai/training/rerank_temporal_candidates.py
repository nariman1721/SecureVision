from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.video.candidate_reranker import (
    FEATURE_COLUMNS,
    assign_candidate_labels,
    build_candidate_table,
)

from training.diagnose_temporal_signal import (
    build_positive_cases,
    load_train_manifest,
)


AI_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

REPORTS_DIR = (
    AI_ROOT
    / "reports"
)

MODELS_DIR = (
    AI_ROOT
    / "models"
)

WINDOWS_FILE = (
    REPORTS_DIR
    / "temporal_signal_windows.csv"
)

OOF_FILE = (
    REPORTS_DIR
    / "candidate_reranker_oof.csv"
)

CANDIDATES_FILE = (
    REPORTS_DIR
    / "candidate_reranker_candidates.csv"
)

SUMMARY_FILE = (
    REPORTS_DIR
    / "candidate_reranker_summary.json"
)

MODEL_FILE = (
    MODELS_DIR
    / "securevision_candidate_reranker.joblib"
)

MODEL_METADATA_FILE = (
    MODELS_DIR
    / "securevision_candidate_reranker.json"
)


POSITIVE_RADIUS_SEC = 5.0


# =========================================================
# MODEL
# =========================================================

def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=0.1,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=42,
                ),
            ),
        ]
    )


def positive_probability(
    model: Pipeline,
    X: np.ndarray,
) -> np.ndarray:
    probabilities = (
        model.predict_proba(
            X
        )
    )

    classifier = (
        model.named_steps[
            "classifier"
        ]
    )

    positive_index = int(
        np.where(
            classifier.classes_
            == 1
        )[0][0]
    )

    return probabilities[
        :,
        positive_index,
    ]


# =========================================================
# SUMMARY
# =========================================================

def summarize_errors(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "events": int(
            len(values)
        ),
        "mean_error_sec": float(
            values.mean()
        ),
        "median_error_sec": float(
            np.median(
                values
            )
        ),
        "within_3_sec": int(
            (
                values
                <= 3.0
            ).sum()
        ),
        "within_5_sec": int(
            (
                values
                <= 5.0
            ).sum()
        ),
        "within_10_sec": int(
            (
                values
                <= 10.0
            ).sum()
        ),
    }


# =========================================================
# LOAD
# =========================================================

def load_windows():
    if not WINDOWS_FILE.exists():
        raise FileNotFoundError(
            f"Missing Phase 8B file:\n"
            f"{WINDOWS_FILE}"
        )

    dataframe = pd.read_csv(
        WINDOWS_FILE
    )

    required = [
        "content_id",
        "video_id",
        "window_index",
        "start_sec",
        "end_sec",
        "center_sec",
        "raw_probability",
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    return dataframe


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 8D "
        "Candidate Re-ranking"
    )
    print("=" * 72)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    windows = (
        load_windows()
    )

    manifest = (
        load_train_manifest()
    )

    cases = (
        build_positive_cases(
            manifest
        )
    )

    case_by_video = {
        case[
            "video_id"
        ]: case
        for case in cases
    }

    print()
    print("INPUT")
    print("-" * 72)

    print(
        "Videos:",
        len(
            cases
        ),
    )

    print(
        "Events:",
        sum(
            len(
                case[
                    "event_times"
                ]
            )
            for case in cases
        ),
    )

    print(
        "Cached scan windows:",
        len(
            windows
        ),
    )

    print(
        "Running X3D again:",
        False,
    )

    # =====================================================
    # BUILD CANDIDATE DATASET
    # =====================================================

    all_candidates = []

    for video_id, timeline in (
        windows.groupby(
            "video_id"
        )
    ):
        video_id = str(
            video_id
        )

        if video_id not in case_by_video:
            continue

        case = (
            case_by_video[
                video_id
            ]
        )

        candidates = (
            build_candidate_table(
                timeline,
                scan_stride_sec=2.0,
                support_radius=1,
            )
        )

        candidates = (
            assign_candidate_labels(
                candidates,
                event_times=[
                    float(value)
                    for value
                    in case[
                        "event_times"
                    ]
                ],
                positive_radius_sec=(
                    POSITIVE_RADIUS_SEC
                ),
            )
        )

        candidates.insert(
            0,
            "video_id",
            video_id,
        )

        candidates.insert(
            1,
            "content_id",
            case[
                "content_id"
            ],
        )

        all_candidates.append(
            candidates
        )

    candidate_data = pd.concat(
        all_candidates,
        ignore_index=True,
    )

    candidate_data.to_csv(
        CANDIDATES_FILE,
        index=False,
    )

    print()
    print("CANDIDATE DATASET")
    print("-" * 72)

    print(
        "Candidates:",
        len(
            candidate_data
        ),
    )

    print(
        "Positive candidates:",
        int(
            candidate_data[
                "reranker_label"
            ].sum()
        ),
    )

    print(
        "Negative candidates:",
        int(
            (
                candidate_data[
                    "reranker_label"
                ]
                == 0
            ).sum()
        ),
    )

    print(
        "Features:",
        len(
            FEATURE_COLUMNS
        ),
    )

    # =====================================================
    # LEAVE-ONE-VIDEO-OUT
    # =====================================================

    oof_candidate_rows = []

    video_ids = sorted(
        candidate_data[
            "video_id"
        ]
        .unique()
        .tolist()
    )

    print()
    print("LEAVE-ONE-VIDEO-OUT CV")
    print("-" * 72)

    for fold_index, holdout_video in enumerate(
        video_ids,
        start=1,
    ):
        train = (
            candidate_data[
                candidate_data[
                    "video_id"
                ]
                != holdout_video
            ]
            .copy()
        )

        test = (
            candidate_data[
                candidate_data[
                    "video_id"
                ]
                == holdout_video
            ]
            .copy()
        )

        if (
            train[
                "reranker_label"
            ].nunique()
            < 2
        ):
            raise RuntimeError(
                "Training fold has only one class."
            )

        model = (
            build_model()
        )

        X_train = (
            train[
                FEATURE_COLUMNS
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        y_train = (
            train[
                "reranker_label"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        X_test = (
            test[
                FEATURE_COLUMNS
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        model.fit(
            X_train,
            y_train,
        )

        test[
            "reranker_probability"
        ] = (
            positive_probability(
                model,
                X_test,
            )
        )

        test = (
            test
            .sort_values(
                "reranker_probability",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        test[
            "reranked_rank"
        ] = np.arange(
            1,
            len(
                test
            )
            + 1,
        )

        print(
            f"[{fold_index:02d}/{len(video_ids)}] "
            f"{holdout_video}: "
            f"{len(test)} candidates"
        )

        oof_candidate_rows.append(
            test
        )

    oof_candidates = pd.concat(
        oof_candidate_rows,
        ignore_index=True,
    )

    # =====================================================
    # EVENT-LEVEL EVALUATION
    # =====================================================

    event_rows = []

    for case in cases:
        video_id = (
            case[
                "video_id"
            ]
        )

        predictions = (
            oof_candidates[
                oof_candidates[
                    "video_id"
                ]
                == video_id
            ]
            .sort_values(
                "reranked_rank"
            )
            .reset_index(
                drop=True
            )
        )

        original = (
            candidate_data[
                candidate_data[
                    "video_id"
                ]
                == video_id
            ]
            .sort_values(
                "original_rank"
            )
            .reset_index(
                drop=True
            )
        )

        for event_index, event_time in enumerate(
            case[
                "event_times"
            ],
            start=1,
        ):
            event_time = float(
                event_time
            )

            original_top1 = float(
                original.iloc[0][
                    "timecode_sec"
                ]
            )

            reranked_top1 = float(
                predictions.iloc[0][
                    "timecode_sec"
                ]
            )

            original_error = abs(
                original_top1
                - event_time
            )

            reranked_error = abs(
                reranked_top1
                - event_time
            )

            reranked_top3 = (
                predictions
                .head(3)
            )

            nearest_top3_row = (
                reranked_top3.iloc[
                    (
                        reranked_top3[
                            "timecode_sec"
                        ]
                        - event_time
                    )
                    .abs()
                    .argmin()
                ]
            )

            reranked_top3_error = abs(
                float(
                    nearest_top3_row[
                        "timecode_sec"
                    ]
                )
                - event_time
            )

            event_rows.append(
                {
                    "video_id": (
                        video_id
                    ),
                    "event_index": int(
                        event_index
                    ),
                    "event_time_sec": (
                        event_time
                    ),
                    "original_top1_sec": (
                        original_top1
                    ),
                    "original_error_sec": (
                        original_error
                    ),
                    "reranked_top1_sec": (
                        reranked_top1
                    ),
                    "reranked_top1_error_sec": (
                        reranked_error
                    ),
                    "nearest_reranked_top3_sec": float(
                        nearest_top3_row[
                            "timecode_sec"
                        ]
                    ),
                    "nearest_reranked_top3_rank": int(
                        nearest_top3_row[
                            "reranked_rank"
                        ]
                    ),
                    "reranked_top3_error_sec": (
                        reranked_top3_error
                    ),
                }
            )

    events = pd.DataFrame(
        event_rows
    )

    # Add event rows into OOF report file.
    oof_candidates.to_csv(
        OOF_FILE,
        index=False,
    )

    original_summary = (
        summarize_errors(
            events[
                "original_error_sec"
            ]
        )
    )

    reranked_top1_summary = (
        summarize_errors(
            events[
                "reranked_top1_error_sec"
            ]
        )
    )

    reranked_top3_summary = (
        summarize_errors(
            events[
                "reranked_top3_error_sec"
            ]
        )
    )

    better = int(
        (
            events[
                "reranked_top1_error_sec"
            ]
            <
            events[
                "original_error_sec"
            ]
        ).sum()
    )

    equal = int(
        np.isclose(
            events[
                "reranked_top1_error_sec"
            ],
            events[
                "original_error_sec"
            ],
        ).sum()
    )

    worse = int(
        (
            events[
                "reranked_top1_error_sec"
            ]
            >
            events[
                "original_error_sec"
            ]
        ).sum()
    )

    # =====================================================
    # TRAIN FINAL RERANKER ON ALL TRAIN CANDIDATES
    # =====================================================

    final_model = (
        build_model()
    )

    final_X = (
        candidate_data[
            FEATURE_COLUMNS
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    final_y = (
        candidate_data[
            "reranker_label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    final_model.fit(
        final_X,
        final_y,
    )

    payload = {
        "pipeline": (
            final_model
        ),
        "feature_columns": (
            FEATURE_COLUMNS
        ),
        "positive_radius_sec": (
            POSITIVE_RADIUS_SEC
        ),
        "training_candidates": int(
            len(
                candidate_data
            )
        ),
        "training_videos": int(
            candidate_data[
                "video_id"
            ].nunique()
        ),
    }

    joblib.dump(
        payload,
        MODEL_FILE,
    )

    metadata = {
        "model_name": (
            "SecureVision Temporal Candidate Reranker"
        ),
        "model_type": (
            "StandardScaler + LogisticRegression"
        ),
        "validation_strategy": (
            "leave-one-video-out"
        ),
        "development_validation_used": (
            False
        ),
        "x3d_recomputed": (
            False
        ),
        "feature_columns": (
            FEATURE_COLUMNS
        ),
        "positive_radius_sec": (
            POSITIVE_RADIUS_SEC
        ),
        "training_candidates": int(
            len(
                candidate_data
            )
        ),
        "training_positive_candidates": int(
            final_y.sum()
        ),
        "training_negative_candidates": int(
            (
                final_y
                == 0
            ).sum()
        ),
        "training_videos": int(
            candidate_data[
                "video_id"
            ].nunique()
        ),
    }

    MODEL_METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report = {
        "configuration": {
            "development_validation_used": (
                False
            ),
            "x3d_recomputed": (
                False
            ),
            "validation": (
                "leave-one-video-out"
            ),
            "positive_radius_sec": (
                POSITIVE_RADIUS_SEC
            ),
            "features": (
                FEATURE_COLUMNS
            ),
        },
        "dataset": {
            "videos": int(
                candidate_data[
                    "video_id"
                ].nunique()
            ),
            "candidates": int(
                len(
                    candidate_data
                )
            ),
            "positive_candidates": int(
                final_y.sum()
            ),
            "negative_candidates": int(
                (
                    final_y
                    == 0
                ).sum()
            ),
        },
        "original_top1": (
            original_summary
        ),
        "reranked_top1": (
            reranked_top1_summary
        ),
        "reranked_top3": (
            reranked_top3_summary
        ),
        "top1_vs_original": {
            "better": (
                better
            ),
            "equal": (
                equal
            ),
            "worse": (
                worse
            ),
        },
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # =====================================================
    # PRINT
    # =====================================================

    print()
    print("=" * 72)
    print("ORIGINAL 8C TOP-1")
    print("=" * 72)

    for key, value in (
        original_summary.items()
    ):
        print(
            f"{key:22}",
            (
                round(
                    value,
                    4,
                )
                if isinstance(
                    value,
                    float,
                )
                else value
            ),
        )

    print()
    print("=" * 72)
    print("RERANKED OOF TOP-1")
    print("=" * 72)

    for key, value in (
        reranked_top1_summary.items()
    ):
        print(
            f"{key:22}",
            (
                round(
                    value,
                    4,
                )
                if isinstance(
                    value,
                    float,
                )
                else value
            ),
        )

    print()
    print("=" * 72)
    print("RERANKED OOF TOP-3")
    print("=" * 72)

    for key, value in (
        reranked_top3_summary.items()
    ):
        print(
            f"{key:22}",
            (
                round(
                    value,
                    4,
                )
                if isinstance(
                    value,
                    float,
                )
                else value
            ),
        )

    print()
    print("TOP-1 VS ORIGINAL")
    print("-" * 72)

    print(
        "Better:",
        better,
    )

    print(
        "Equal:",
        equal,
    )

    print(
        "Worse:",
        worse,
    )

    print()
    print("=" * 72)
    print("EVENT TABLE")
    print("=" * 72)

    print(
        events
        .round(3)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    print(
        CANDIDATES_FILE
    )

    print(
        OOF_FILE
    )

    print(
        SUMMARY_FILE
    )

    print(
        MODEL_FILE
    )

    print(
        MODEL_METADATA_FILE
    )

    print()
    print(
        "✅ Candidate reranking complete."
    )

    print(
        "✅ Leave-one-video-out evaluation complete."
    )

    print(
        "✅ X3D was NOT run again."
    )


if __name__ == "__main__":
    main()