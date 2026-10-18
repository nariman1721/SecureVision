from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

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

POSITIVE_CANDIDATES_FILE = (
    REPORTS_DIR
    / "candidate_reranker_candidates.csv"
)

NEGATIVE_CANDIDATES_FILE = (
    REPORTS_DIR
    / "negative_candidates.csv"
)

DISTRIBUTIONS_FILE = (
    REPORTS_DIR
    / "candidate_feature_distributions.csv"
)

SINGLE_SEARCH_FILE = (
    REPORTS_DIR
    / "candidate_threshold_search.csv"
)

PAIR_SEARCH_FILE = (
    REPORTS_DIR
    / "candidate_pair_threshold_search.csv"
)

SUMMARY_FILE = (
    REPORTS_DIR
    / "candidate_calibration_summary.json"
)

RULE_FILE = (
    MODELS_DIR
    / "securevision_candidate_acceptance_rule.json"
)


TARGET_EVENT_RECALL = 0.90

# Candidate generation already achieved
# 18/18 coverage within ±10 sec on train.
# This radius is for verdict calibration,
# not final timestamp accuracy.
EVENT_MATCH_RADIUS_SEC = 10.0


SINGLE_METRICS = [
    "region_score",
    "max_candidate_score",
    "mean_candidate_score",
    "peak_probability",
    "peak_prominence",
    "mean_prominence",
    "excess_area",
    "normalized_area",
    "consistency",
    "persistence_mean",
]


PAIR_METRICS = [
    (
        "region_score",
        "peak_probability",
    ),
    (
        "region_score",
        "peak_prominence",
    ),
    (
        "region_score",
        "normalized_area",
    ),
    (
        "peak_probability",
        "peak_prominence",
    ),
]


# =========================================================
# LOAD
# =========================================================

def load_candidates():
    if not POSITIVE_CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n"
            f"{POSITIVE_CANDIDATES_FILE}"
        )

    if not NEGATIVE_CANDIDATES_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n"
            f"{NEGATIVE_CANDIDATES_FILE}"
        )

    positive = pd.read_csv(
        POSITIVE_CANDIDATES_FILE
    )

    negative = pd.read_csv(
        NEGATIVE_CANDIDATES_FILE
    )

    required = (
        set(
            SINGLE_METRICS
        )
        | {
            "video_id",
            "timecode_sec",
        }
    )

    for name, dataframe in [
        (
            "positive",
            positive,
        ),
        (
            "negative",
            negative,
        ),
    ]:
        missing = [
            column
            for column in required
            if column not in dataframe.columns
        ]

        if missing:
            raise RuntimeError(
                f"{name} candidates missing "
                f"columns: {missing}"
            )

    return (
        positive,
        negative,
    )


# =========================================================
# EVENT TABLE
# =========================================================

def build_event_table():
    manifest = (
        load_train_manifest()
    )

    cases = (
        build_positive_cases(
            manifest
        )
    )

    rows = []

    for case in cases:
        for event_index, event_time in enumerate(
            case[
                "event_times"
            ],
            start=1,
        ):
            rows.append(
                {
                    "video_id": str(
                        case[
                            "video_id"
                        ]
                    ),

                    "event_index": int(
                        event_index
                    ),

                    "event_time_sec": float(
                        event_time
                    ),
                }
            )

    return (
        pd.DataFrame(
            rows
        ),
        cases,
    )


# =========================================================
# THRESHOLDS
# =========================================================

def threshold_grid(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = np.unique(
        values[
            np.isfinite(
                values
            )
        ]
    )

    values.sort()

    if len(values) == 0:
        raise ValueError(
            "No finite threshold values."
        )

    if len(values) == 1:
        epsilon = max(
            abs(
                values[0]
            )
            * 1e-6,
            1e-6,
        )

        return np.asarray(
            [
                values[0]
                - epsilon,
                values[0]
                + epsilon,
            ],
            dtype=np.float64,
        )

    midpoints = (
        values[:-1]
        + values[1:]
    ) / 2.0

    epsilon = max(
        float(
            np.ptp(
                values
            )
        )
        * 1e-6,
        1e-9,
    )

    return np.concatenate(
        [
            [
                values[0]
                - epsilon
            ],
            midpoints,
            [
                values[-1]
                + epsilon
            ],
        ]
    )


# =========================================================
# RULE EVALUATION
# =========================================================

def evaluate_acceptance_mask(
    *,
    positive_candidates,
    negative_candidates,
    positive_mask,
    negative_mask,
    events,
    positive_cases,
):
    accepted_positive = (
        positive_candidates[
            positive_mask
        ]
    )

    accepted_negative = (
        negative_candidates[
            negative_mask
        ]
    )

    event_hits = 0

    for _, event in (
        events.iterrows()
    ):
        candidates = (
            accepted_positive[
                accepted_positive[
                    "video_id"
                ]
                .astype(str)
                .eq(
                    str(
                        event[
                            "video_id"
                        ]
                    )
                )
            ]
        )

        if not candidates.empty:
            distances = (
                candidates[
                    "timecode_sec"
                ]
                .astype(float)
                .sub(
                    float(
                        event[
                            "event_time_sec"
                        ]
                    )
                )
                .abs()
            )

            if (
                distances
                <= EVENT_MATCH_RADIUS_SEC
            ).any():
                event_hits += 1

    event_recall = float(
        event_hits
        / len(
            events
        )
    )

    positive_video_hits = 0

    for case in positive_cases:
        video_id = str(
            case[
                "video_id"
            ]
        )

        video_candidates = (
            accepted_positive[
                accepted_positive[
                    "video_id"
                ]
                .astype(str)
                .eq(
                    video_id
                )
            ]
        )

        if not video_candidates.empty:
            positive_video_hits += 1

    positive_video_recall = float(
        positive_video_hits
        / len(
            positive_cases
        )
    )

    normal_video_ids = (
        negative_candidates[
            "video_id"
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    false_positive_videos = 0

    for video_id in (
        normal_video_ids
    ):
        count = int(
            accepted_negative[
                accepted_negative[
                    "video_id"
                ]
                .astype(str)
                .eq(
                    video_id
                )
            ].shape[0]
        )

        if count > 0:
            false_positive_videos += 1

    normal_specificity = float(
        (
            len(
                normal_video_ids
            )
            - false_positive_videos
        )
        / len(
            normal_video_ids
        )
    )

    false_positive_rate = float(
        false_positive_videos
        / len(
            normal_video_ids
        )
    )

    balanced_score = float(
        (
            event_recall
            + normal_specificity
        )
        / 2.0
    )

    return {
        "event_hits": int(
            event_hits
        ),

        "events": int(
            len(
                events
            )
        ),

        "event_recall": (
            event_recall
        ),

        "positive_video_hits": int(
            positive_video_hits
        ),

        "positive_videos": int(
            len(
                positive_cases
            )
        ),

        "positive_video_recall": (
            positive_video_recall
        ),

        "false_positive_videos": int(
            false_positive_videos
        ),

        "normal_videos": int(
            len(
                normal_video_ids
            )
        ),

        "normal_specificity": (
            normal_specificity
        ),

        "false_positive_rate": (
            false_positive_rate
        ),

        "balanced_score": (
            balanced_score
        ),

        "accepted_positive_candidates": int(
            positive_mask.sum()
        ),

        "accepted_negative_candidates": int(
            negative_mask.sum()
        ),
    }


# =========================================================
# SINGLE SEARCH
# =========================================================

def run_single_search(
    positive,
    negative,
    events,
    positive_cases,
):
    rows = []

    for metric in (
        SINGLE_METRICS
    ):
        all_values = np.concatenate(
            [
                positive[
                    metric
                ].to_numpy(
                    dtype=np.float64
                ),

                negative[
                    metric
                ].to_numpy(
                    dtype=np.float64
                ),
            ]
        )

        for threshold in (
            threshold_grid(
                all_values
            )
        ):
            positive_mask = (
                positive[
                    metric
                ].to_numpy(
                    dtype=np.float64
                )
                >= threshold
            )

            negative_mask = (
                negative[
                    metric
                ].to_numpy(
                    dtype=np.float64
                )
                >= threshold
            )

            metrics = (
                evaluate_acceptance_mask(
                    positive_candidates=(
                        positive
                    ),
                    negative_candidates=(
                        negative
                    ),
                    positive_mask=(
                        positive_mask
                    ),
                    negative_mask=(
                        negative_mask
                    ),
                    events=(
                        events
                    ),
                    positive_cases=(
                        positive_cases
                    ),
                )
            )

            rows.append(
                {
                    "rule_type": (
                        "single"
                    ),

                    "metric": (
                        metric
                    ),

                    "threshold": float(
                        threshold
                    ),

                    **metrics,
                }
            )

    return pd.DataFrame(
        rows
    )


# =========================================================
# PAIR SEARCH
# =========================================================

def compact_grid(
    values,
    max_points=30,
):
    full = (
        threshold_grid(
            values
        )
    )

    if len(full) <= max_points:
        return full

    indices = np.linspace(
        0,
        len(full) - 1,
        max_points,
    )

    indices = np.unique(
        np.round(
            indices
        ).astype(int)
    )

    return full[
        indices
    ]


def run_pair_search(
    positive,
    negative,
    events,
    positive_cases,
):
    rows = []

    for (
        metric_a,
        metric_b,
    ) in PAIR_METRICS:
        values_a = np.concatenate(
            [
                positive[
                    metric_a
                ].to_numpy(
                    dtype=np.float64
                ),

                negative[
                    metric_a
                ].to_numpy(
                    dtype=np.float64
                ),
            ]
        )

        values_b = np.concatenate(
            [
                positive[
                    metric_b
                ].to_numpy(
                    dtype=np.float64
                ),

                negative[
                    metric_b
                ].to_numpy(
                    dtype=np.float64
                ),
            ]
        )

        grid_a = compact_grid(
            values_a
        )

        grid_b = compact_grid(
            values_b
        )

        for (
            threshold_a,
            threshold_b,
        ) in itertools.product(
            grid_a,
            grid_b,
        ):
            positive_mask = (
                (
                    positive[
                        metric_a
                    ].to_numpy(
                        dtype=np.float64
                    )
                    >= threshold_a
                )
                &
                (
                    positive[
                        metric_b
                    ].to_numpy(
                        dtype=np.float64
                    )
                    >= threshold_b
                )
            )

            negative_mask = (
                (
                    negative[
                        metric_a
                    ].to_numpy(
                        dtype=np.float64
                    )
                    >= threshold_a
                )
                &
                (
                    negative[
                        metric_b
                    ].to_numpy(
                        dtype=np.float64
                    )
                    >= threshold_b
                )
            )

            metrics = (
                evaluate_acceptance_mask(
                    positive_candidates=(
                        positive
                    ),
                    negative_candidates=(
                        negative
                    ),
                    positive_mask=(
                        positive_mask
                    ),
                    negative_mask=(
                        negative_mask
                    ),
                    events=(
                        events
                    ),
                    positive_cases=(
                        positive_cases
                    ),
                )
            )

            rows.append(
                {
                    "rule_type": (
                        "pair_and"
                    ),

                    "metric_a": (
                        metric_a
                    ),

                    "threshold_a": float(
                        threshold_a
                    ),

                    "metric_b": (
                        metric_b
                    ),

                    "threshold_b": float(
                        threshold_b
                    ),

                    **metrics,
                }
            )

    return pd.DataFrame(
        rows
    )


# =========================================================
# BEST RULE
# =========================================================

def select_best(
    dataframe,
):
    safe = (
        dataframe[
            dataframe[
                "event_recall"
            ]
            >= TARGET_EVENT_RECALL
        ]
        .copy()
    )

    if not safe.empty:
        ranked = (
            safe
            .sort_values(
                [
                    "normal_specificity",
                    "event_recall",
                    "balanced_score",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
        )

        return (
            ranked.iloc[0],
            True,
        )

    ranked = (
        dataframe
        .sort_values(
            [
                "event_recall",
                "normal_specificity",
                "balanced_score",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
    )

    return (
        ranked.iloc[0],
        False,
    )


# =========================================================
# DISTRIBUTIONS
# =========================================================

def build_distributions(
    positive,
    negative,
):
    rows = []

    for metric in (
        SINGLE_METRICS
    ):
        for (
            source,
            dataframe,
        ) in [
            (
                "violation_video_candidates",
                positive,
            ),
            (
                "normal_video_candidates",
                negative,
            ),
        ]:
            values = (
                dataframe[
                    metric
                ]
                .astype(float)
            )

            rows.append(
                {
                    "metric": (
                        metric
                    ),

                    "source": (
                        source
                    ),

                    "count": int(
                        len(
                            values
                        )
                    ),

                    "mean": float(
                        values.mean()
                    ),

                    "std": float(
                        values.std()
                    ),

                    "min": float(
                        values.min()
                    ),

                    "q25": float(
                        values.quantile(
                            0.25
                        )
                    ),

                    "median": float(
                        values.median()
                    ),

                    "q75": float(
                        values.quantile(
                            0.75
                        )
                    ),

                    "max": float(
                        values.max()
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 8F-B "
        "Candidate Threshold Calibration"
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

    (
        positive,
        negative,
    ) = load_candidates()

    (
        events,
        positive_cases,
    ) = build_event_table()

    print()
    print("DATASET")
    print("-" * 72)

    print(
        "Violation-video candidates:",
        len(
            positive
        ),
    )

    print(
        "Normal-video candidates:",
        len(
            negative
        ),
    )

    print(
        "Violation videos:",
        len(
            positive_cases
        ),
    )

    print(
        "Annotated events:",
        len(
            events
        ),
    )

    print(
        "Normal videos:",
        negative[
            "video_id"
        ].nunique(),
    )

    print(
        "Event match radius:",
        EVENT_MATCH_RADIUS_SEC,
        "sec",
    )

    print(
        "Target event recall:",
        TARGET_EVENT_RECALL,
    )

    print(
        "Development validation used:",
        False,
    )

    # =====================================================
    # DISTRIBUTIONS
    # =====================================================

    distributions = (
        build_distributions(
            positive,
            negative,
        )
    )

    distributions.to_csv(
        DISTRIBUTIONS_FILE,
        index=False,
    )

    # =====================================================
    # SINGLE
    # =====================================================

    print()
    print("SEARCHING SINGLE-METRIC RULES")
    print("-" * 72)

    single = (
        run_single_search(
            positive,
            negative,
            events,
            positive_cases,
        )
    )

    single.to_csv(
        SINGLE_SEARCH_FILE,
        index=False,
    )

    (
        best_single,
        single_meets_target,
    ) = select_best(
        single
    )

    print(
        "Best single metric:",
        best_single[
            "metric"
        ],
    )

    print(
        "Threshold:",
        round(
            float(
                best_single[
                    "threshold"
                ]
            ),
            6,
        ),
    )

    print(
        "Event recall:",
        round(
            float(
                best_single[
                    "event_recall"
                ]
            ),
            4,
        ),
    )

    print(
        "Normal specificity:",
        round(
            float(
                best_single[
                    "normal_specificity"
                ]
            ),
            4,
        ),
    )

    # =====================================================
    # PAIRS
    # =====================================================

    print()
    print("SEARCHING TWO-METRIC AND RULES")
    print("-" * 72)

    pairs = (
        run_pair_search(
            positive,
            negative,
            events,
            positive_cases,
        )
    )

    pairs.to_csv(
        PAIR_SEARCH_FILE,
        index=False,
    )

    (
        best_pair,
        pair_meets_target,
    ) = select_best(
        pairs
    )

    print(
        "Best pair:",
        best_pair[
            "metric_a"
        ],
        "+",
        best_pair[
            "metric_b"
        ],
    )

    print(
        "Threshold A:",
        round(
            float(
                best_pair[
                    "threshold_a"
                ]
            ),
            6,
        ),
    )

    print(
        "Threshold B:",
        round(
            float(
                best_pair[
                    "threshold_b"
                ]
            ),
            6,
        ),
    )

    print(
        "Event recall:",
        round(
            float(
                best_pair[
                    "event_recall"
                ]
            ),
            4,
        ),
    )

    print(
        "Normal specificity:",
        round(
            float(
                best_pair[
                    "normal_specificity"
                ]
            ),
            4,
        ),
    )

    # =====================================================
    # CHOOSE SINGLE VS PAIR
    # =====================================================

    candidate_rules = []

    candidate_rules.append(
        {
            "kind": (
                "single"
            ),

            "meets_target": (
                single_meets_target
            ),

            "normal_specificity": float(
                best_single[
                    "normal_specificity"
                ]
            ),

            "event_recall": float(
                best_single[
                    "event_recall"
                ]
            ),

            "balanced_score": float(
                best_single[
                    "balanced_score"
                ]
            ),

            "row": (
                best_single
            ),
        }
    )

    candidate_rules.append(
        {
            "kind": (
                "pair_and"
            ),

            "meets_target": (
                pair_meets_target
            ),

            "normal_specificity": float(
                best_pair[
                    "normal_specificity"
                ]
            ),

            "event_recall": float(
                best_pair[
                    "event_recall"
                ]
            ),

            "balanced_score": float(
                best_pair[
                    "balanced_score"
                ]
            ),

            "row": (
                best_pair
            ),
        }
    )

    target_rules = [
        item
        for item in candidate_rules
        if item[
            "meets_target"
        ]
    ]

    if target_rules:
        selected = max(
            target_rules,
            key=lambda item: (
                item[
                    "normal_specificity"
                ],
                item[
                    "event_recall"
                ],
                item[
                    "balanced_score"
                ],
            ),
        )
    else:
        selected = max(
            candidate_rules,
            key=lambda item: (
                item[
                    "event_recall"
                ],
                item[
                    "normal_specificity"
                ],
                item[
                    "balanced_score"
                ],
            ),
        )

    # =====================================================
    # SERIALIZE RULE
    # =====================================================

    if (
        selected[
            "kind"
        ]
        == "single"
    ):
        row = (
            selected[
                "row"
            ]
        )

        rule = {
            "rule_type": (
                "single_threshold"
            ),

            "metric": str(
                row[
                    "metric"
                ]
            ),

            "threshold": float(
                row[
                    "threshold"
                ]
            ),
        }

    else:
        row = (
            selected[
                "row"
            ]
        )

        rule = {
            "rule_type": (
                "and_threshold"
            ),

            "metric_a": str(
                row[
                    "metric_a"
                ]
            ),

            "threshold_a": float(
                row[
                    "threshold_a"
                ]
            ),

            "metric_b": str(
                row[
                    "metric_b"
                ]
            ),

            "threshold_b": float(
                row[
                    "threshold_b"
                ]
            ),
        }

    selected_metrics = {
        "event_recall": float(
            row[
                "event_recall"
            ]
        ),

        "normal_specificity": float(
            row[
                "normal_specificity"
            ]
        ),

        "false_positive_rate": float(
            row[
                "false_positive_rate"
            ]
        ),

        "balanced_score": float(
            row[
                "balanced_score"
            ]
        ),

        "event_hits": int(
            row[
                "event_hits"
            ]
        ),

        "events": int(
            row[
                "events"
            ]
        ),

        "false_positive_videos": int(
            row[
                "false_positive_videos"
            ]
        ),

        "normal_videos": int(
            row[
                "normal_videos"
            ]
        ),
    }

    rule_payload = {
        "name": (
            "SecureVision Candidate Acceptance Rule"
        ),

        "status": (
            "training_calibrated_not_yet_development_validated"
        ),

        "development_validation_used": (
            False
        ),

        "target_event_recall": (
            TARGET_EVENT_RECALL
        ),

        "event_match_radius_sec": (
            EVENT_MATCH_RADIUS_SEC
        ),

        "rule": (
            rule
        ),

        "training_metrics": (
            selected_metrics
        ),
    }

    RULE_FILE.write_text(
        json.dumps(
            rule_payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = {
        "configuration": {
            "development_validation_used": (
                False
            ),

            "target_event_recall": (
                TARGET_EVENT_RECALL
            ),

            "event_match_radius_sec": (
                EVENT_MATCH_RADIUS_SEC
            ),
        },

        "dataset": {
            "violation_videos": int(
                len(
                    positive_cases
                )
            ),

            "events": int(
                len(
                    events
                )
            ),

            "normal_videos": int(
                negative[
                    "video_id"
                ].nunique()
            ),

            "violation_candidates": int(
                len(
                    positive
                )
            ),

            "normal_candidates": int(
                len(
                    negative
                )
            ),
        },

        "best_single": {
            "metric": str(
                best_single[
                    "metric"
                ]
            ),

            "threshold": float(
                best_single[
                    "threshold"
                ]
            ),

            "event_recall": float(
                best_single[
                    "event_recall"
                ]
            ),

            "normal_specificity": float(
                best_single[
                    "normal_specificity"
                ]
            ),

            "false_positive_rate": float(
                best_single[
                    "false_positive_rate"
                ]
            ),

            "balanced_score": float(
                best_single[
                    "balanced_score"
                ]
            ),

            "meets_target": bool(
                single_meets_target
            ),
        },

        "best_pair": {
            "metric_a": str(
                best_pair[
                    "metric_a"
                ]
            ),

            "threshold_a": float(
                best_pair[
                    "threshold_a"
                ]
            ),

            "metric_b": str(
                best_pair[
                    "metric_b"
                ]
            ),

            "threshold_b": float(
                best_pair[
                    "threshold_b"
                ]
            ),

            "event_recall": float(
                best_pair[
                    "event_recall"
                ]
            ),

            "normal_specificity": float(
                best_pair[
                    "normal_specificity"
                ]
            ),

            "false_positive_rate": float(
                best_pair[
                    "false_positive_rate"
                ]
            ),

            "balanced_score": float(
                best_pair[
                    "balanced_score"
                ]
            ),

            "meets_target": bool(
                pair_meets_target
            ),
        },

        "selected_rule": (
            rule_payload
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # =====================================================
    # PRINT
    # =====================================================

    print()
    print("=" * 72)
    print("SELECTED TRAINING RULE")
    print("=" * 72)

    print(
        json.dumps(
            rule,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(
        "Event recall:",
        round(
            selected_metrics[
                "event_recall"
            ],
            4,
        ),
    )

    print(
        "Event hits:",
        f"{selected_metrics['event_hits']}/"
        f"{selected_metrics['events']}",
    )

    print(
        "Normal specificity:",
        round(
            selected_metrics[
                "normal_specificity"
            ],
            4,
        ),
    )

    print(
        "Normal false positives:",
        f"{selected_metrics['false_positive_videos']}/"
        f"{selected_metrics['normal_videos']}",
    )

    print(
        "Balanced score:",
        round(
            selected_metrics[
                "balanced_score"
            ],
            4,
        ),
    )

    print()
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    print(
        DISTRIBUTIONS_FILE
    )

    print(
        SINGLE_SEARCH_FILE
    )

    print(
        PAIR_SEARCH_FILE
    )

    print(
        SUMMARY_FILE
    )

    print(
        RULE_FILE
    )

    print()
    print(
        "✅ Candidate calibration complete."
    )

    print(
        "✅ Development validation was not used."
    )

    print(
        "⚠ Rule is training-calibrated only."
    )


if __name__ == "__main__":
    main()