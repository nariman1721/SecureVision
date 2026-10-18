from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.routes.analyze import (
    build_api_result,
)

from app.video.production_inference import (
    ProductionVideoInference,
)


AI_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RAW_DIR = (
    AI_ROOT
    / "data"
    / "raw"
)

PROCESSED_DIR = (
    AI_ROOT
    / "data"
    / "processed"
)

REPORTS_DIR = (
    AI_ROOT
    / "reports"
)


VIDEO_REPORT_FILE = (
    REPORTS_DIR
    / "final_development_validation_videos.csv"
)

EVENT_REPORT_FILE = (
    REPORTS_DIR
    / "final_development_validation_events.csv"
)

SUMMARY_FILE = (
    REPORTS_DIR
    / "final_development_validation_summary.json"
)

API_PREVIEW_FILE = (
    REPORTS_DIR
    / "final_development_validation_api_preview.json"
)


# =========================================================
# FROZEN DEVELOPMENT SET
# =========================================================
#
# These are the 8 development-validation videos created
# by the earlier content-safe split.
#
# IMPORTANT:
# This set has already been inspected during previous
# development phases. It must NOT be described as an
# untouched test set.
#
# After this Phase 9D run, do not tune the classifier,
# thresholds or temporal ranking using these results.
# =========================================================

DEV_CASES = [
    {
        "video_id": "X-1",
        "label": 1,
        "event_times": [22.0],
    },
    {
        "video_id": "X-2",
        "label": 1,
        "event_times": [10.0],
    },
    {
        "video_id": "X-26",
        "label": 1,
        "event_times": [9.0],
    },
    {
        "video_id": "X-40",
        "label": 1,
        "event_times": [17.0],
    },
    {
        "video_id": "X-16",
        "label": 0,
        "event_times": [],
    },
    {
        "video_id": "X-19",
        "label": 0,
        "event_times": [],
    },
    {
        "video_id": "X-25",
        "label": 0,
        "event_times": [],
    },
    {
        "video_id": "X-35",
        "label": 0,
        "event_times": [],
    },
]


# =========================================================
# VIDEO RESOLUTION
# =========================================================

def candidate_video_names(
    video_id: str,
) -> list[str]:
    return [
        f"{video_id}.mov",

        (
            f"{video_id.replace('X', 'Х')}"
            ".mov"
        ),
    ]


def resolve_video_path(
    video_id: str,
) -> Path:
    for name in candidate_video_names(
        video_id
    ):
        path = (
            RAW_DIR
            / name
        )

        if path.exists():
            return path.resolve()

    raise FileNotFoundError(
        f"Unable to find raw video "
        f"for {video_id}"
    )


# =========================================================
# SPLIT SAFETY CHECK
# =========================================================

def check_train_leakage() -> dict[str, Any]:
    train_manifest = (
        PROCESSED_DIR
        / "train_clips.csv"
    )

    if not train_manifest.exists():
        return {
            "checked": False,
            "reason": (
                "train_clips.csv "
                "not found"
            ),
            "development_ids_in_train": [],
        }

    dataframe = pd.read_csv(
        train_manifest
    )

    if (
        "canonical_video_id"
        not in dataframe.columns
    ):
        return {
            "checked": False,
            "reason": (
                "canonical_video_id "
                "column not found"
            ),
            "development_ids_in_train": [],
        }

    train_ids = set(
        dataframe[
            "canonical_video_id"
        ]
        .astype(str)
        .tolist()
    )

    development_ids = {
        case[
            "video_id"
        ]
        for case
        in DEV_CASES
    }

    overlap = sorted(
        development_ids
        & train_ids
    )

    return {
        "checked": True,
        "development_ids_in_train": (
            overlap
        ),
        "leakage_detected": bool(
            overlap
        ),
    }


# =========================================================
# API CONTRACT VALIDATION
# =========================================================

def validate_jpeg_base64(
    value: str,
) -> bool:
    try:
        raw = base64.b64decode(
            value,
            validate=True,
        )

    except Exception:
        return False

    if len(raw) < 4:
        return False

    # JPEG SOI
    if raw[:2] != b"\xff\xd8":
        return False

    # JPEG EOI
    if raw[-2:] != b"\xff\xd9":
        return False

    return True


def validate_api_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    required_top_level = [
        "verdict",
        "detected",
        "violation_count",
        "violations",
        "analysis",
    ]

    missing_top_level = [
        key
        for key
        in required_top_level
        if key not in result
    ]

    violations_valid = True
    evidence_valid = True
    legal_valid = True

    evidence_frames_total = 0

    violation_checks = []

    for index, violation in enumerate(
        result.get(
            "violations",
            [],
        ),
        start=1,
    ):
        required_violation = [
            "type",
            "timecode_sec",
            "confidence",
            "description",
            "evidence_frames",
        ]

        missing = [
            key
            for key
            in required_violation
            if key not in violation
        ]

        frame_values = (
            violation.get(
                "evidence_frames",
                [],
            )
        )

        frame_count = len(
            frame_values
        )

        evidence_frames_total += (
            frame_count
        )

        correct_frame_count = (
            1
            <= frame_count
            <= 3
        )

        frames_are_valid = (
            correct_frame_count
            and all(
                isinstance(
                    value,
                    str,
                )
                and validate_jpeg_base64(
                    value
                )
                for value
                in frame_values
            )
        )

        source = (
            violation.get(
                "normative_source"
            )
        )

        rule_context_valid = (
            violation.get(
                "legal_status"
            )
            == "requires_human_review"
            and violation.get(
                "automated_legal_conclusion"
            )
            is False
            and isinstance(
                source,
                dict,
            )
            and bool(
                source.get(
                    "official_url"
                )
            )
        )

        current_valid = (
            not missing
            and frames_are_valid
            and rule_context_valid
        )

        violations_valid = (
            violations_valid
            and not missing
        )

        evidence_valid = (
            evidence_valid
            and frames_are_valid
        )

        legal_valid = (
            legal_valid
            and rule_context_valid
        )

        violation_checks.append(
            {
                "violation_index": (
                    index
                ),
                "missing_fields": (
                    missing
                ),
                "evidence_count": (
                    frame_count
                ),
                "evidence_valid": (
                    frames_are_valid
                ),
                "legal_context_valid": (
                    rule_context_valid
                ),
                "valid": (
                    current_valid
                ),
            }
        )

    # For a normal result there are no violation
    # evidence frames to validate.
    if not result.get(
        "detected",
        False,
    ):
        evidence_valid = (
            len(
                result.get(
                    "violations",
                    [],
                )
            )
            == 0
        )

        legal_valid = True

    payload_valid = (
        not missing_top_level
        and violations_valid
        and evidence_valid
        and legal_valid
        and int(
            result.get(
                "violation_count",
                -1,
            )
        )
        == len(
            result.get(
                "violations",
                [],
            )
        )
    )

    return {
        "payload_valid": (
            payload_valid
        ),
        "missing_top_level": (
            missing_top_level
        ),
        "violations_valid": (
            violations_valid
        ),
        "evidence_valid": (
            evidence_valid
        ),
        "legal_context_valid": (
            legal_valid
        ),
        "evidence_frames_total": (
            evidence_frames_total
        ),
        "violation_checks": (
            violation_checks
        ),
    }


# =========================================================
# EVENT MATCHING
# =========================================================

def one_to_one_event_matching(
    event_times: list[float],
    predicted_times: list[float],
) -> list[dict[str, Any]]:
    if not event_times:
        return []

    if not predicted_times:
        return [
            {
                "event_time_sec": float(
                    event
                ),
                "predicted_time_sec": None,
                "error_sec": None,
            }
            for event
            in event_times
        ]

    pairs = []

    for event_index, event in enumerate(
        event_times
    ):
        for prediction_index, prediction in enumerate(
            predicted_times
        ):
            pairs.append(
                (
                    abs(
                        float(
                            event
                        )
                        - float(
                            prediction
                        )
                    ),
                    event_index,
                    prediction_index,
                )
            )

    pairs.sort(
        key=lambda item: (
            item[0]
        )
    )

    matched_events = set()
    matched_predictions = set()

    matches: dict[
        int,
        tuple[float, float]
    ] = {}

    for (
        error,
        event_index,
        prediction_index,
    ) in pairs:
        if (
            event_index
            in matched_events
        ):
            continue

        if (
            prediction_index
            in matched_predictions
        ):
            continue

        matched_events.add(
            event_index
        )

        matched_predictions.add(
            prediction_index
        )

        matches[
            event_index
        ] = (
            float(
                predicted_times[
                    prediction_index
                ]
            ),
            float(
                error
            ),
        )

    rows = []

    for event_index, event in enumerate(
        event_times
    ):
        if event_index in matches:
            (
                prediction,
                error,
            ) = matches[
                event_index
            ]

        else:
            prediction = None
            error = None

        rows.append(
            {
                "event_time_sec": float(
                    event
                ),
                "predicted_time_sec": (
                    prediction
                ),
                "error_sec": (
                    error
                ),
            }
        )

    return rows


# =========================================================
# METRICS
# =========================================================

def safe_divide(
    numerator: int | float,
    denominator: int | float,
) -> float:
    if denominator == 0:
        return 0.0

    return float(
        numerator
        / denominator
    )


def build_classification_metrics(
    rows: pd.DataFrame,
) -> dict[str, Any]:
    true_values = (
        rows[
            "true_label"
        ]
        .astype(int)
    )

    predicted_values = (
        rows[
            "predicted_label"
        ]
        .astype(int)
    )

    tp = int(
        (
            (
                true_values == 1
            )
            & (
                predicted_values == 1
            )
        )
        .sum()
    )

    tn = int(
        (
            (
                true_values == 0
            )
            & (
                predicted_values == 0
            )
        )
        .sum()
    )

    fp = int(
        (
            (
                true_values == 0
            )
            & (
                predicted_values == 1
            )
        )
        .sum()
    )

    fn = int(
        (
            (
                true_values == 1
            )
            & (
                predicted_values == 0
            )
        )
        .sum()
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    accuracy = safe_divide(
        tp + tn,
        len(
            rows
        ),
    )

    balanced_accuracy = (
        recall
        + specificity
    ) / 2.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (
            accuracy
        ),
        "balanced_accuracy": (
            balanced_accuracy
        ),
        "precision": (
            precision
        ),
        "recall": (
            recall
        ),
        "specificity": (
            specificity
        ),
        "false_positive_rate": (
            1.0
            - specificity
        ),
    }


def build_localization_metrics(
    events: pd.DataFrame,
) -> dict[str, Any]:
    if events.empty:
        return {
            "events": 0,
        }

    errors = (
        events[
            "error_sec"
        ]
        .dropna()
        .astype(float)
        .to_numpy()
    )

    total_events = int(
        len(
            events
        )
    )

    matched_events = int(
        len(
            errors
        )
    )

    within_1 = int(
        (
            errors <= 1.0
        )
        .sum()
    )

    within_3 = int(
        (
            errors <= 3.0
        )
        .sum()
    )

    within_5 = int(
        (
            errors <= 5.0
        )
        .sum()
    )

    within_10 = int(
        (
            errors <= 10.0
        )
        .sum()
    )

    return {
        "events": (
            total_events
        ),

        "events_with_accepted_candidate": (
            matched_events
        ),

        "events_without_accepted_candidate": int(
            total_events
            - matched_events
        ),

        "mean_error_sec_matched": (
            float(
                np.mean(
                    errors
                )
            )
            if matched_events
            else None
        ),

        "median_error_sec_matched": (
            float(
                np.median(
                    errors
                )
            )
            if matched_events
            else None
        ),

        "within_1_sec": (
            within_1
        ),

        "within_3_sec": (
            within_3
        ),

        "within_5_sec": (
            within_5
        ),

        "within_10_sec": (
            within_10
        ),

        "within_1_sec_rate": (
            safe_divide(
                within_1,
                total_events,
            )
        ),

        "within_3_sec_rate": (
            safe_divide(
                within_3,
                total_events,
            )
        ),

        "within_5_sec_rate": (
            safe_divide(
                within_5,
                total_events,
            )
        ),

        "within_10_sec_rate": (
            safe_divide(
                within_10,
                total_events,
            )
        ),
    }


# =========================================================
# COMPACT API PREVIEW
# =========================================================

def build_compact_preview(
    video_id: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    compact_violations = []

    for violation in (
        result.get(
            "violations",
            []
        )
    ):
        evidence = (
            violation.get(
                "evidence_frames",
                [],
            )
        )

        compact_violations.append(
            {
                "type": (
                    violation.get(
                        "type"
                    )
                ),

                "timecode_sec": (
                    violation.get(
                        "timecode_sec"
                    )
                ),

                "confidence": (
                    violation.get(
                        "confidence"
                    )
                ),

                "evidence_count": len(
                    evidence
                ),

                "evidence_base64_sizes": [
                    len(
                        value
                    )
                    for value
                    in evidence
                ],

                "legal_status": (
                    violation.get(
                        "legal_status"
                    )
                ),

                "normative_source_id": (
                    (
                        violation.get(
                            "normative_source"
                        )
                        or {}
                    )
                    .get(
                        "source_id"
                    )
                ),
            }
        )

    return {
        "video_id": (
            video_id
        ),

        "verdict": (
            result[
                "verdict"
            ]
        ),

        "detected": (
            result[
                "detected"
            ]
        ),

        "violation_count": (
            result[
                "violation_count"
            ]
        ),

        "violations": (
            compact_violations
        ),

        "analysis": (
            result.get(
                "analysis"
            )
        ),
    }


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 9D "
        "Final Development Validation"
    )
    print("=" * 72)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    positive_cases = [
        case
        for case
        in DEV_CASES
        if case[
            "label"
        ]
        == 1
    ]

    normal_cases = [
        case
        for case
        in DEV_CASES
        if case[
            "label"
        ]
        == 0
    ]

    if len(
        DEV_CASES
    ) != 8:
        raise RuntimeError(
            "Expected exactly "
            "8 development videos."
        )

    if len(
        positive_cases
    ) != 4:
        raise RuntimeError(
            "Expected exactly "
            "4 violation videos."
        )

    if len(
        normal_cases
    ) != 4:
        raise RuntimeError(
            "Expected exactly "
            "4 normal videos."
        )

    split_safety = (
        check_train_leakage()
    )

    print()
    print("DATASET")
    print("-" * 72)

    print(
        "Development videos:",
        len(
            DEV_CASES
        ),
    )

    print(
        "Violation videos:",
        len(
            positive_cases
        ),
    )

    print(
        "Normal videos:",
        len(
            normal_cases
        ),
    )

    print(
        "Annotated events:",
        sum(
            len(
                case[
                    "event_times"
                ]
            )
            for case
            in DEV_CASES
        ),
    )

    print(
        "Previously inspected "
        "development set:",
        True,
    )

    print(
        "Training leakage check:",
        split_safety,
    )

    if (
        split_safety.get(
            "leakage_detected"
        )
    ):
        raise RuntimeError(
            "Development video IDs "
            "were found in train_clips.csv."
        )

    print()
    print("LOADING PRODUCTION PIPELINE")
    print("-" * 72)

    inference = (
        ProductionVideoInference()
    )

    print(
        "Classifier:",
        inference.classifier_path.name,
    )

    print(
        "Device:",
        inference.localizer.device,
    )

    print(
        "Threshold:",
        inference.localizer.threshold,
    )

    video_rows = []
    event_rows = []
    api_previews = []

    all_contract_valid = True
    all_evidence_valid = True
    all_legal_valid = True

    total_evidence_frames = 0

    # =====================================================
    # PROCESS ALL 8 DEVELOPMENT VIDEOS
    # =====================================================

    for index, case in enumerate(
        DEV_CASES,
        start=1,
    ):
        video_id = (
            case[
                "video_id"
            ]
        )

        true_label = int(
            case[
                "label"
            ]
        )

        event_times = [
            float(
                value
            )
            for value
            in case[
                "event_times"
            ]
        ]

        video_path = (
            resolve_video_path(
                video_id
            )
        )

        print()
        print("=" * 72)

        print(
            f"[{index}/"
            f"{len(DEV_CASES)}] "
            f"{video_id}"
        )

        print("=" * 72)

        print(
            "Ground truth:",
            (
                "violation"
                if true_label
                else "normal"
            ),
        )

        if event_times:
            print(
                "Event times:",
                event_times,
            )

        print(
            "Video:",
            video_path,
        )

        production_result = (
            inference.analyze(
                video_path,
                show_progress=True,
                include_evidence=True,
            )
        )

        # Convert through the exact adapter
        # used by Phase 9C API.
        api_result = (
            build_api_result(
                production_result
            )
        )

        contract = (
            validate_api_result(
                api_result
            )
        )

        all_contract_valid = (
            all_contract_valid
            and bool(
                contract[
                    "payload_valid"
                ]
            )
        )

        all_evidence_valid = (
            all_evidence_valid
            and bool(
                contract[
                    "evidence_valid"
                ]
            )
        )

        all_legal_valid = (
            all_legal_valid
            and bool(
                contract[
                    "legal_context_valid"
                ]
            )
        )

        total_evidence_frames += int(
            contract[
                "evidence_frames_total"
            ]
        )

        predicted_label = int(
            bool(
                api_result[
                    "detected"
                ]
            )
        )

        predicted_times = [
            float(
                violation[
                    "timecode_sec"
                ]
            )
            for violation
            in api_result[
                "violations"
            ]
        ]

        matches = (
            one_to_one_event_matching(
                event_times,
                predicted_times,
            )
        )

        for event_index, match in enumerate(
            matches,
            start=1,
        ):
            error = (
                match[
                    "error_sec"
                ]
            )

            event_rows.append(
                {
                    "video_id": (
                        video_id
                    ),

                    "event_index": (
                        event_index
                    ),

                    "event_time_sec": (
                        match[
                            "event_time_sec"
                        ]
                    ),

                    "predicted_time_sec": (
                        match[
                            "predicted_time_sec"
                        ]
                    ),

                    "error_sec": (
                        error
                    ),

                    "within_1_sec": (
                        False
                        if error is None
                        else error <= 1.0
                    ),

                    "within_3_sec": (
                        False
                        if error is None
                        else error <= 3.0
                    ),

                    "within_5_sec": (
                        False
                        if error is None
                        else error <= 5.0
                    ),

                    "within_10_sec": (
                        False
                        if error is None
                        else error <= 10.0
                    ),
                }
            )

        processing_time = float(
            api_result[
                "analysis"
            ][
                "processing_time_sec"
            ]
        )

        video_rows.append(
            {
                "video_id": (
                    video_id
                ),

                "true_label": (
                    true_label
                ),

                "predicted_label": (
                    predicted_label
                ),

                "correct": bool(
                    true_label
                    == predicted_label
                ),

                "true_class": (
                    "violation"
                    if true_label
                    else "normal"
                ),

                "predicted_class": (
                    "violation"
                    if predicted_label
                    else "normal"
                ),

                "violation_count": int(
                    api_result[
                        "violation_count"
                    ]
                ),

                "predicted_timecodes": (
                    json.dumps(
                        predicted_times
                    )
                ),

                "processing_time_sec": (
                    processing_time
                ),

                "duration_sec": float(
                    api_result[
                        "analysis"
                    ][
                        "duration_sec"
                    ]
                ),

                "windows_scanned": int(
                    api_result[
                        "analysis"
                    ][
                        "windows_scanned"
                    ]
                ),

                "payload_valid": bool(
                    contract[
                        "payload_valid"
                    ]
                ),

                "evidence_valid": bool(
                    contract[
                        "evidence_valid"
                    ]
                ),

                "legal_context_valid": bool(
                    contract[
                        "legal_context_valid"
                    ]
                ),

                "evidence_frames": int(
                    contract[
                        "evidence_frames_total"
                    ]
                ),
            }
        )

        api_previews.append(
            build_compact_preview(
                video_id,
                api_result,
            )
        )

        print()
        print("RESULT")
        print("-" * 72)

        print(
            "Prediction:",
            api_result[
                "verdict"
            ],
        )

        print(
            "Correct:",
            true_label
            == predicted_label,
        )

        print(
            "Violations:",
            api_result[
                "violation_count"
            ],
        )

        print(
            "Timecodes:",
            predicted_times,
        )

        print(
            "Processing time:",
            processing_time,
        )

        print(
            "API payload valid:",
            contract[
                "payload_valid"
            ],
        )

        print(
            "Evidence valid:",
            contract[
                "evidence_valid"
            ],
        )

        print(
            "Legal context valid:",
            contract[
                "legal_context_valid"
            ],
        )

    # =====================================================
    # BUILD REPORTS
    # =====================================================

    videos = pd.DataFrame(
        video_rows
    )

    events = pd.DataFrame(
        event_rows
    )

    classification = (
        build_classification_metrics(
            videos
        )
    )

    localization = (
        build_localization_metrics(
            events
        )
    )

    processing_times = (
        videos[
            "processing_time_sec"
        ]
        .astype(float)
        .to_numpy()
    )

    runtime_summary = {
        "total_processing_time_sec": float(
            processing_times.sum()
        ),

        "mean_processing_time_sec": float(
            processing_times.mean()
        ),

        "median_processing_time_sec": float(
            np.median(
                processing_times
            )
        ),

        "max_processing_time_sec": float(
            processing_times.max()
        ),

        "min_processing_time_sec": float(
            processing_times.min()
        ),

        "videos_under_300_sec": int(
            (
                processing_times
                < 300.0
            )
            .sum()
        ),

        "all_videos_under_300_sec": bool(
            (
                processing_times
                < 300.0
            )
            .all()
        ),
    }

    api_contract = {
        "all_payloads_valid": bool(
            all_contract_valid
        ),

        "all_evidence_valid_jpeg_base64": bool(
            all_evidence_valid
        ),

        "all_legal_context_valid": bool(
            all_legal_valid
        ),

        "total_evidence_frames": int(
            total_evidence_frames
        ),

        "api_adapter": (
            "app.routes.analyze.build_api_result"
        ),
    }

    report = {
        "phase": (
            "9D"
        ),

        "name": (
            "Final End-to-End "
            "Development Validation"
        ),

        "methodology": {
            "split": (
                "development validation"
            ),

            "previously_inspected": (
                True
            ),

            "untouched_test_set": (
                False
            ),

            "model_or_threshold_tuning_after_run": (
                False
            ),

            "true_final_test": (
                "competition hidden videos"
            ),

            "classifier": (
                inference.classifier_path.name
            ),

            "classifier_threshold": float(
                inference.localizer.threshold
            ),

            "evidence_enabled": (
                True
            ),

            "api_adapter_used": (
                True
            ),

            "training_leakage_check": (
                split_safety
            ),
        },

        "dataset": {
            "videos": int(
                len(
                    videos
                )
            ),

            "violation_videos": int(
                (
                    videos[
                        "true_label"
                    ]
                    == 1
                )
                .sum()
            ),

            "normal_videos": int(
                (
                    videos[
                        "true_label"
                    ]
                    == 0
                )
                .sum()
            ),

            "annotated_events": int(
                len(
                    events
                )
            ),
        },

        "classification": (
            classification
        ),

        "localization": (
            localization
        ),

        "runtime": (
            runtime_summary
        ),

        "api_contract": (
            api_contract
        ),
    }

    videos.to_csv(
        VIDEO_REPORT_FILE,
        index=False,
    )

    events.to_csv(
        EVENT_REPORT_FILE,
        index=False,
    )

    SUMMARY_FILE.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    API_PREVIEW_FILE.write_text(
        json.dumps(
            api_previews,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # =====================================================
    # FINAL CONSOLE SUMMARY
    # =====================================================

    print()
    print("=" * 72)
    print("FINAL DEVELOPMENT VALIDATION")
    print("=" * 72)

    print()
    print("CLASSIFICATION")
    print("-" * 72)

    print(
        "TP:",
        classification[
            "tp"
        ],
    )

    print(
        "TN:",
        classification[
            "tn"
        ],
    )

    print(
        "FP:",
        classification[
            "fp"
        ],
    )

    print(
        "FN:",
        classification[
            "fn"
        ],
    )

    print(
        "Accuracy:",
        round(
            classification[
                "accuracy"
            ],
            4,
        ),
    )

    print(
        "Balanced accuracy:",
        round(
            classification[
                "balanced_accuracy"
            ],
            4,
        ),
    )

    print(
        "Recall:",
        round(
            classification[
                "recall"
            ],
            4,
        ),
    )

    print(
        "Specificity:",
        round(
            classification[
                "specificity"
            ],
            4,
        ),
    )

    print(
        "Precision:",
        round(
            classification[
                "precision"
            ],
            4,
        ),
    )

    print()
    print("LOCALIZATION")
    print("-" * 72)

    print(
        json.dumps(
            localization,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("RUNTIME")
    print("-" * 72)

    print(
        json.dumps(
            runtime_summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("API CONTRACT")
    print("-" * 72)

    print(
        json.dumps(
            api_contract,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("PER VIDEO")
    print("-" * 72)

    print(
        videos[
            [
                "video_id",
                "true_class",
                "predicted_class",
                "correct",
                "violation_count",
                "predicted_timecodes",
                "processing_time_sec",
            ]
        ]
        .to_string(
            index=False
        )
    )

    if not events.empty:
        print()
        print("PER EVENT")
        print("-" * 72)

        print(
            events.to_string(
                index=False
            )
        )

    print()
    print("FILES CREATED")
    print("-" * 72)

    print(
        VIDEO_REPORT_FILE
    )

    print(
        EVENT_REPORT_FILE
    )

    print(
        SUMMARY_FILE
    )

    print(
        API_PREVIEW_FILE
    )

    print()
    print(
        "✅ Phase 9D development "
        "validation complete."
    )

    print(
        "✅ Production inference, "
        "evidence and API payload "
        "contract were evaluated."
    )

    print(
        "⚠ This development set was "
        "previously inspected."
    )

    print(
        "⚠ Do not tune the model or "
        "threshold using these results."
    )

    print(
        "✅ Competition hidden videos "
        "remain the true final test."
    )


if __name__ == "__main__":
    main()