from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.video.temporal_localizer import (
    TemporalLocalizer,
)


AI_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

VALIDATION_CLIPS_FILE = (
    AI_ROOT
    / "data"
    / "processed"
    / "validation_clips.csv"
)

REPORTS_DIR = (
    AI_ROOT
    / "reports"
)

CSV_REPORT_FILE = (
    REPORTS_DIR
    / "temporal_localization_validation.csv"
)

JSON_REPORT_FILE = (
    REPORTS_DIR
    / "temporal_localization_validation.json"
)


TARGET_VIDEO_IDS = [
    "X-1",
    "X-2",
    "X-26",
    "X-40",
]


# =========================================================
# MANIFEST HELPERS
# =========================================================

def load_validation_manifest():
    if not VALIDATION_CLIPS_FILE.exists():
        raise FileNotFoundError(
            f"Missing validation manifest:\n"
            f"{VALIDATION_CLIPS_FILE}"
        )

    dataframe = pd.read_csv(
        VALIDATION_CLIPS_FILE
    )

    required = [
        "canonical_video_id",
        "video_path",
        "clip_type",
        "start_sec",
        "end_sec",
        "label",
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise RuntimeError(
            "Validation manifest missing "
            f"columns: {missing}"
        )

    return dataframe


def resolve_expected_event_time(
    row: pd.Series,
) -> float:
    possible_columns = [
        "event_time_sec",
        "event_sec",
        "timestamp_sec",
        "violation_time_sec",
    ]

    for column in possible_columns:
        if (
            column in row.index
            and pd.notna(
                row[
                    column
                ]
            )
        ):
            return float(
                row[
                    column
                ]
            )

    # Positive clips were generated around
    # the annotated event timestamp.
    return float(
        (
            float(
                row[
                    "start_sec"
                ]
            )
            + float(
                row[
                    "end_sec"
                ]
            )
        )
        / 2.0
    )


def get_test_cases(
    manifest: pd.DataFrame,
):
    cases = []

    for video_id in (
        TARGET_VIDEO_IDS
    ):
        rows = manifest[
            manifest[
                "canonical_video_id"
            ]
            .astype(str)
            .eq(
                video_id
            )
        ]

        if rows.empty:
            raise RuntimeError(
                f"No validation rows "
                f"for {video_id}"
            )

        positive_rows = rows[
            rows[
                "clip_type"
            ]
            .astype(str)
            .eq(
                "positive_event"
            )
        ]

        if positive_rows.empty:
            raise RuntimeError(
                f"No positive_event row "
                f"for {video_id}"
            )

        positive_row = (
            positive_rows.iloc[0]
        )

        video_path = Path(
            str(
                positive_row[
                    "video_path"
                ]
            )
        )

        if not video_path.is_absolute():
            video_path = (
                AI_ROOT
                / video_path
            )

        video_path = (
            video_path.resolve()
        )

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found for "
                f"{video_id}: "
                f"{video_path}"
            )

        event_time_sec = (
            resolve_expected_event_time(
                positive_row
            )
        )

        cases.append(
            {
                "video_id": (
                    video_id
                ),

                "video_path": (
                    video_path
                ),

                "expected_event_sec": (
                    event_time_sec
                ),
            }
        )

    return cases


# =========================================================
# TOP WINDOWS
# =========================================================

def get_top_windows(
    timeline,
    limit: int = 5,
):
    ranked = sorted(
        timeline,
        key=lambda item: (
            item[
                "smoothed_probability"
            ]
        ),
        reverse=True,
    )

    return (
        ranked[
            :limit
        ]
    )


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 8A "
        "Temporal Localization Test"
    )
    print("=" * 72)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = (
        load_validation_manifest()
    )

    test_cases = (
        get_test_cases(
            manifest
        )
    )

    print()
    print("TEST VIDEOS")
    print("-" * 72)

    for case in test_cases:
        print(
            f"{case['video_id']:6} "
            f"expected event = "
            f"{case['expected_event_sec']:.3f}s"
        )

    print()
    print("LOADING MODELS")
    print("-" * 72)

    localizer = (
        TemporalLocalizer(
            scan_window_sec=6.0,
            scan_stride_sec=2.0,
            smoothing_radius=1,
        )
    )

    print(
        "Device:",
        localizer.device,
    )

    print(
        "Classifier threshold:",
        localizer.threshold,
    )

    print(
        "Scan window:",
        localizer.scan_window_sec,
    )

    print(
        "Scan stride:",
        localizer.scan_stride_sec,
    )

    results = []

    full_results = []

    # -----------------------------------------------------
    # Run localization
    # -----------------------------------------------------

    for case in test_cases:
        video_id = (
            case[
                "video_id"
            ]
        )

        expected = float(
            case[
                "expected_event_sec"
            ]
        )

        print()
        print("=" * 72)
        print(
            f"LOCALIZING {video_id}"
        )
        print("=" * 72)

        result = (
            localizer.localize(
                video_path=(
                    case[
                        "video_path"
                    ]
                ),
                show_progress=True,
            )
        )

        predicted = float(
            result[
                "peak"
            ][
                "timecode_sec"
            ]
        )

        probability = float(
            result[
                "peak"
            ][
                "probability"
            ]
        )

        error_sec = abs(
            predicted
            - expected
        )

        top_windows = (
            get_top_windows(
                result[
                    "timeline"
                ],
                limit=5,
            )
        )

        print()
        print("RESULT")
        print("-" * 72)

        print(
            "Duration:",
            round(
                result[
                    "duration_sec"
                ],
                3,
            ),
        )

        print(
            "Windows scanned:",
            result[
                "windows_scanned"
            ],
        )

        print(
            "Detected:",
            result[
                "detected"
            ],
        )

        print(
            "Expected timestamp:",
            round(
                expected,
                3,
            ),
        )

        print(
            "Predicted peak:",
            round(
                predicted,
                3,
            ),
        )

        print(
            "Localization error:",
            round(
                error_sec,
                3,
            ),
        )

        print(
            "Peak probability:",
            round(
                probability,
                4,
            ),
        )

        print()
        print("TOP 5 WINDOWS")
        print("-" * 72)

        for index, item in enumerate(
            top_windows,
            start=1,
        ):
            print(
                f"{index}. "
                f"{item['start_sec']:.3f}"
                f" → "
                f"{item['end_sec']:.3f} "
                f"| center="
                f"{item['center_sec']:.3f} "
                f"| raw="
                f"{item['probability']:.4f} "
                f"| smooth="
                f"{item['smoothed_probability']:.4f}"
            )

        results.append(
            {
                "video_id": (
                    video_id
                ),

                "duration_sec": float(
                    result[
                        "duration_sec"
                    ]
                ),

                "expected_event_sec": (
                    expected
                ),

                "predicted_peak_sec": (
                    predicted
                ),

                "localization_error_sec": (
                    error_sec
                ),

                "peak_probability": (
                    probability
                ),

                "detected": bool(
                    result[
                        "detected"
                    ]
                ),

                "windows_scanned": int(
                    result[
                        "windows_scanned"
                    ]
                ),

                "within_3_sec": bool(
                    error_sec
                    <= 3.0
                ),

                "within_5_sec": bool(
                    error_sec
                    <= 5.0
                ),

                "within_10_sec": bool(
                    error_sec
                    <= 10.0
                ),
            }
        )

        full_results.append(
            {
                "video_id": (
                    video_id
                ),

                "expected_event_sec": (
                    expected
                ),

                "localization": (
                    result
                ),
            }
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    results_df = (
        pd.DataFrame(
            results
        )
    )

    results_df.to_csv(
        CSV_REPORT_FILE,
        index=False,
    )

    mean_error = float(
        results_df[
            "localization_error_sec"
        ].mean()
    )

    median_error = float(
        results_df[
            "localization_error_sec"
        ].median()
    )

    detected_count = int(
        results_df[
            "detected"
        ].sum()
    )

    within_3 = int(
        results_df[
            "within_3_sec"
        ].sum()
    )

    within_5 = int(
        results_df[
            "within_5_sec"
        ].sum()
    )

    within_10 = int(
        results_df[
            "within_10_sec"
        ].sum()
    )

    report = {
        "configuration": {
            "scan_window_sec": (
                6.0
            ),

            "scan_stride_sec": (
                2.0
            ),

            "smoothing_radius": (
                1
            ),

            "classifier_threshold": (
                localizer.threshold
            ),

            "event_timestamps_used_by_localizer": (
                False
            ),
        },

        "summary": {
            "videos": int(
                len(
                    results_df
                )
            ),

            "detected": (
                detected_count
            ),

            "mean_localization_error_sec": (
                mean_error
            ),

            "median_localization_error_sec": (
                median_error
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
        },

        "results": (
            full_results
        ),
    }

    JSON_REPORT_FILE.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("TEMPORAL LOCALIZATION SUMMARY")
    print("=" * 72)

    print(
        results_df[
            [
                "video_id",
                "expected_event_sec",
                "predicted_peak_sec",
                "localization_error_sec",
                "peak_probability",
                "detected",
                "within_3_sec",
                "within_5_sec",
            ]
        ]
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Mean error:",
        round(
            mean_error,
            3,
        ),
        "sec",
    )

    print(
        "Median error:",
        round(
            median_error,
            3,
        ),
        "sec",
    )

    print(
        "Detected:",
        f"{detected_count}/"
        f"{len(results_df)}",
    )

    print(
        "Within ±3 sec:",
        f"{within_3}/"
        f"{len(results_df)}",
    )

    print(
        "Within ±5 sec:",
        f"{within_5}/"
        f"{len(results_df)}",
    )

    print(
        "Within ±10 sec:",
        f"{within_10}/"
        f"{len(results_df)}",
    )

    print()
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    print(
        CSV_REPORT_FILE
    )

    print(
        JSON_REPORT_FILE
    )

    print()
    print(
        "✅ Whole-video temporal scan complete."
    )

    print(
        "✅ Ground-truth timestamps were "
        "used only for evaluation."
    )


if __name__ == "__main__":
    main()