from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from app.video.temporal_localizer import (
    TemporalLocalizer,
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

MODEL_FILE = (
    AI_ROOT
    / "models"
    / "securevision_hardneg_classifier.joblib"
)

REPORTS_DIR = (
    AI_ROOT
    / "reports"
)

VIDEO_RESULTS_FILE = (
    REPORTS_DIR
    / "hardneg_fullvideo_results.csv"
)

EVENT_RESULTS_FILE = (
    REPORTS_DIR
    / "hardneg_event_results.csv"
)

SUMMARY_FILE = (
    REPORTS_DIR
    / "hardneg_fullvideo_evaluation.json"
)

OLD_NEGATIVE_REPORT = (
    REPORTS_DIR
    / "negative_candidate_analysis.json"
)

OLD_POSITIVE_REPORT = (
    REPORTS_DIR
    / "temporal_signal_diagnostics.json"
)


def resolve_video_path(
    value,
):
    path = Path(
        str(value)
    )

    if not path.is_absolute():
        path = (
            AI_ROOT
            / path
        )

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Missing video: {path}"
        )

    return path


def build_normal_cases(
    manifest: pd.DataFrame,
):
    normal = (
        manifest[
            manifest[
                "clip_type"
            ]
            .astype(str)
            .eq(
                "negative_normal"
            )
        ]
        .copy()
    )

    cases = []

    for content_id, group in (
        normal.groupby(
            "content_id"
        )
    ):
        row = (
            group.iloc[0]
        )

        cases.append(
            {
                "content_id": str(
                    content_id
                ),

                "video_id": str(
                    row[
                        "canonical_video_id"
                    ]
                ),

                "video_path": str(
                    resolve_video_path(
                        row[
                            "video_path"
                        ]
                    )
                ),
            }
        )

    return sorted(
        cases,
        key=lambda item: (
            item[
                "video_id"
            ]
        ),
    )


def summarize_event_errors(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "events": int(
            len(
                values
            )
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
                values <= 3
            ).sum()
        ),

        "within_5_sec": int(
            (
                values <= 5
            ).sum()
        ),

        "within_10_sec": int(
            (
                values <= 10
            ).sum()
        ),
    }


def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 8G-C "
        "Hard Negative Full-Video Evaluation"
    )
    print("=" * 72)

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Missing model:\n"
            f"{MODEL_FILE}"
        )

    manifest = (
        load_train_manifest()
    )

    positive_cases = (
        build_positive_cases(
            manifest
        )
    )

    normal_cases = (
        build_normal_cases(
            manifest
        )
    )

    print()
    print("DATASET")
    print("-" * 72)

    print(
        "Positive videos:",
        len(
            positive_cases
        ),
    )

    print(
        "Positive events:",
        sum(
            len(
                case[
                    "event_times"
                ]
            )
            for case
            in positive_cases
        ),
    )

    print(
        "Normal videos:",
        len(
            normal_cases
        ),
    )

    print(
        "Development validation used:",
        False,
    )

    print()
    print("LOADING HARD-NEGATIVE MODEL")
    print("-" * 72)

    localizer = (
        TemporalLocalizer(
            classifier_path=(
                MODEL_FILE
            ),
            scan_window_sec=6.0,
            scan_stride_sec=2.0,
            smoothing_radius=1,
            support_radius=1,
            top_candidates=5,
        )
    )

    print(
        "Device:",
        localizer.device,
    )

    print(
        "Threshold:",
        localizer.threshold,
    )

    video_rows = []
    event_rows = []

    # =====================================================
    # POSITIVE VIDEOS
    # =====================================================

    print()
    print("=" * 72)
    print("POSITIVE TRAIN VIDEOS")
    print("=" * 72)

    for index, case in enumerate(
        positive_cases,
        start=1,
    ):
        video_id = (
            case[
                "video_id"
            ]
        )

        print()
        print(
            f"[P {index}/{len(positive_cases)}] "
            f"{video_id}"
        )

        result = (
            localizer.localize(
                case[
                    "video_path"
                ],
                show_progress=True,
            )
        )

        candidates = (
            result[
                "top_candidates"
            ]
        )

        top1 = float(
            candidates[0][
                "timecode_sec"
            ]
        )

        video_rows.append(
            {
                "video_id": (
                    video_id
                ),

                "class": (
                    "violation"
                ),

                "detected": bool(
                    result[
                        "detected"
                    ]
                ),

                "baseline_probability": float(
                    result[
                        "baseline_probability"
                    ]
                ),

                "legacy_peak_probability": float(
                    result[
                        "peak"
                    ][
                        "probability"
                    ]
                ),

                "top_candidate_probability": float(
                    candidates[0][
                        "peak_probability"
                    ]
                ),

                "top_candidate_time_sec": (
                    top1
                ),

                "candidate_count": int(
                    len(
                        candidates
                    )
                ),
            }
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

            top1_error = abs(
                top1
                - event_time
            )

            nearest_top3 = min(
                candidates[:3],
                key=lambda item: abs(
                    float(
                        item[
                            "timecode_sec"
                        ]
                    )
                    - event_time
                ),
            )

            top3_error = abs(
                float(
                    nearest_top3[
                        "timecode_sec"
                    ]
                )
                - event_time
            )

            near_windows = [
                item
                for item
                in result[
                    "timeline"
                ]
                if abs(
                    float(
                        item[
                            "center_sec"
                        ]
                    )
                    - event_time
                )
                <= 3.0
            ]

            if near_windows:
                best_near_probability = max(
                    float(
                        item[
                            "probability"
                        ]
                    )
                    for item
                    in near_windows
                )
            else:
                best_near_probability = 0.0

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

                    "top1_sec": (
                        top1
                    ),

                    "top1_error_sec": (
                        top1_error
                    ),

                    "nearest_top3_sec": float(
                        nearest_top3[
                            "timecode_sec"
                        ]
                    ),

                    "top3_error_sec": (
                        top3_error
                    ),

                    "best_near_event_probability": (
                        best_near_probability
                    ),

                    "event_region_above_threshold": bool(
                        best_near_probability
                        >= localizer.threshold
                    ),
                }
            )

    # =====================================================
    # NORMAL VIDEOS
    # =====================================================

    print()
    print("=" * 72)
    print("NORMAL TRAIN VIDEOS")
    print("=" * 72)

    for index, case in enumerate(
        normal_cases,
        start=1,
    ):
        video_id = (
            case[
                "video_id"
            ]
        )

        print()
        print(
            f"[N {index}/{len(normal_cases)}] "
            f"{video_id}"
        )

        result = (
            localizer.localize(
                case[
                    "video_path"
                ],
                show_progress=True,
            )
        )

        candidates = (
            result[
                "top_candidates"
            ]
        )

        video_rows.append(
            {
                "video_id": (
                    video_id
                ),

                "class": (
                    "normal"
                ),

                "detected": bool(
                    result[
                        "detected"
                    ]
                ),

                "baseline_probability": float(
                    result[
                        "baseline_probability"
                    ]
                ),

                "legacy_peak_probability": float(
                    result[
                        "peak"
                    ][
                        "probability"
                    ]
                ),

                "top_candidate_probability": float(
                    candidates[0][
                        "peak_probability"
                    ]
                ),

                "top_candidate_time_sec": float(
                    candidates[0][
                        "timecode_sec"
                    ]
                ),

                "candidate_count": int(
                    len(
                        candidates
                    )
                ),
            }
        )

    videos = pd.DataFrame(
        video_rows
    )

    events = pd.DataFrame(
        event_rows
    )

    videos.to_csv(
        VIDEO_RESULTS_FILE,
        index=False,
    )

    events.to_csv(
        EVENT_RESULTS_FILE,
        index=False,
    )

    positive_video_rows = (
        videos[
            videos[
                "class"
            ]
            == "violation"
        ]
    )

    normal_video_rows = (
        videos[
            videos[
                "class"
            ]
            == "normal"
        ]
    )

    positive_detected = int(
        positive_video_rows[
            "detected"
        ].sum()
    )

    normal_false_positive = int(
        normal_video_rows[
            "detected"
        ].sum()
    )

    event_above_threshold = int(
        events[
            "event_region_above_threshold"
        ].sum()
    )

    top1_summary = (
        summarize_event_errors(
            events[
                "top1_error_sec"
            ]
        )
    )

    top3_summary = (
        summarize_event_errors(
            events[
                "top3_error_sec"
            ]
        )
    )

    report = {
        "configuration": {
            "classifier": str(
                MODEL_FILE
            ),

            "threshold": float(
                localizer.threshold
            ),

            "development_validation_used": (
                False
            ),
        },

        "dataset": {
            "positive_videos": int(
                len(
                    positive_video_rows
                )
            ),

            "events": int(
                len(
                    events
                )
            ),

            "normal_videos": int(
                len(
                    normal_video_rows
                )
            ),
        },

        "positive_detection": {
            "detected_videos": (
                positive_detected
            ),

            "total_videos": int(
                len(
                    positive_video_rows
                )
            ),

            "video_recall": float(
                positive_detected
                / len(
                    positive_video_rows
                )
            ),

            "event_regions_above_threshold": (
                event_above_threshold
            ),

            "event_recall": float(
                event_above_threshold
                / len(
                    events
                )
            ),
        },

        "normal_detection": {
            "false_positive_videos": (
                normal_false_positive
            ),

            "total_videos": int(
                len(
                    normal_video_rows
                )
            ),

            "false_positive_rate": float(
                normal_false_positive
                / len(
                    normal_video_rows
                )
            ),

            "specificity": float(
                1.0
                - (
                    normal_false_positive
                    / len(
                        normal_video_rows
                    )
                )
            ),
        },

        "localization": {
            "top1": (
                top1_summary
            ),

            "top3": (
                top3_summary
            ),
        },
    }

    # Previous train-only reference results.
    if OLD_NEGATIVE_REPORT.exists():
        old = json.loads(
            OLD_NEGATIVE_REPORT.read_text(
                encoding="utf-8"
            )
        )

        report[
            "comparison_old_normal"
        ] = {
            "old_false_positive_rate": float(
                old[
                    "legacy_detection"
                ][
                    "false_positive_rate"
                ]
            ),

            "new_false_positive_rate": float(
                report[
                    "normal_detection"
                ][
                    "false_positive_rate"
                ]
            ),
        }

    if OLD_POSITIVE_REPORT.exists():
        old = json.loads(
            OLD_POSITIVE_REPORT.read_text(
                encoding="utf-8"
            )
        )

        report[
            "comparison_old_positive"
        ] = {
            "old_event_region_above_threshold": int(
                old[
                    "event_level"
                ][
                    "event_region_above_threshold"
                ]
            ),

            "new_event_region_above_threshold": (
                event_above_threshold
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

    print()
    print("=" * 72)
    print("FINAL HARD-NEGATIVE EVALUATION")
    print("=" * 72)

    print()
    print("POSITIVE DETECTION")
    print("-" * 72)

    print(
        "Video recall:",
        f"{positive_detected}/"
        f"{len(positive_video_rows)}",
    )

    print(
        "Event regions above threshold:",
        f"{event_above_threshold}/"
        f"{len(events)}",
    )

    print()
    print("NORMAL DETECTION")
    print("-" * 72)

    print(
        "False-positive videos:",
        f"{normal_false_positive}/"
        f"{len(normal_video_rows)}",
    )

    print(
        "False-positive rate:",
        round(
            report[
                "normal_detection"
            ][
                "false_positive_rate"
            ],
            4,
        ),
    )

    print(
        "Specificity:",
        round(
            report[
                "normal_detection"
            ][
                "specificity"
            ],
            4,
        ),
    )

    print()
    print("TOP-1 LOCALIZATION")
    print("-" * 72)

    print(
        json.dumps(
            top1_summary,
            indent=2,
        )
    )

    print()
    print("TOP-3 LOCALIZATION")
    print("-" * 72)

    print(
        json.dumps(
            top3_summary,
            indent=2,
        )
    )

    print()
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    print(
        VIDEO_RESULTS_FILE
    )

    print(
        EVENT_RESULTS_FILE
    )

    print(
        SUMMARY_FILE
    )

    print()
    print(
        "✅ Hard-negative full-video evaluation complete."
    )

    print(
        "✅ Development validation was not used."
    )


if __name__ == "__main__":
    main()
