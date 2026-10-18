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

REPORTS_DIR = (
    AI_ROOT
    / "reports"
)

CSV_FILE = (
    REPORTS_DIR
    / "temporal_scoring_comparison.csv"
)

JSON_FILE = (
    REPORTS_DIR
    / "temporal_scoring_comparison.json"
)


# =========================================================
# METRICS
# =========================================================

def summarize_errors(
    errors: list[float],
) -> dict:
    values = np.asarray(
        errors,
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
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 8C "
        "Temporal Scoring Comparison"
    )
    print("=" * 72)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = (
        load_train_manifest()
    )

    cases = (
        build_positive_cases(
            manifest
        )
    )

    total_events = int(
        sum(
            len(
                case[
                    "event_times"
                ]
            )
            for case in cases
        )
    )

    print()
    print("DATASET")
    print("-" * 72)

    print(
        "Positive train videos:",
        len(
            cases
        ),
    )

    print(
        "Events:",
        total_events,
    )

    print(
        "Development validation used:",
        False,
    )

    print()
    print("LOADING LOCALIZER")
    print("-" * 72)

    localizer = (
        TemporalLocalizer(
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

    rows = []

    # =====================================================
    # SCAN
    # =====================================================

    for case_index, case in enumerate(
        cases,
        start=1,
    ):
        video_id = (
            case[
                "video_id"
            ]
        )

        event_times = [
            float(value)
            for value
            in case[
                "event_times"
            ]
        ]

        print()
        print("=" * 72)
        print(
            f"[{case_index}/{len(cases)}] "
            f"{video_id}"
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

        legacy_peak = float(
            result[
                "peak"
            ][
                "timecode_sec"
            ]
        )

        improved_peak = float(
            result[
                "improved_peak"
            ][
                "timecode_sec"
            ]
        )

        candidates = (
            result[
                "top_candidates"
            ]
        )

        print()
        print(
            "Baseline:",
            round(
                result[
                    "baseline_probability"
                ],
                4,
            ),
        )

        print(
            "Legacy peak:",
            round(
                legacy_peak,
                3,
            ),
        )

        print(
            "Improved peak:",
            round(
                improved_peak,
                3,
            ),
        )

        print()
        print("TOP CANDIDATES")

        for candidate in candidates:
            print(
                f"  #{candidate['rank']} "
                f"time={candidate['timecode_sec']:.3f}s "
                f"score={candidate['score']:.4f} "
                f"prob={candidate['peak_probability']:.4f} "
                f"windows={candidate['windows_in_region']}"
            )

        # =================================================
        # EVENT COMPARISON
        # =================================================

        for event_index, event_time in enumerate(
            event_times,
            start=1,
        ):
            legacy_error = abs(
                legacy_peak
                - event_time
            )

            improved_top1_error = abs(
                improved_peak
                - event_time
            )

            top3 = (
                candidates[:3]
            )

            nearest_top3 = min(
                top3,
                key=lambda candidate: abs(
                    float(
                        candidate[
                            "timecode_sec"
                        ]
                    )
                    - event_time
                ),
            )

            improved_top3_error = abs(
                float(
                    nearest_top3[
                        "timecode_sec"
                    ]
                )
                - event_time
            )

            all_candidates = (
                candidates
            )

            nearest_candidate = min(
                all_candidates,
                key=lambda candidate: abs(
                    float(
                        candidate[
                            "timecode_sec"
                        ]
                    )
                    - event_time
                ),
            )

            best_candidate_error = abs(
                float(
                    nearest_candidate[
                        "timecode_sec"
                    ]
                )
                - event_time
            )

            row = {
                "video_id": (
                    video_id
                ),

                "event_index": int(
                    event_index
                ),

                "event_time_sec": float(
                    event_time
                ),

                "legacy_peak_sec": (
                    legacy_peak
                ),

                "legacy_error_sec": (
                    legacy_error
                ),

                "improved_peak_sec": (
                    improved_peak
                ),

                "improved_top1_error_sec": (
                    improved_top1_error
                ),

                "nearest_top3_sec": float(
                    nearest_top3[
                        "timecode_sec"
                    ]
                ),

                "nearest_top3_rank": int(
                    nearest_top3[
                        "rank"
                    ]
                ),

                "improved_top3_error_sec": (
                    improved_top3_error
                ),

                "nearest_candidate_sec": float(
                    nearest_candidate[
                        "timecode_sec"
                    ]
                ),

                "nearest_candidate_rank": int(
                    nearest_candidate[
                        "rank"
                    ]
                ),

                "best_candidate_error_sec": (
                    best_candidate_error
                ),

                "baseline_probability": float(
                    result[
                        "baseline_probability"
                    ]
                ),

                "top_candidate_score": float(
                    candidates[0][
                        "score"
                    ]
                ),

                "top_candidate_probability": float(
                    candidates[0][
                        "peak_probability"
                    ]
                ),
            }

            rows.append(
                row
            )

            print()
            print(
                f"Event {event_index}: "
                f"{event_time:.3f}s"
            )

            print(
                "  legacy error:",
                round(
                    legacy_error,
                    3,
                ),
            )

            print(
                "  improved top1 error:",
                round(
                    improved_top1_error,
                    3,
                ),
            )

            print(
                "  improved top3 error:",
                round(
                    improved_top3_error,
                    3,
                ),
            )

    # =====================================================
    # SUMMARY
    # =====================================================

    dataframe = pd.DataFrame(
        rows
    )

    dataframe.to_csv(
        CSV_FILE,
        index=False,
    )

    legacy_summary = (
        summarize_errors(
            dataframe[
                "legacy_error_sec"
            ].tolist()
        )
    )

    improved_top1_summary = (
        summarize_errors(
            dataframe[
                "improved_top1_error_sec"
            ].tolist()
        )
    )

    improved_top3_summary = (
        summarize_errors(
            dataframe[
                "improved_top3_error_sec"
            ].tolist()
        )
    )

    best_candidate_summary = (
        summarize_errors(
            dataframe[
                "best_candidate_error_sec"
            ].tolist()
        )
    )

    improved_top1_better = int(
        (
            dataframe[
                "improved_top1_error_sec"
            ]
            <
            dataframe[
                "legacy_error_sec"
            ]
        ).sum()
    )

    improved_top1_equal = int(
        np.isclose(
            dataframe[
                "improved_top1_error_sec"
            ],
            dataframe[
                "legacy_error_sec"
            ],
        ).sum()
    )

    improved_top1_worse = int(
        (
            dataframe[
                "improved_top1_error_sec"
            ]
            >
            dataframe[
                "legacy_error_sec"
            ]
        ).sum()
    )

    report = {
        "configuration": {
            "dataset": (
                "positive training videos"
            ),

            "development_validation_used": (
                False
            ),

            "scan_window_sec": (
                6.0
            ),

            "scan_stride_sec": (
                2.0
            ),

            "support_radius": (
                1
            ),

            "top_candidates": (
                5
            ),
        },

        "legacy": (
            legacy_summary
        ),

        "improved_top1": (
            improved_top1_summary
        ),

        "improved_top3": (
            improved_top3_summary
        ),

        "best_of_saved_candidates": (
            best_candidate_summary
        ),

        "top1_vs_legacy": {
            "better": (
                improved_top1_better
            ),

            "equal": (
                improved_top1_equal
            ),

            "worse": (
                improved_top1_worse
            ),
        },
    }

    JSON_FILE.write_text(
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
    print("FINAL COMPARISON")
    print("=" * 72)

    print()
    print("LEGACY GLOBAL MAX")
    print("-" * 72)

    for key, value in (
        legacy_summary.items()
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
    print("IMPROVED TOP-1")
    print("-" * 72)

    for key, value in (
        improved_top1_summary.items()
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
    print("IMPROVED TOP-3")
    print("-" * 72)

    for key, value in (
        improved_top3_summary.items()
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
    print("TOP-1 VS LEGACY")
    print("-" * 72)

    print(
        "Better:",
        improved_top1_better,
    )

    print(
        "Equal:",
        improved_top1_equal,
    )

    print(
        "Worse:",
        improved_top1_worse,
    )

    print()
    print("=" * 72)
    print("EVENT TABLE")
    print("=" * 72)

    print(
        dataframe[
            [
                "video_id",
                "event_time_sec",
                "legacy_peak_sec",
                "legacy_error_sec",
                "improved_peak_sec",
                "improved_top1_error_sec",
                "nearest_top3_sec",
                "nearest_top3_rank",
                "improved_top3_error_sec",
            ]
        ]
        .round(
            3
        )
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    print(
        CSV_FILE
    )

    print(
        JSON_FILE
    )

    print()
    print(
        "✅ Temporal scoring comparison complete."
    )


if __name__ == "__main__":
    main()