from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.video.candidate_reranker import (
    build_candidate_table,
)

from app.video.fine_temporal_refiner import (
    FineTemporalRefiner,
    robust_baseline_and_scale,
)

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

CACHED_WINDOWS_FILE = (
    REPORTS_DIR
    / "temporal_signal_windows.csv"
)

EVENTS_FILE = (
    REPORTS_DIR
    / "fine_temporal_refinement_events.csv"
)

CANDIDATES_FILE = (
    REPORTS_DIR
    / "fine_temporal_refinement_candidates.csv"
)

SUMMARY_FILE = (
    REPORTS_DIR
    / "fine_temporal_refinement_summary.json"
)


# =========================================================
# HELPERS
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

        "within_1_sec": int(
            (
                values <= 1.0
            ).sum()
        ),

        "within_3_sec": int(
            (
                values <= 3.0
            ).sum()
        ),

        "within_5_sec": int(
            (
                values <= 5.0
            ).sum()
        ),

        "within_10_sec": int(
            (
                values <= 10.0
            ).sum()
        ),
    }


def load_cached_windows():
    if not CACHED_WINDOWS_FILE.exists():
        raise FileNotFoundError(
            f"Missing Phase 8B cache:\n"
            f"{CACHED_WINDOWS_FILE}"
        )

    dataframe = pd.read_csv(
        CACHED_WINDOWS_FILE
    )

    required = [
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
        "SecureVision — Phase 8E "
        "Fine Temporal Refinement"
    )
    print("=" * 72)

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cached_windows = (
        load_cached_windows()
    )

    manifest = (
        load_train_manifest()
    )

    cases = (
        build_positive_cases(
            manifest
        )
    )

    print()
    print("CONFIGURATION")
    print("-" * 72)

    print(
        "Positive training videos:",
        len(
            cases
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
            for case in cases
        ),
    )

    print(
        "Coarse X3D scan reused from cache:",
        True,
    )

    print(
        "Fine X3D rescan:",
        True,
    )

    print(
        "Fine stride:",
        0.5,
        "sec",
    )

    print(
        "Search radius:",
        5.0,
        "sec",
    )

    print(
        "Top coarse candidates:",
        3,
    )

    print(
        "Development validation used:",
        False,
    )

    print()
    print("LOADING X3D + CLASSIFIER")
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

    refiner = (
        FineTemporalRefiner(
            localizer=(
                localizer
            ),
            fine_window_sec=6.0,
            fine_stride_sec=0.5,
            search_radius_sec=5.0,
            support_radius_windows=2,
        )
    )

    print(
        "Device:",
        localizer.device,
    )

    event_rows = []

    candidate_rows = []

    total_fine_windows = 0

    # =====================================================
    # VIDEOS
    # =====================================================

    for case_index, case in enumerate(
        cases,
        start=1,
    ):
        video_id = str(
            case[
                "video_id"
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

        coarse_timeline = (
            cached_windows[
                cached_windows[
                    "video_id"
                ]
                .astype(str)
                .eq(
                    video_id
                )
            ]
            .sort_values(
                "window_index"
            )
            .reset_index(
                drop=True
            )
        )

        if coarse_timeline.empty:
            raise RuntimeError(
                f"No cached timeline "
                f"for {video_id}"
            )

        duration_sec = float(
            max(
                coarse_timeline[
                    "end_sec"
                ]
            )
        )

        coarse_candidates = (
            build_candidate_table(
                coarse_timeline,
                scan_stride_sec=2.0,
                support_radius=1,
            )
        )

        coarse_candidates = (
            coarse_candidates
            .head(3)
            .reset_index(
                drop=True
            )
        )

        candidate_dicts = (
            coarse_candidates
            .to_dict(
                orient="records"
            )
        )

        raw_probabilities = (
            coarse_timeline[
                "raw_probability"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        (
            global_baseline,
            global_scale,
        ) = (
            robust_baseline_and_scale(
                raw_probabilities
            )
        )

        print()
        print("=" * 72)

        print(
            f"[{case_index}/{len(cases)}] "
            f"{video_id}"
        )

        print("=" * 72)

        print(
            "Events:",
            ", ".join(
                f"{value:.3f}s"
                for value
                in event_times
            ),
        )

        print(
            "Coarse candidates:",
            len(
                candidate_dicts
            ),
        )

        for index, candidate in enumerate(
            candidate_dicts,
            start=1,
        ):
            print(
                f"  coarse #{index}: "
                f"time="
                f"{candidate['timecode_sec']:.3f}s "
                f"score="
                f"{candidate['region_score']:.4f}"
            )

        refined = (
            refiner.refine_candidates(
                video_path=(
                    case[
                        "video_path"
                    ]
                ),
                duration_sec=(
                    duration_sec
                ),
                candidates=(
                    candidate_dicts
                ),
                global_baseline=(
                    global_baseline
                ),
                global_scale=(
                    global_scale
                ),
                top_k=3,
                show_progress=True,
            )
        )

        print()
        print("REFINED CANDIDATES")
        print("-" * 72)

        for item in refined:
            print(
                f"  refined "
                f"#{item['refined_rank']} "
                f"(coarse "
                f"#{item['original_rank']}): "
                f"time="
                f"{item['fine_peak_sec']:.3f}s "
                f"score="
                f"{item['fine_score']:.4f} "
                f"prob="
                f"{item['fine_peak_probability']:.4f}"
            )

            total_fine_windows += int(
                item[
                    "fine_windows_scanned"
                ]
            )

            candidate_rows.append(
                {
                    "video_id": (
                        video_id
                    ),
                    "original_rank": int(
                        item[
                            "original_rank"
                        ]
                    ),
                    "refined_rank": int(
                        item[
                            "refined_rank"
                        ]
                    ),
                    "coarse_timecode_sec": float(
                        item[
                            "coarse_timecode_sec"
                        ]
                    ),
                    "fine_peak_sec": float(
                        item[
                            "fine_peak_sec"
                        ]
                    ),
                    "fine_score": float(
                        item[
                            "fine_score"
                        ]
                    ),
                    "fine_peak_probability": float(
                        item[
                            "fine_peak_probability"
                        ]
                    ),
                    "fine_support_probability": float(
                        item[
                            "fine_support_probability"
                        ]
                    ),
                    "global_baseline": float(
                        item[
                            "global_baseline"
                        ]
                    ),
                    "local_baseline": float(
                        item[
                            "local_baseline"
                        ]
                    ),
                    "local_prominence": float(
                        item[
                            "local_prominence"
                        ]
                    ),
                    "edge_contrast": float(
                        item[
                            "edge_contrast"
                        ]
                    ),
                    "area_density": float(
                        item[
                            "area_density"
                        ]
                    ),
                    "fine_windows_scanned": int(
                        item[
                            "fine_windows_scanned"
                        ]
                    ),
                }
            )

        # =================================================
        # EVENT EVALUATION
        # =================================================

        coarse_top1_sec = float(
            candidate_dicts[0][
                "timecode_sec"
            ]
        )

        fine_top1_sec = float(
            refined[0][
                "fine_peak_sec"
            ]
        )

        fine_top3 = (
            refined[
                :3
            ]
        )

        for event_index, event_time in enumerate(
            event_times,
            start=1,
        ):
            coarse_error = abs(
                coarse_top1_sec
                - event_time
            )

            fine_top1_error = abs(
                fine_top1_sec
                - event_time
            )

            nearest_fine_top3 = min(
                fine_top3,
                key=lambda item: abs(
                    float(
                        item[
                            "fine_peak_sec"
                        ]
                    )
                    - event_time
                ),
            )

            fine_top3_error = abs(
                float(
                    nearest_fine_top3[
                        "fine_peak_sec"
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
                    "event_time_sec": float(
                        event_time
                    ),
                    "coarse_top1_sec": (
                        coarse_top1_sec
                    ),
                    "coarse_top1_error_sec": (
                        coarse_error
                    ),
                    "fine_top1_sec": (
                        fine_top1_sec
                    ),
                    "fine_top1_error_sec": (
                        fine_top1_error
                    ),
                    "nearest_fine_top3_sec": float(
                        nearest_fine_top3[
                            "fine_peak_sec"
                        ]
                    ),
                    "nearest_fine_top3_rank": int(
                        nearest_fine_top3[
                            "refined_rank"
                        ]
                    ),
                    "fine_top3_error_sec": (
                        fine_top3_error
                    ),
                }
            )

            print()
            print(
                f"Event {event_index}: "
                f"{event_time:.3f}s"
            )

            print(
                "  coarse Top-1 error:",
                round(
                    coarse_error,
                    3,
                ),
            )

            print(
                "  fine Top-1 error:",
                round(
                    fine_top1_error,
                    3,
                ),
            )

            print(
                "  fine Top-3 error:",
                round(
                    fine_top3_error,
                    3,
                ),
            )

    # =====================================================
    # REPORTS
    # =====================================================

    events = pd.DataFrame(
        event_rows
    )

    candidates = pd.DataFrame(
        candidate_rows
    )

    events.to_csv(
        EVENTS_FILE,
        index=False,
    )

    candidates.to_csv(
        CANDIDATES_FILE,
        index=False,
    )

    coarse_summary = (
        summarize_errors(
            events[
                "coarse_top1_error_sec"
            ]
        )
    )

    fine_top1_summary = (
        summarize_errors(
            events[
                "fine_top1_error_sec"
            ]
        )
    )

    fine_top3_summary = (
        summarize_errors(
            events[
                "fine_top3_error_sec"
            ]
        )
    )

    better = int(
        (
            events[
                "fine_top1_error_sec"
            ]
            <
            events[
                "coarse_top1_error_sec"
            ]
        ).sum()
    )

    equal = int(
        np.isclose(
            events[
                "fine_top1_error_sec"
            ],
            events[
                "coarse_top1_error_sec"
            ],
        ).sum()
    )

    worse = int(
        (
            events[
                "fine_top1_error_sec"
            ]
            >
            events[
                "coarse_top1_error_sec"
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
            "ground_truth_used_for_scoring": (
                False
            ),
            "coarse_scan_reused_from_cache": (
                True
            ),
            "fine_x3d_rescan": (
                True
            ),
            "fine_window_sec": (
                6.0
            ),
            "fine_stride_sec": (
                0.5
            ),
            "search_radius_sec": (
                5.0
            ),
            "top_coarse_candidates": (
                3
            ),
        },

        "dataset": {
            "videos": int(
                len(
                    cases
                )
            ),
            "events": int(
                len(
                    events
                )
            ),
            "refined_candidates": int(
                len(
                    candidates
                )
            ),
            "fine_windows_scanned": int(
                total_fine_windows
            ),
        },

        "coarse_top1": (
            coarse_summary
        ),

        "fine_top1": (
            fine_top1_summary
        ),

        "fine_top3": (
            fine_top3_summary
        ),

        "fine_top1_vs_coarse": {
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
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # =====================================================
    # CONSOLE SUMMARY
    # =====================================================

    print()
    print("=" * 72)
    print("COARSE TOP-1")
    print("=" * 72)

    for key, value in (
        coarse_summary.items()
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
    print("FINE TOP-1")
    print("=" * 72)

    for key, value in (
        fine_top1_summary.items()
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
    print("FINE TOP-3")
    print("=" * 72)

    for key, value in (
        fine_top3_summary.items()
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
    print("FINE TOP-1 VS COARSE")
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
        EVENTS_FILE
    )

    print(
        CANDIDATES_FILE
    )

    print(
        SUMMARY_FILE
    )

    print()
    print(
        "✅ Fine temporal refinement complete."
    )

    print(
        "✅ Coarse X3D results were reused."
    )

    print(
        "✅ X3D was run only for fine candidate regions."
    )

    print(
        "✅ Ground truth was used only for evaluation."
    )


if __name__ == "__main__":
    main()