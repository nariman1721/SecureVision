from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.video.candidate_reranker import (
    build_candidate_table,
)

from app.video.temporal_localizer import (
    TemporalLocalizer,
)

from training.diagnose_temporal_signal import (
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

WINDOWS_FILE = (
    REPORTS_DIR
    / "negative_temporal_windows.csv"
)

CANDIDATES_FILE = (
    REPORTS_DIR
    / "negative_candidates.csv"
)

VIDEO_SUMMARY_FILE = (
    REPORTS_DIR
    / "negative_candidate_video_summary.csv"
)

JSON_FILE = (
    REPORTS_DIR
    / "negative_candidate_analysis.json"
)


# =========================================================
# PATH
# =========================================================

def resolve_video_path(
    value: str,
) -> Path:
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
            f"Video not found: {path}"
        )

    return path


# =========================================================
# NORMAL TRAIN VIDEOS
# =========================================================

def build_normal_cases(
    manifest: pd.DataFrame,
) -> list[dict]:
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

    if normal.empty:
        raise RuntimeError(
            "No negative_normal rows found."
        )

    cases = []

    for content_id, group in (
        normal.groupby(
            "content_id"
        )
    ):
        first = (
            group.iloc[0]
        )

        cases.append(
            {
                "content_id": str(
                    content_id
                ),

                "video_id": str(
                    first[
                        "canonical_video_id"
                    ]
                ),

                "video_path": (
                    resolve_video_path(
                        str(
                            first[
                                "video_path"
                            ]
                        )
                    )
                ),
            }
        )

    cases.sort(
        key=lambda item: (
            item[
                "video_id"
            ]
        )
    )

    return cases


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 8F-A "
        "Negative Candidate Analysis"
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
        build_normal_cases(
            manifest
        )
    )

    print()
    print("DATASET")
    print("-" * 72)

    print(
        "Normal training videos:",
        len(
            cases
        ),
    )

    print(
        "Expected normal training videos:",
        15,
    )

    print(
        "Development validation used:",
        False,
    )

    if len(cases) != 15:
        print()
        print(
            "WARNING: expected 15 "
            f"normal train videos, got {len(cases)}"
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

    print(
        "Device:",
        localizer.device,
    )

    print(
        "Classifier threshold:",
        localizer.threshold,
    )

    window_rows = []
    candidate_rows = []
    video_rows = []

    # =====================================================
    # FULL NORMAL VIDEO SCAN
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

        print()
        print("=" * 72)

        print(
            f"[{case_index}/{len(cases)}] "
            f"SCANNING {video_id}"
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

        timeline_rows = []

        for item in (
            result[
                "timeline"
            ]
        ):
            row = {
                "content_id": (
                    case[
                        "content_id"
                    ]
                ),

                "video_id": (
                    video_id
                ),

                "window_index": int(
                    item[
                        "window_index"
                    ]
                ),

                "start_sec": float(
                    item[
                        "start_sec"
                    ]
                ),

                "end_sec": float(
                    item[
                        "end_sec"
                    ]
                ),

                "center_sec": float(
                    item[
                        "center_sec"
                    ]
                ),

                "raw_probability": float(
                    item[
                        "probability"
                    ]
                ),

                "smoothed_probability": float(
                    item[
                        "smoothed_probability"
                    ]
                ),

                "normalized_score": float(
                    item[
                        "normalized_score"
                    ]
                ),

                "support_score": float(
                    item[
                        "support_score"
                    ]
                ),

                "persistence": float(
                    item[
                        "persistence"
                    ]
                ),

                "candidate_score": float(
                    item[
                        "candidate_score"
                    ]
                ),
            }

            window_rows.append(
                row
            )

            timeline_rows.append(
                row
            )

        timeline_df = pd.DataFrame(
            timeline_rows
        )

        candidates = (
            build_candidate_table(
                timeline_df,
                scan_stride_sec=2.0,
                support_radius=1,
            )
        )

        if not candidates.empty:
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

            candidates[
                "source_class"
            ] = "normal"

            candidate_rows.append(
                candidates
            )

            top_candidate = (
                candidates
                .sort_values(
                    "region_score",
                    ascending=False,
                )
                .iloc[0]
            )

            top_region_score = float(
                top_candidate[
                    "region_score"
                ]
            )

            top_probability = float(
                top_candidate[
                    "peak_probability"
                ]
            )

            top_prominence = float(
                top_candidate[
                    "peak_prominence"
                ]
            )

            top_timecode = float(
                top_candidate[
                    "timecode_sec"
                ]
            )

        else:
            top_region_score = None
            top_probability = None
            top_prominence = None
            top_timecode = None

        max_window_probability = float(
            timeline_df[
                "raw_probability"
            ].max()
        )

        mean_window_probability = float(
            timeline_df[
                "raw_probability"
            ].mean()
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
            "Legacy detected:",
            result[
                "detected"
            ],
        )

        print(
            "Candidates:",
            len(
                candidates
            ),
        )

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
            "Max window probability:",
            round(
                max_window_probability,
                4,
            ),
        )

        if top_region_score is not None:
            print(
                "Top candidate time:",
                round(
                    top_timecode,
                    3,
                ),
            )

            print(
                "Top candidate probability:",
                round(
                    top_probability,
                    4,
                ),
            )

            print(
                "Top candidate region score:",
                round(
                    top_region_score,
                    4,
                ),
            )

            print(
                "Top candidate prominence:",
                round(
                    top_prominence,
                    4,
                ),
            )

        video_rows.append(
            {
                "content_id": (
                    case[
                        "content_id"
                    ]
                ),

                "video_id": (
                    video_id
                ),

                "duration_sec": float(
                    result[
                        "duration_sec"
                    ]
                ),

                "windows_scanned": int(
                    result[
                        "windows_scanned"
                    ]
                ),

                "legacy_detected": bool(
                    result[
                        "detected"
                    ]
                ),

                "baseline_probability": float(
                    result[
                        "baseline_probability"
                    ]
                ),

                "max_window_probability": (
                    max_window_probability
                ),

                "mean_window_probability": (
                    mean_window_probability
                ),

                "candidate_count": int(
                    len(
                        candidates
                    )
                ),

                "top_candidate_time_sec": (
                    top_timecode
                ),

                "top_candidate_region_score": (
                    top_region_score
                ),

                "top_candidate_probability": (
                    top_probability
                ),

                "top_candidate_prominence": (
                    top_prominence
                ),
            }
        )

    # =====================================================
    # SAVE
    # =====================================================

    windows_df = pd.DataFrame(
        window_rows
    )

    videos_df = pd.DataFrame(
        video_rows
    )

    if candidate_rows:
        candidates_df = pd.concat(
            candidate_rows,
            ignore_index=True,
        )
    else:
        candidates_df = pd.DataFrame()

    windows_df.to_csv(
        WINDOWS_FILE,
        index=False,
    )

    candidates_df.to_csv(
        CANDIDATES_FILE,
        index=False,
    )

    videos_df.to_csv(
        VIDEO_SUMMARY_FILE,
        index=False,
    )

    # =====================================================
    # SUMMARY
    # =====================================================

    detected_count = int(
        videos_df[
            "legacy_detected"
        ].sum()
    )

    total_candidates = int(
        len(
            candidates_df
        )
    )

    videos_with_candidates = int(
        (
            videos_df[
                "candidate_count"
            ]
            > 0
        ).sum()
    )

    report = {
        "configuration": {
            "dataset": (
                "normal training videos only"
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

            "classifier_threshold": float(
                localizer.threshold
            ),
        },

        "dataset": {
            "normal_videos": int(
                len(
                    cases
                )
            ),

            "scan_windows": int(
                len(
                    windows_df
                )
            ),

            "candidate_regions": (
                total_candidates
            ),

            "videos_with_candidates": (
                videos_with_candidates
            ),
        },

        "legacy_detection": {
            "false_positive_videos": (
                detected_count
            ),

            "false_positive_rate": float(
                detected_count
                / len(
                    videos_df
                )
            ),
        },

        "window_probability": {
            "mean": float(
                windows_df[
                    "raw_probability"
                ].mean()
            ),

            "median": float(
                windows_df[
                    "raw_probability"
                ].median()
            ),

            "max": float(
                windows_df[
                    "raw_probability"
                ].max()
            ),
        },
    }

    if not candidates_df.empty:
        report[
            "candidate_statistics"
        ] = {
            "region_score_mean": float(
                candidates_df[
                    "region_score"
                ].mean()
            ),

            "region_score_median": float(
                candidates_df[
                    "region_score"
                ].median()
            ),

            "region_score_max": float(
                candidates_df[
                    "region_score"
                ].max()
            ),

            "peak_probability_mean": float(
                candidates_df[
                    "peak_probability"
                ].mean()
            ),

            "peak_probability_max": float(
                candidates_df[
                    "peak_probability"
                ].max()
            ),

            "peak_prominence_mean": float(
                candidates_df[
                    "peak_prominence"
                ].mean()
            ),

            "peak_prominence_max": float(
                candidates_df[
                    "peak_prominence"
                ].max()
            ),
        }

    JSON_FILE.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("NEGATIVE VIDEO SUMMARY")
    print("=" * 72)

    print(
        "Normal videos:",
        len(
            videos_df
        ),
    )

    print(
        "Scan windows:",
        len(
            windows_df
        ),
    )

    print(
        "Candidate regions:",
        total_candidates,
    )

    print(
        "Videos with candidates:",
        videos_with_candidates,
    )

    print(
        "Legacy false positives:",
        f"{detected_count}/"
        f"{len(videos_df)}",
    )

    print(
        "Legacy false-positive rate:",
        round(
            detected_count
            / len(
                videos_df
            ),
            4,
        ),
    )

    print()
    print("=" * 72)
    print("PER VIDEO")
    print("=" * 72)

    print(
        videos_df[
            [
                "video_id",
                "windows_scanned",
                "legacy_detected",
                "baseline_probability",
                "max_window_probability",
                "candidate_count",
                "top_candidate_region_score",
                "top_candidate_probability",
                "top_candidate_prominence",
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
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    print(
        WINDOWS_FILE
    )

    print(
        CANDIDATES_FILE
    )

    print(
        VIDEO_SUMMARY_FILE
    )

    print(
        JSON_FILE
    )

    print()
    print(
        "✅ Normal training videos scanned."
    )

    print(
        "✅ Negative candidates collected."
    )


if __name__ == "__main__":
    main()