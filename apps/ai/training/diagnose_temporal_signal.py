from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from app.video.temporal_localizer import (
    TemporalLocalizer,
)


# =========================================================
# PATHS
# =========================================================

AI_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

TRAIN_CLIPS_FILE = (
    AI_ROOT
    / "data"
    / "processed"
    / "train_clips.csv"
)

REPORTS_DIR = (
    AI_ROOT
    / "reports"
)

EVENT_REPORT_FILE = (
    REPORTS_DIR
    / "temporal_signal_events.csv"
)

WINDOW_REPORT_FILE = (
    REPORTS_DIR
    / "temporal_signal_windows.csv"
)

SUMMARY_FILE = (
    REPORTS_DIR
    / "temporal_signal_diagnostics.json"
)


# =========================================================
# TEMPORAL DEFINITIONS
# =========================================================

# Our scan window is 6 seconds.
#
# If its center is within ±3 seconds
# of the ground-truth event, the
# 6-second window contains the event.
NEAR_EVENT_RADIUS_SEC = 3.0

# Windows whose centers are at least
# 6 seconds from every event are treated
# as clean temporal background.
BACKGROUND_RADIUS_SEC = 6.0


# =========================================================
# MANIFEST
# =========================================================

def load_train_manifest() -> pd.DataFrame:
    if not TRAIN_CLIPS_FILE.exists():
        raise FileNotFoundError(
            f"Missing train manifest:\n"
            f"{TRAIN_CLIPS_FILE}"
        )

    dataframe = pd.read_csv(
        TRAIN_CLIPS_FILE
    )

    required = [
        "clip_id",
        "content_id",
        "canonical_video_id",
        "video_path",
        "label",
        "clip_type",
        "start_sec",
        "end_sec",
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise RuntimeError(
            "Train manifest missing "
            f"columns: {missing}"
        )

    return dataframe


# =========================================================
# EVENT TIME
# =========================================================

def resolve_event_time(
    row: pd.Series,
) -> float:
    """
    Prefer an explicit event timestamp if it exists.

    Otherwise positive_event clips were generated
    around the annotated event, so their midpoint
    is the event timestamp.
    """

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
                row[column]
            )
        ):
            return float(
                row[column]
            )

    return float(
        (
            float(
                row["start_sec"]
            )
            + float(
                row["end_sec"]
            )
        )
        / 2.0
    )


# =========================================================
# VIDEO PATH
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
# BUILD POSITIVE VIDEO TABLE
# =========================================================

def build_positive_cases(
    manifest: pd.DataFrame,
):
    positive_rows = (
        manifest[
            manifest[
                "clip_type"
            ]
            .astype(str)
            .eq(
                "positive_event"
            )
        ]
        .copy()
    )

    if positive_rows.empty:
        raise RuntimeError(
            "No positive_event clips found."
        )

    positive_rows[
        "event_time_sec"
    ] = positive_rows.apply(
        resolve_event_time,
        axis=1,
    )

    cases = []

    for content_id, group in (
        positive_rows.groupby(
            "content_id"
        )
    ):
        group = (
            group
            .sort_values(
                "event_time_sec"
            )
            .reset_index(
                drop=True
            )
        )

        first = (
            group.iloc[0]
        )

        event_times = (
            group[
                "event_time_sec"
            ]
            .astype(float)
            .tolist()
        )

        cases.append(
            {
                "content_id": (
                    str(
                        content_id
                    )
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

                "event_times": (
                    event_times
                ),
            }
        )

    return cases


# =========================================================
# WINDOW LABELS
# =========================================================

def minimum_event_distance(
    center_sec: float,
    event_times: list[float],
) -> float:
    return float(
        min(
            abs(
                float(center_sec)
                - float(event_time)
            )
            for event_time
            in event_times
        )
    )


def temporal_region(
    distance_sec: float,
) -> str:
    if (
        distance_sec
        <= NEAR_EVENT_RADIUS_SEC
    ):
        return "near_event"

    if (
        distance_sec
        >= BACKGROUND_RADIUS_SEC
    ):
        return "background"

    return "ambiguous"


# =========================================================
# EVENT METRICS
# =========================================================

def analyze_event(
    *,
    event_index: int,
    event_time_sec: float,
    timeline: list[dict],
    all_event_times: list[float],
    global_peak_sec: float,
):
    near_event_windows = [
        item
        for item in timeline
        if abs(
            float(
                item[
                    "center_sec"
                ]
            )
            - event_time_sec
        )
        <= NEAR_EVENT_RADIUS_SEC
    ]

    if not near_event_windows:
        raise RuntimeError(
            "No temporal scan window "
            "covers event."
        )

    background_windows = [
        item
        for item in timeline
        if minimum_event_distance(
            float(
                item[
                    "center_sec"
                ]
            ),
            all_event_times,
        )
        >= BACKGROUND_RADIUS_SEC
    ]

    best_event_window = max(
        near_event_windows,
        key=lambda item: (
            item[
                "smoothed_probability"
            ]
        ),
    )

    best_event_raw = max(
        float(
            item[
                "probability"
            ]
        )
        for item in near_event_windows
    )

    best_event_smooth = float(
        best_event_window[
            "smoothed_probability"
        ]
    )

    # Rank windows from highest suspiciousness
    # to lowest.
    ranked = sorted(
        timeline,
        key=lambda item: (
            item[
                "smoothed_probability"
            ]
        ),
        reverse=True,
    )

    best_event_window_rank = next(
        index
        for index, item
        in enumerate(
            ranked,
            start=1,
        )
        if item is best_event_window
    )

    if background_windows:
        best_background_window = max(
            background_windows,
            key=lambda item: (
                item[
                    "smoothed_probability"
                ]
            ),
        )

        background_max_raw = float(
            max(
                item[
                    "probability"
                ]
                for item
                in background_windows
            )
        )

        background_max_smooth = float(
            best_background_window[
                "smoothed_probability"
            ]
        )

        background_mean_smooth = float(
            np.mean(
                [
                    item[
                        "smoothed_probability"
                    ]
                    for item
                    in background_windows
                ]
            )
        )

        temporal_margin = float(
            best_event_smooth
            - background_max_smooth
        )

    else:
        background_max_raw = None
        background_max_smooth = None
        background_mean_smooth = None
        temporal_margin = None

    peak_error_sec = abs(
        float(
            global_peak_sec
        )
        - float(
            event_time_sec
        )
    )

    event_region_detected = bool(
        best_event_smooth
        >= 0.5
    )

    return {
        "event_index": int(
            event_index
        ),

        "event_time_sec": float(
            event_time_sec
        ),

        "global_peak_sec": float(
            global_peak_sec
        ),

        "peak_error_sec": float(
            peak_error_sec
        ),

        "best_event_window_center_sec": float(
            best_event_window[
                "center_sec"
            ]
        ),

        "best_event_window_raw_probability": float(
            best_event_raw
        ),

        "best_event_window_probability": float(
            best_event_smooth
        ),

        "best_event_window_rank": int(
            best_event_window_rank
        ),

        "background_max_raw_probability": (
            background_max_raw
        ),

        "background_max_probability": (
            background_max_smooth
        ),

        "background_mean_probability": (
            background_mean_smooth
        ),

        "temporal_margin": (
            temporal_margin
        ),

        "event_region_above_threshold": (
            event_region_detected
        ),

        "peak_within_3_sec": bool(
            peak_error_sec
            <= 3.0
        ),

        "peak_within_5_sec": bool(
            peak_error_sec
            <= 5.0
        ),

        "peak_within_10_sec": bool(
            peak_error_sec
            <= 10.0
        ),
    }


# =========================================================
# GLOBAL WINDOW METRICS
# =========================================================

def calculate_window_metrics(
    windows_dataframe: pd.DataFrame,
):
    diagnostic = windows_dataframe[
        windows_dataframe[
            "region"
        ].isin(
            [
                "near_event",
                "background",
            ]
        )
    ].copy()

    if diagnostic.empty:
        return {
            "windows": 0,
            "near_event_windows": 0,
            "background_windows": 0,
            "raw_roc_auc": None,
            "raw_pr_auc": None,
            "smoothed_roc_auc": None,
            "smoothed_pr_auc": None,
        }

    diagnostic[
        "target"
    ] = (
        diagnostic[
            "region"
        ]
        .eq(
            "near_event"
        )
        .astype(int)
    )

    y_true = (
        diagnostic[
            "target"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    raw = (
        diagnostic[
            "raw_probability"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    smooth = (
        diagnostic[
            "smoothed_probability"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    if len(
        np.unique(
            y_true
        )
    ) < 2:
        raw_roc = None
        raw_pr = None
        smooth_roc = None
        smooth_pr = None

    else:
        raw_roc = float(
            roc_auc_score(
                y_true,
                raw,
            )
        )

        raw_pr = float(
            average_precision_score(
                y_true,
                raw,
            )
        )

        smooth_roc = float(
            roc_auc_score(
                y_true,
                smooth,
            )
        )

        smooth_pr = float(
            average_precision_score(
                y_true,
                smooth,
            )
        )

    return {
        "windows": int(
            len(
                diagnostic
            )
        ),

        "near_event_windows": int(
            diagnostic[
                "region"
            ]
            .eq(
                "near_event"
            )
            .sum()
        ),

        "background_windows": int(
            diagnostic[
                "region"
            ]
            .eq(
                "background"
            )
            .sum()
        ),

        "raw_roc_auc": (
            raw_roc
        ),

        "raw_pr_auc": (
            raw_pr
        ),

        "smoothed_roc_auc": (
            smooth_roc
        ),

        "smoothed_pr_auc": (
            smooth_pr
        ),

        "near_event_mean_raw": float(
            diagnostic.loc[
                diagnostic[
                    "region"
                ].eq(
                    "near_event"
                ),
                "raw_probability",
            ].mean()
        ),

        "background_mean_raw": float(
            diagnostic.loc[
                diagnostic[
                    "region"
                ].eq(
                    "background"
                ),
                "raw_probability",
            ].mean()
        ),

        "near_event_mean_smoothed": float(
            diagnostic.loc[
                diagnostic[
                    "region"
                ].eq(
                    "near_event"
                ),
                "smoothed_probability",
            ].mean()
        ),

        "background_mean_smoothed": float(
            diagnostic.loc[
                diagnostic[
                    "region"
                ].eq(
                    "background"
                ),
                "smoothed_probability",
            ].mean()
        ),
    }


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 8B "
        "Temporal Signal Diagnostics"
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
    print("DIAGNOSTIC DATASET")
    print("-" * 72)

    print(
        "Positive training videos:",
        len(
            cases
        ),
    )

    print(
        "Annotated training events:",
        total_events,
    )

    print(
        "Development validation used:",
        False,
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
        "Threshold:",
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

    event_rows = []

    window_rows = []

    # -----------------------------------------------------
    # Scan each positive TRAIN video only once
    # -----------------------------------------------------

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
            float(
                value
            )
            for value
            in case[
                "event_times"
            ]
        ]

        print()
        print("=" * 72)

        print(
            f"[{case_index}/{len(cases)}] "
            f"SCANNING {video_id}"
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

        global_peak_sec = float(
            result[
                "peak"
            ][
                "timecode_sec"
            ]
        )

        global_peak_probability = float(
            result[
                "peak"
            ][
                "probability"
            ]
        )

        # ---------------------------------------------
        # Save every temporal window
        # ---------------------------------------------

        for item in (
            result[
                "timeline"
            ]
        ):
            center_sec = float(
                item[
                    "center_sec"
                ]
            )

            distance_sec = (
                minimum_event_distance(
                    center_sec,
                    event_times,
                )
            )

            region = (
                temporal_region(
                    distance_sec
                )
            )

            window_rows.append(
                {
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

                    "center_sec": (
                        center_sec
                    ),

                    "min_event_distance_sec": (
                        distance_sec
                    ),

                    "region": (
                        region
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
                }
            )

        # ---------------------------------------------
        # Evaluate each annotated event
        # ---------------------------------------------

        for event_index, event_time in enumerate(
            event_times,
            start=1,
        ):
            event_metrics = (
                analyze_event(
                    event_index=(
                        event_index
                    ),
                    event_time_sec=(
                        event_time
                    ),
                    timeline=(
                        result[
                            "timeline"
                        ]
                    ),
                    all_event_times=(
                        event_times
                    ),
                    global_peak_sec=(
                        global_peak_sec
                    ),
                )
            )

            row = {
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

                "global_peak_probability": (
                    global_peak_probability
                ),

                **event_metrics,
            }

            event_rows.append(
                row
            )

            print()
            print(
                f"Event "
                f"{event_index}: "
                f"{event_time:.3f}s"
            )

            print(
                "  global peak:",
                f"{global_peak_sec:.3f}s",
            )

            print(
                "  peak error:",
                f"{event_metrics['peak_error_sec']:.3f}s",
            )

            print(
                "  event probability:",
                round(
                    event_metrics[
                        "best_event_window_probability"
                    ],
                    4,
                ),
            )

            print(
                "  background max:",
                (
                    round(
                        event_metrics[
                            "background_max_probability"
                        ],
                        4,
                    )
                    if (
                        event_metrics[
                            "background_max_probability"
                        ]
                        is not None
                    )
                    else None
                ),
            )

            print(
                "  temporal margin:",
                (
                    round(
                        event_metrics[
                            "temporal_margin"
                        ],
                        4,
                    )
                    if (
                        event_metrics[
                            "temporal_margin"
                        ]
                        is not None
                    )
                    else None
                ),
            )

            print(
                "  event window rank:",
                event_metrics[
                    "best_event_window_rank"
                ],
            )

    # -----------------------------------------------------
    # DataFrames
    # -----------------------------------------------------

    events_df = pd.DataFrame(
        event_rows
    )

    windows_df = pd.DataFrame(
        window_rows
    )

    events_df.to_csv(
        EVENT_REPORT_FILE,
        index=False,
    )

    windows_df.to_csv(
        WINDOW_REPORT_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Aggregate diagnostics
    # -----------------------------------------------------

    window_metrics = (
        calculate_window_metrics(
            windows_df
        )
    )

    valid_margins = (
        events_df[
            "temporal_margin"
        ]
        .dropna()
        .astype(float)
    )

    positive_margin_count = int(
        (
            valid_margins
            > 0
        ).sum()
    )

    events_above_threshold = int(
        events_df[
            "event_region_above_threshold"
        ].sum()
    )

    peak_within_3 = int(
        events_df[
            "peak_within_3_sec"
        ].sum()
    )

    peak_within_5 = int(
        events_df[
            "peak_within_5_sec"
        ].sum()
    )

    peak_within_10 = int(
        events_df[
            "peak_within_10_sec"
        ].sum()
    )

    summary = {
        "configuration": {
            "dataset": (
                "training positive videos only"
            ),

            "development_validation_used": (
                False
            ),

            "scan_window_sec": (
                localizer.scan_window_sec
            ),

            "scan_stride_sec": (
                localizer.scan_stride_sec
            ),

            "smoothing_radius": (
                localizer.smoothing_radius
            ),

            "threshold": (
                localizer.threshold
            ),

            "near_event_radius_sec": (
                NEAR_EVENT_RADIUS_SEC
            ),

            "background_radius_sec": (
                BACKGROUND_RADIUS_SEC
            ),
        },

        "dataset": {
            "positive_videos": int(
                len(
                    cases
                )
            ),

            "events": int(
                len(
                    events_df
                )
            ),

            "scan_windows": int(
                len(
                    windows_df
                )
            ),
        },

        "event_level": {
            "mean_peak_error_sec": float(
                events_df[
                    "peak_error_sec"
                ].mean()
            ),

            "median_peak_error_sec": float(
                events_df[
                    "peak_error_sec"
                ].median()
            ),

            "peak_within_3_sec": (
                peak_within_3
            ),

            "peak_within_5_sec": (
                peak_within_5
            ),

            "peak_within_10_sec": (
                peak_within_10
            ),

            "event_region_above_threshold": (
                events_above_threshold
            ),

            "mean_event_probability": float(
                events_df[
                    "best_event_window_probability"
                ].mean()
            ),

            "mean_background_max_probability": (
                float(
                    events_df[
                        "background_max_probability"
                    ]
                    .dropna()
                    .mean()
                )
            ),

            "mean_temporal_margin": (
                float(
                    valid_margins.mean()
                )
            ),

            "median_temporal_margin": (
                float(
                    valid_margins.median()
                )
            ),

            "positive_temporal_margin": (
                positive_margin_count
            ),

            "events_with_background": int(
                len(
                    valid_margins
                )
            ),

            "mean_best_event_window_rank": float(
                events_df[
                    "best_event_window_rank"
                ].mean()
            ),
        },

        "window_level": (
            window_metrics
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Console summary
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print("TEMPORAL SIGNAL SUMMARY")
    print("=" * 72)

    print(
        "Positive videos:",
        summary[
            "dataset"
        ][
            "positive_videos"
        ],
    )

    print(
        "Events:",
        summary[
            "dataset"
        ][
            "events"
        ],
    )

    print(
        "Scan windows:",
        summary[
            "dataset"
        ][
            "scan_windows"
        ],
    )

    print()
    print("EVENT LEVEL")
    print("-" * 72)

    print(
        "Mean peak error:",
        round(
            summary[
                "event_level"
            ][
                "mean_peak_error_sec"
            ],
            3,
        ),
        "sec",
    )

    print(
        "Median peak error:",
        round(
            summary[
                "event_level"
            ][
                "median_peak_error_sec"
            ],
            3,
        ),
        "sec",
    )

    print(
        "Peak within ±3 sec:",
        f"{peak_within_3}/"
        f"{len(events_df)}",
    )

    print(
        "Peak within ±5 sec:",
        f"{peak_within_5}/"
        f"{len(events_df)}",
    )

    print(
        "Peak within ±10 sec:",
        f"{peak_within_10}/"
        f"{len(events_df)}",
    )

    print(
        "Event region above threshold:",
        f"{events_above_threshold}/"
        f"{len(events_df)}",
    )

    print(
        "Mean event probability:",
        round(
            summary[
                "event_level"
            ][
                "mean_event_probability"
            ],
            4,
        ),
    )

    print(
        "Mean background max:",
        round(
            summary[
                "event_level"
            ][
                "mean_background_max_probability"
            ],
            4,
        ),
    )

    print(
        "Mean temporal margin:",
        round(
            summary[
                "event_level"
            ][
                "mean_temporal_margin"
            ],
            4,
        ),
    )

    print(
        "Positive temporal margin:",
        f"{positive_margin_count}/"
        f"{len(valid_margins)}",
    )

    print(
        "Mean event-window rank:",
        round(
            summary[
                "event_level"
            ][
                "mean_best_event_window_rank"
            ],
            3,
        ),
    )

    print()
    print("WINDOW LEVEL")
    print("-" * 72)

    for key, value in (
        window_metrics.items()
    ):
        if isinstance(
            value,
            float,
        ):
            print(
                f"{key:28}",
                round(
                    value,
                    4,
                ),
            )
        else:
            print(
                f"{key:28}",
                value,
            )

    print()
    print("=" * 72)
    print("EVENT DETAILS")
    print("=" * 72)

    print(
        events_df[
            [
                "video_id",
                "event_time_sec",
                "global_peak_sec",
                "peak_error_sec",
                "best_event_window_probability",
                "background_max_probability",
                "temporal_margin",
                "best_event_window_rank",
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
        EVENT_REPORT_FILE
    )

    print(
        WINDOW_REPORT_FILE
    )

    print(
        SUMMARY_FILE
    )

    print()
    print(
        "✅ Training temporal signal "
        "diagnostics complete."
    )

    print(
        "✅ Development validation "
        "was not used."
    )


if __name__ == "__main__":
    main()