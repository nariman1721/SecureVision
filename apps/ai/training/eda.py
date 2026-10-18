from __future__ import annotations
import io
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

AI_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = AI_ROOT / "data" / "processed"
REPORTS_DIR = AI_ROOT / "reports"

VIDEOS_FILE = PROCESSED_DIR / "videos.csv"
EVENTS_FILE = PROCESSED_DIR / "events.csv"

STATISTICS_FILE = REPORTS_DIR / "dataset_statistics.json"
REPORT_FILE = REPORTS_DIR / "dataset_report.txt"


# =========================================================
# HELPERS
# =========================================================

def label_name(label: int) -> str:
    if int(label) == 1:
        return "Violation"

    return "No violation"


def save_current_figure(filename: str) -> None:
    output_path = REPORTS_DIR / filename

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved: {output_path}")


def dataframe_info_to_string(
    dataframe: pd.DataFrame,
) -> str:
    buffer = io.StringIO()

    dataframe.info(buf=buffer)

    return buffer.getvalue()


def safe_float(value):
    if pd.isna(value):
        return None

    return float(value)


# =========================================================
# LOAD DATA
# =========================================================

def load_data():
    if not VIDEOS_FILE.exists():
        raise FileNotFoundError(
            f"videos.csv not found:\n{VIDEOS_FILE}\n\n"
            "Run build_canonical_dataset.py first."
        )

    if not EVENTS_FILE.exists():
        raise FileNotFoundError(
            f"events.csv not found:\n{EVENTS_FILE}\n\n"
            "Run build_canonical_dataset.py first."
        )

    videos = pd.read_csv(
        VIDEOS_FILE
    )

    events = pd.read_csv(
        EVENTS_FILE
    )

    return videos, events


# =========================================================
# VALIDATION
# =========================================================

def validate_data(
    videos: pd.DataFrame,
    events: pd.DataFrame,
):
    required_video_columns = [
        "content_id",
        "canonical_video_id",
        "canonical_filename",
        "label",
        "event_count",
        "duration_sec",
        "fps",
        "frame_count",
        "width",
        "height",
        "file_size_mb",
    ]

    required_event_columns = [
        "event_id",
        "content_id",
        "source_video_id",
        "source_filename",
        "timecode_sec",
    ]

    missing_video_columns = [
        column
        for column in required_video_columns
        if column not in videos.columns
    ]

    missing_event_columns = [
        column
        for column in required_event_columns
        if column not in events.columns
    ]

    if missing_video_columns:
        raise ValueError(
            "videos.csv is missing columns: "
            f"{missing_video_columns}"
        )

    if missing_event_columns:
        raise ValueError(
            "events.csv is missing columns: "
            f"{missing_event_columns}"
        )


# =========================================================
# PREPARE ANALYSIS TABLES
# =========================================================

def prepare_analysis_tables(
    videos: pd.DataFrame,
    events: pd.DataFrame,
):
    videos = videos.copy()
    events = events.copy()

    videos["label_name"] = (
        videos["label"]
        .astype(int)
        .apply(label_name)
    )

    # Merge event annotations with canonical video metadata.
    events_analysis = events.merge(
        videos[
            [
                "content_id",
                "canonical_video_id",
                "duration_sec",
                "label",
            ]
        ],
        on="content_id",
        how="left",
        validate="many_to_one",
    )

    events_analysis["normalized_position"] = (
        events_analysis["timecode_sec"]
        / events_analysis["duration_sec"]
    )

    events_analysis[
        "event_position_percent"
    ] = (
        events_analysis[
            "normalized_position"
        ]
        * 100
    )

    return videos, events_analysis


# =========================================================
# CONSOLE EDA
# =========================================================

def print_basic_eda(
    videos: pd.DataFrame,
    events: pd.DataFrame,
):
    print()
    print("=" * 72)
    print("PHASE 2B — EXPLORATORY DATA ANALYSIS")
    print("=" * 72)

    print()
    print("VIDEOS DATAFRAME INFO")
    print("-" * 72)

    videos.info()

    print()
    print("EVENTS DATAFRAME INFO")
    print("-" * 72)

    events.info()

    print()
    print("VIDEO LABEL COUNTS")
    print("-" * 72)

    print(
        videos[
            "label_name"
        ].value_counts()
    )

    print()
    print("VIDEO NUMERIC DESCRIPTION")
    print("-" * 72)

    print(
        videos[
            [
                "duration_sec",
                "fps",
                "frame_count",
                "width",
                "height",
                "file_size_mb",
                "event_count",
            ]
        ]
        .describe()
        .round(3)
    )

    print()
    print("EVENT NUMERIC DESCRIPTION")
    print("-" * 72)

    print(
        events[
            [
                "timecode_sec",
                "normalized_position",
                "event_position_percent",
            ]
        ]
        .describe()
        .round(3)
    )

    print()
    print("DURATION GROUPED BY CLASS")
    print("-" * 72)

    duration_grouped = (
        videos
        .groupby(
            "label_name"
        )
        .agg(
            video_count=(
                "content_id",
                "count",
            ),
            mean_duration_sec=(
                "duration_sec",
                "mean",
            ),
            median_duration_sec=(
                "duration_sec",
                "median",
            ),
            min_duration_sec=(
                "duration_sec",
                "min",
            ),
            max_duration_sec=(
                "duration_sec",
                "max",
            ),
            mean_file_size_mb=(
                "file_size_mb",
                "mean",
            ),
        )
        .round(3)
    )

    print(duration_grouped)

    print()
    print("PIVOT TABLE")
    print("-" * 72)

    pivot = pd.pivot_table(
        videos,
        index="label_name",
        values=[
            "duration_sec",
            "file_size_mb",
            "fps",
        ],
        aggfunc=[
            "mean",
            "median",
            "min",
            "max",
        ],
    )

    print(
        pivot.round(3)
    )

    print()
    print("MULTI-EVENT VIDEOS")
    print("-" * 72)

    multi_event = videos.loc[
        videos["event_count"] > 1,
        [
            "canonical_video_id",
            "event_count",
            "event_timecodes",
        ],
    ]

    if multi_event.empty:
        print(
            "No multi-event videos."
        )
    else:
        print(
            multi_event.to_string(
                index=False
            )
        )


# =========================================================
# CHART 1 — CLASS DISTRIBUTION
# =========================================================

def plot_class_distribution(
    videos: pd.DataFrame,
):
    counts = (
        videos[
            "label_name"
        ]
        .value_counts()
        .reindex(
            [
                "No violation",
                "Violation",
            ]
        )
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    bars = ax.bar(
        counts.index,
        counts.values,
    )

    ax.set_title(
        "SecureVision Dataset — Class Distribution"
    )

    ax.set_xlabel(
        "Class"
    )

    ax.set_ylabel(
        "Number of Unique Videos"
    )

    for bar, value in zip(
        bars,
        counts.values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + 0.2,
            str(int(value)),
            ha="center",
        )

    save_current_figure(
        "class_distribution.png"
    )


# =========================================================
# CHART 2 — DURATION DISTRIBUTION
# =========================================================

def plot_duration_distribution(
    videos: pd.DataFrame,
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        videos["duration_sec"],
        bins=10,
    )

    ax.set_title(
        "Video Duration Distribution"
    )

    ax.set_xlabel(
        "Duration (seconds)"
    )

    ax.set_ylabel(
        "Number of Videos"
    )

    save_current_figure(
        "duration_distribution.png"
    )


# =========================================================
# CHART 3 — DURATION BY CLASS
# =========================================================

def plot_duration_by_class(
    videos: pd.DataFrame,
):
    normal = videos.loc[
        videos["label"].eq(0),
        "duration_sec",
    ]

    violations = videos.loc[
        videos["label"].eq(1),
        "duration_sec",
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.boxplot(
        [
            normal,
            violations,
        ],
        tick_labels=[
            "No violation",
            "Violation",
        ],
    )

    ax.set_title(
        "Video Duration by Class"
    )

    ax.set_ylabel(
        "Duration (seconds)"
    )

    save_current_figure(
        "duration_by_class.png"
    )


# =========================================================
# CHART 4 — VIOLATION TIMESTAMPS
# =========================================================

def plot_timestamp_distribution(
    events: pd.DataFrame,
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        events["timecode_sec"],
        bins=10,
    )

    ax.set_title(
        "Violation Timestamp Distribution"
    )

    ax.set_xlabel(
        "Violation Timecode (seconds)"
    )

    ax.set_ylabel(
        "Number of Events"
    )

    save_current_figure(
        "timestamp_distribution.png"
    )


# =========================================================
# CHART 5 — NORMALIZED EVENT POSITION
# =========================================================

def plot_normalized_timestamp_distribution(
    events: pd.DataFrame,
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        events["normalized_position"],
        bins=np.linspace(
            0,
            1,
            11,
        ),
    )

    ax.set_title(
        "Normalized Violation Position"
    )

    ax.set_xlabel(
        "Relative Position in Video"
    )

    ax.set_ylabel(
        "Number of Events"
    )

    ax.set_xlim(
        0,
        1,
    )

    save_current_figure(
        "normalized_timestamp_distribution.png"
    )


# =========================================================
# CHART 6 — EVENT POSITION PERCENT
# =========================================================

def plot_event_position_distribution(
    events: pd.DataFrame,
):
    sorted_events = (
        events
        .sort_values(
            "event_position_percent"
        )
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.scatter(
        sorted_events[
            "event_position_percent"
        ],
        range(
            1,
            len(sorted_events) + 1,
        ),
    )

    ax.axvline(
        25,
        linestyle="--",
        linewidth=1,
    )

    ax.axvline(
        50,
        linestyle="--",
        linewidth=1,
    )

    ax.axvline(
        75,
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(
        "Violation Position Within Videos"
    )

    ax.set_xlabel(
        "Video Progress (%)"
    )

    ax.set_ylabel(
        "Violation Event"
    )

    ax.set_xlim(
        0,
        100,
    )

    save_current_figure(
        "event_position_distribution.png"
    )


# =========================================================
# CHART 7 — FPS DISTRIBUTION
# =========================================================

def plot_fps_distribution(
    videos: pd.DataFrame,
):
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        videos["fps"],
        bins=10,
    )

    ax.set_title(
        "FPS Distribution"
    )

    ax.set_xlabel(
        "Frames Per Second"
    )

    ax.set_ylabel(
        "Number of Videos"
    )

    save_current_figure(
        "fps_distribution.png"
    )


# =========================================================
# STATISTICS
# =========================================================

def build_statistics(
    videos: pd.DataFrame,
    events: pd.DataFrame,
):
    label_counts = (
        videos[
            "label_name"
        ]
        .value_counts()
        .to_dict()
    )

    duration_by_class = {}

    grouped = videos.groupby(
        "label_name"
    )

    for name, group in grouped:
        duration_by_class[name] = {
            "count": int(
                len(group)
            ),
            "mean_sec": safe_float(
                group[
                    "duration_sec"
                ].mean()
            ),
            "median_sec": safe_float(
                group[
                    "duration_sec"
                ].median()
            ),
            "min_sec": safe_float(
                group[
                    "duration_sec"
                ].min()
            ),
            "max_sec": safe_float(
                group[
                    "duration_sec"
                ].max()
            ),
        }

    event_quartiles = (
        events[
            "normalized_position"
        ]
        .quantile(
            [
                0.25,
                0.50,
                0.75,
            ]
        )
    )

    first_quarter = int(
        (
            events[
                "normalized_position"
            ]
            < 0.25
        ).sum()
    )

    second_quarter = int(
        (
            (
                events[
                    "normalized_position"
                ]
                >= 0.25
            )
            &
            (
                events[
                    "normalized_position"
                ]
                < 0.50
            )
        ).sum()
    )

    third_quarter = int(
        (
            (
                events[
                    "normalized_position"
                ]
                >= 0.50
            )
            &
            (
                events[
                    "normalized_position"
                ]
                < 0.75
            )
        ).sum()
    )

    fourth_quarter = int(
        (
            events[
                "normalized_position"
            ]
            >= 0.75
        ).sum()
    )

    statistics = {
        "dataset": {
            "unique_videos": int(
                len(videos)
            ),
            "violation_videos": int(
                videos[
                    "label"
                ].eq(1).sum()
            ),
            "no_violation_videos": int(
                videos[
                    "label"
                ].eq(0).sum()
            ),
            "violation_events": int(
                len(events)
            ),
            "multi_event_videos": int(
                videos[
                    "event_count"
                ].gt(1).sum()
            ),
        },

        "class_distribution": {
            str(key): int(value)
            for key, value
            in label_counts.items()
        },

        "video_duration_seconds": {
            "mean": safe_float(
                videos[
                    "duration_sec"
                ].mean()
            ),
            "median": safe_float(
                videos[
                    "duration_sec"
                ].median()
            ),
            "min": safe_float(
                videos[
                    "duration_sec"
                ].min()
            ),
            "max": safe_float(
                videos[
                    "duration_sec"
                ].max()
            ),
            "std": safe_float(
                videos[
                    "duration_sec"
                ].std()
            ),
        },

        "duration_by_class": (
            duration_by_class
        ),

        "fps": {
            "mean": safe_float(
                videos["fps"].mean()
            ),
            "min": safe_float(
                videos["fps"].min()
            ),
            "max": safe_float(
                videos["fps"].max()
            ),
            "std": safe_float(
                videos["fps"].std()
            ),
        },

        "resolution": {
            "unique_widths": (
                sorted(
                    int(value)
                    for value
                    in videos[
                        "width"
                    ].unique()
                )
            ),
            "unique_heights": (
                sorted(
                    int(value)
                    for value
                    in videos[
                        "height"
                    ].unique()
                )
            ),
        },

        "event_timecode_seconds": {
            "mean": safe_float(
                events[
                    "timecode_sec"
                ].mean()
            ),
            "median": safe_float(
                events[
                    "timecode_sec"
                ].median()
            ),
            "min": safe_float(
                events[
                    "timecode_sec"
                ].min()
            ),
            "max": safe_float(
                events[
                    "timecode_sec"
                ].max()
            ),
        },

        "normalized_event_position": {
            "mean": safe_float(
                events[
                    "normalized_position"
                ].mean()
            ),
            "median": safe_float(
                events[
                    "normalized_position"
                ].median()
            ),
            "q25": safe_float(
                event_quartiles.loc[
                    0.25
                ]
            ),
            "q50": safe_float(
                event_quartiles.loc[
                    0.50
                ]
            ),
            "q75": safe_float(
                event_quartiles.loc[
                    0.75
                ]
            ),
        },

        "events_by_video_quarter": {
            "0_25_percent": (
                first_quarter
            ),
            "25_50_percent": (
                second_quarter
            ),
            "50_75_percent": (
                third_quarter
            ),
            "75_100_percent": (
                fourth_quarter
            ),
        },
    }

    return statistics


# =========================================================
# TEXT REPORT
# =========================================================

def build_text_report(
    videos: pd.DataFrame,
    events: pd.DataFrame,
    statistics: dict,
):
    videos_info = (
        dataframe_info_to_string(
            videos
        )
    )

    events_info = (
        dataframe_info_to_string(
            events
        )
    )

    duration_grouped = (
        videos
        .groupby(
            "label_name"
        )
        .agg(
            count=(
                "content_id",
                "count",
            ),
            mean_duration=(
                "duration_sec",
                "mean",
            ),
            median_duration=(
                "duration_sec",
                "median",
            ),
            min_duration=(
                "duration_sec",
                "min",
            ),
            max_duration=(
                "duration_sec",
                "max",
            ),
        )
        .round(3)
    )

    lines = []

    lines.append(
        "SecureVision Dataset EDA Report"
    )

    lines.append(
        "=" * 72
    )

    lines.append("")

    lines.append(
        "1. DATASET OVERVIEW"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        f"Unique video contents: "
        f"{len(videos)}"
    )

    lines.append(
        "Violation videos: "
        f"{int(videos['label'].eq(1).sum())}"
    )

    lines.append(
        "No-violation videos: "
        f"{int(videos['label'].eq(0).sum())}"
    )

    lines.append(
        f"Violation events: "
        f"{len(events)}"
    )

    lines.append(
        "Multi-event videos: "
        f"{int(videos['event_count'].gt(1).sum())}"
    )

    lines.append("")

    lines.append(
        "2. CLASS DISTRIBUTION"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        videos[
            "label_name"
        ]
        .value_counts()
        .to_string()
    )

    lines.append("")

    lines.append(
        "3. DURATION BY CLASS"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        duration_grouped.to_string()
    )

    lines.append("")

    lines.append(
        "4. VIOLATION TIMECODES"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        events[
            [
                "timecode_sec",
                "normalized_position",
                "event_position_percent",
            ]
        ]
        .describe()
        .round(3)
        .to_string()
    )

    lines.append("")

    lines.append(
        "5. EVENTS BY VIDEO QUARTER"
    )

    lines.append(
        "-" * 72
    )

    for key, value in (
        statistics[
            "events_by_video_quarter"
        ].items()
    ):
        lines.append(
            f"{key}: {value}"
        )

    lines.append("")

    lines.append(
        "6. VIDEOS DATAFRAME INFO"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        videos_info
    )

    lines.append("")

    lines.append(
        "7. EVENTS DATAFRAME INFO"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        events_info
    )

    lines.append("")

    lines.append(
        "8. ML NOTES"
    )

    lines.append(
        "-" * 72
    )

    lines.append(
        "- Split train/validation by content_id, "
        "never by extracted clip."
    )

    lines.append(
        "- Duplicate source filenames were already "
        "collapsed into unique video contents."
    )

    lines.append(
        "- Violation event locations should be used "
        "to construct positive temporal clips."
    )

    lines.append(
        "- Normal videos should provide negative clips."
    )

    lines.append(
        "- Check normalized event positions to avoid "
        "learning dataset timing shortcuts."
    )

    return "\n".join(lines)


# =========================================================
# MAIN
# =========================================================

def main():
    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    videos, events = load_data()

    validate_data(
        videos,
        events,
    )

    videos, events_analysis = (
        prepare_analysis_tables(
            videos,
            events,
        )
    )

    print_basic_eda(
        videos,
        events_analysis,
    )

    # -----------------------------------------------------
    # Charts
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print("GENERATING CHARTS")
    print("=" * 72)

    plot_class_distribution(
        videos
    )

    plot_duration_distribution(
        videos
    )

    plot_duration_by_class(
        videos
    )

    plot_timestamp_distribution(
        events_analysis
    )

    plot_normalized_timestamp_distribution(
        events_analysis
    )

    plot_event_position_distribution(
        events_analysis
    )

    plot_fps_distribution(
        videos
    )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    statistics = build_statistics(
        videos,
        events_analysis,
    )

    STATISTICS_FILE.write_text(
        json.dumps(
            statistics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Text report
    # -----------------------------------------------------

    report = build_text_report(
        videos,
        events_analysis,
        statistics,
    )

    REPORT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("EDA COMPLETE")
    print("=" * 72)

    print(
        f"Unique videos: {len(videos)}"
    )

    print(
        "Violation videos: "
        f"{int(videos['label'].eq(1).sum())}"
    )

    print(
        "No-violation videos: "
        f"{int(videos['label'].eq(0).sum())}"
    )

    print(
        f"Violation events: "
        f"{len(events_analysis)}"
    )

    print()

    print(
        f"Statistics: {STATISTICS_FILE}"
    )

    print(
        f"Report:     {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()