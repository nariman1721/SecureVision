from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


# =========================================================
# CONFIG
# =========================================================

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20


# =========================================================
# PATHS
# =========================================================

AI_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = AI_ROOT / "data" / "processed"

VIDEOS_FILE = PROCESSED_DIR / "videos.csv"
EVENTS_FILE = PROCESSED_DIR / "events.csv"

TRAIN_VIDEOS_FILE = (
    PROCESSED_DIR / "train_videos.csv"
)

VALIDATION_VIDEOS_FILE = (
    PROCESSED_DIR / "validation_videos.csv"
)

TRAIN_EVENTS_FILE = (
    PROCESSED_DIR / "train_events.csv"
)

VALIDATION_EVENTS_FILE = (
    PROCESSED_DIR / "validation_events.csv"
)

SPLIT_ASSIGNMENTS_FILE = (
    PROCESSED_DIR / "split_assignments.csv"
)

SUMMARY_FILE = (
    PROCESSED_DIR / "split_summary.json"
)


# =========================================================
# HELPERS
# =========================================================

def video_number(video_id: str) -> int:
    match = re.search(
        r"(\d+)",
        str(video_id),
    )

    if not match:
        return 999999

    return int(match.group(1))


def sort_videos(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    result["_sort_id"] = (
        result[
            "canonical_video_id"
        ]
        .apply(video_number)
    )

    result = (
        result
        .sort_values(
            "_sort_id"
        )
        .drop(
            columns=["_sort_id"]
        )
        .reset_index(drop=True)
    )

    return result


def sort_events(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()

    result = dataframe.copy()

    result["_sort_id"] = (
        result[
            "source_video_id"
        ]
        .apply(video_number)
    )

    result = (
        result
        .sort_values(
            [
                "_sort_id",
                "timecode_sec",
            ]
        )
        .drop(
            columns=["_sort_id"]
        )
        .reset_index(drop=True)
    )

    return result


def class_counts(
    dataframe: pd.DataFrame,
) -> dict:
    return {
        "total": int(
            len(dataframe)
        ),
        "violation": int(
            dataframe[
                "label"
            ].eq(1).sum()
        ),
        "no_violation": int(
            dataframe[
                "label"
            ].eq(0).sum()
        ),
    }


# =========================================================
# LOAD DATA
# =========================================================

def load_data():
    if not VIDEOS_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {VIDEOS_FILE}\n"
            "Run build_canonical_dataset.py first."
        )

    if not EVENTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: {EVENTS_FILE}\n"
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
# VALIDATE SOURCE DATA
# =========================================================

def validate_source_data(
    videos: pd.DataFrame,
    events: pd.DataFrame,
):
    required_video_columns = [
        "content_id",
        "canonical_video_id",
        "canonical_filename",
        "label",
        "event_count",
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
        for column
        in required_video_columns
        if column not in videos.columns
    ]

    missing_event_columns = [
        column
        for column
        in required_event_columns
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

    if videos[
        "content_id"
    ].duplicated().any():
        duplicates = videos.loc[
            videos[
                "content_id"
            ].duplicated(
                keep=False
            ),
            [
                "content_id",
                "canonical_video_id",
            ],
        ]

        raise ValueError(
            "Duplicate content_id values found "
            "inside canonical videos:\n"
            f"{duplicates.to_string(index=False)}"
        )

    valid_labels = set(
        videos[
            "label"
        ]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if not valid_labels.issubset(
        {0, 1}
    ):
        raise ValueError(
            f"Unexpected labels: {valid_labels}"
        )

    video_content_ids = set(
        videos["content_id"]
    )

    event_content_ids = set(
        events["content_id"]
    )

    orphan_events = (
        event_content_ids
        - video_content_ids
    )

    if orphan_events:
        raise ValueError(
            "Some events do not belong to "
            "canonical videos: "
            f"{sorted(orphan_events)}"
        )


# =========================================================
# SPLIT
# =========================================================

def create_split(
    videos: pd.DataFrame,
):
    train_videos, validation_videos = (
        train_test_split(
            videos,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=videos["label"],
            shuffle=True,
        )
    )

    train_videos = sort_videos(
        train_videos
    )

    validation_videos = sort_videos(
        validation_videos
    )

    return (
        train_videos,
        validation_videos,
    )


# =========================================================
# SPLIT EVENTS USING CONTENT ID
# =========================================================

def split_events(
    events: pd.DataFrame,
    train_videos: pd.DataFrame,
    validation_videos: pd.DataFrame,
):
    train_content_ids = set(
        train_videos[
            "content_id"
        ]
    )

    validation_content_ids = set(
        validation_videos[
            "content_id"
        ]
    )

    train_events = events[
        events[
            "content_id"
        ].isin(
            train_content_ids
        )
    ].copy()

    validation_events = events[
        events[
            "content_id"
        ].isin(
            validation_content_ids
        )
    ].copy()

    train_events = sort_events(
        train_events
    )

    validation_events = sort_events(
        validation_events
    )

    return (
        train_events,
        validation_events,
    )


# =========================================================
# LEAKAGE CHECKS
# =========================================================

def validate_split(
    original_videos: pd.DataFrame,
    original_events: pd.DataFrame,
    train_videos: pd.DataFrame,
    validation_videos: pd.DataFrame,
    train_events: pd.DataFrame,
    validation_events: pd.DataFrame,
):
    train_ids = set(
        train_videos[
            "content_id"
        ]
    )

    validation_ids = set(
        validation_videos[
            "content_id"
        ]
    )

    overlap = (
        train_ids
        & validation_ids
    )

    if overlap:
        raise RuntimeError(
            "DATA LEAKAGE DETECTED. "
            "content_id exists in both splits: "
            f"{sorted(overlap)}"
        )

    combined_video_ids = (
        train_ids
        | validation_ids
    )

    original_video_ids = set(
        original_videos[
            "content_id"
        ]
    )

    if combined_video_ids != original_video_ids:
        missing = (
            original_video_ids
            - combined_video_ids
        )

        extra = (
            combined_video_ids
            - original_video_ids
        )

        raise RuntimeError(
            "Video split does not match "
            "original canonical dataset.\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}"
        )

    train_event_ids = set(
        train_events[
            "event_id"
        ]
    )

    validation_event_ids = set(
        validation_events[
            "event_id"
        ]
    )

    event_overlap = (
        train_event_ids
        & validation_event_ids
    )

    if event_overlap:
        raise RuntimeError(
            "Event leakage detected: "
            f"{sorted(event_overlap)}"
        )

    combined_events = (
        train_event_ids
        | validation_event_ids
    )

    original_event_ids = set(
        original_events[
            "event_id"
        ]
    )

    if combined_events != original_event_ids:
        raise RuntimeError(
            "Some events were lost during split."
        )

    # Every train event must belong
    # to a train video.
    if not set(
        train_events[
            "content_id"
        ]
    ).issubset(
        train_ids
    ):
        raise RuntimeError(
            "Train event assigned to "
            "non-train video."
        )

    # Every validation event must belong
    # to a validation video.
    if not set(
        validation_events[
            "content_id"
        ]
    ).issubset(
        validation_ids
    ):
        raise RuntimeError(
            "Validation event assigned to "
            "non-validation video."
        )


# =========================================================
# ASSIGNMENT TABLE
# =========================================================

def build_assignments(
    train_videos: pd.DataFrame,
    validation_videos: pd.DataFrame,
):
    train_assignments = (
        train_videos[
            [
                "content_id",
                "canonical_video_id",
                "canonical_filename",
                "label",
                "event_count",
            ]
        ]
        .copy()
    )

    train_assignments[
        "split"
    ] = "train"

    validation_assignments = (
        validation_videos[
            [
                "content_id",
                "canonical_video_id",
                "canonical_filename",
                "label",
                "event_count",
            ]
        ]
        .copy()
    )

    validation_assignments[
        "split"
    ] = "validation"

    assignments = pd.concat(
        [
            train_assignments,
            validation_assignments,
        ],
        ignore_index=True,
    )

    assignments = sort_videos(
        assignments
    )

    return assignments


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 3 Train / Validation Split"
    )
    print("=" * 72)

    videos, events = load_data()

    validate_source_data(
        videos,
        events,
    )

    print()
    print("SOURCE DATASET")
    print("-" * 72)

    print(
        f"Unique videos:     {len(videos)}"
    )

    print(
        "Violation videos:  "
        f"{int(videos['label'].eq(1).sum())}"
    )

    print(
        "No-violation:      "
        f"{int(videos['label'].eq(0).sum())}"
    )

    print(
        f"Violation events:  {len(events)}"
    )

    # -----------------------------------------------------
    # Create split
    # -----------------------------------------------------

    (
        train_videos,
        validation_videos,
    ) = create_split(
        videos
    )

    # -----------------------------------------------------
    # Split events
    # -----------------------------------------------------

    (
        train_events,
        validation_events,
    ) = split_events(
        events,
        train_videos,
        validation_videos,
    )

    # -----------------------------------------------------
    # Validate
    # -----------------------------------------------------

    validate_split(
        videos,
        events,
        train_videos,
        validation_videos,
        train_events,
        validation_events,
    )

    # -----------------------------------------------------
    # Add split column
    # -----------------------------------------------------

    train_videos = (
        train_videos.copy()
    )

    validation_videos = (
        validation_videos.copy()
    )

    train_videos[
        "split"
    ] = "train"

    validation_videos[
        "split"
    ] = "validation"

    # -----------------------------------------------------
    # Save split files
    # -----------------------------------------------------

    train_videos.to_csv(
        TRAIN_VIDEOS_FILE,
        index=False,
    )

    validation_videos.to_csv(
        VALIDATION_VIDEOS_FILE,
        index=False,
    )

    train_events.to_csv(
        TRAIN_EVENTS_FILE,
        index=False,
    )

    validation_events.to_csv(
        VALIDATION_EVENTS_FILE,
        index=False,
    )

    assignments = build_assignments(
        train_videos,
        validation_videos,
    )

    assignments.to_csv(
        SPLIT_ASSIGNMENTS_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    train_counts = class_counts(
        train_videos
    )

    validation_counts = class_counts(
        validation_videos
    )

    train_ratio = (
        len(train_videos)
        / len(videos)
    )

    validation_ratio = (
        len(validation_videos)
        / len(videos)
    )

    summary = {
        "configuration": {
            "random_state": (
                RANDOM_STATE
            ),
            "validation_size": (
                VALIDATION_SIZE
            ),
            "split_strategy": (
                "stratified_by_video_label"
            ),
            "group_identifier": (
                "content_id"
            ),
        },

        "source": {
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
            "events": int(
                len(events)
            ),
        },

        "train": {
            **train_counts,
            "events": int(
                len(train_events)
            ),
            "ratio": round(
                train_ratio,
                4,
            ),
        },

        "validation": {
            **validation_counts,
            "events": int(
                len(
                    validation_events
                )
            ),
            "ratio": round(
                validation_ratio,
                4,
            ),
        },

        "validation_checks": {
            "content_id_overlap": 0,
            "event_id_overlap": 0,
            "all_videos_preserved": True,
            "all_events_preserved": True,
            "data_leakage_detected": False,
        },
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
    # Console report
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print("TRAIN SPLIT")
    print("=" * 72)

    print(
        f"Videos:            "
        f"{train_counts['total']}"
    )

    print(
        f"Violation:         "
        f"{train_counts['violation']}"
    )

    print(
        f"No violation:      "
        f"{train_counts['no_violation']}"
    )

    print(
        f"Violation events:  "
        f"{len(train_events)}"
    )

    print()
    print(
        train_videos[
            [
                "canonical_video_id",
                "label",
                "event_count",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("VALIDATION SPLIT")
    print("=" * 72)

    print(
        f"Videos:            "
        f"{validation_counts['total']}"
    )

    print(
        f"Violation:         "
        f"{validation_counts['violation']}"
    )

    print(
        f"No violation:      "
        f"{validation_counts['no_violation']}"
    )

    print(
        f"Violation events:  "
        f"{len(validation_events)}"
    )

    print()
    print(
        validation_videos[
            [
                "canonical_video_id",
                "label",
                "event_count",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("LEAKAGE VALIDATION")
    print("=" * 72)

    print(
        "content_id overlap: 0"
    )

    print(
        "event_id overlap:   0"
    )

    print(
        "all videos kept:    True"
    )

    print(
        "all events kept:    True"
    )

    print()
    print(
        "✅ No train/validation data leakage detected."
    )

    print()
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    print(TRAIN_VIDEOS_FILE)
    print(VALIDATION_VIDEOS_FILE)
    print(TRAIN_EVENTS_FILE)
    print(VALIDATION_EVENTS_FILE)
    print(SPLIT_ASSIGNMENTS_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()