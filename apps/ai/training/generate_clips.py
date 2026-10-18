from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd


# =========================================================
# CONFIG
# =========================================================

CLIP_DURATION_SEC = 6.0

# Negative clips from clean no-violation videos.
NORMAL_CLIP_POSITIONS = (
    0.25,
    0.50,
    0.75,
)

# Candidate positions for negative windows inside
# violation videos, away from annotated events.
SAME_VIDEO_NEGATIVE_POSITIONS = (
    0.15,
    0.30,
    0.50,
    0.70,
    0.85,
)

# Additional distance between a negative clip and
# a known violation event.
EVENT_GUARD_SEC = 2.0

# At most one same-video negative clip for every
# positive video.
SAME_VIDEO_NEGATIVES_PER_VIDEO = 1


# =========================================================
# PATHS
# =========================================================

AI_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = AI_ROOT / "data" / "raw"
PROCESSED_DIR = AI_ROOT / "data" / "processed"

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

TRAIN_CLIPS_FILE = (
    PROCESSED_DIR / "train_clips.csv"
)

VALIDATION_CLIPS_FILE = (
    PROCESSED_DIR / "validation_clips.csv"
)

SUMMARY_FILE = (
    PROCESSED_DIR / "clips_summary.json"
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


def clamp_window(
    center_sec: float,
    video_duration_sec: float,
    clip_duration_sec: float,
) -> tuple[float, float]:
    """
    Create a fixed-size clip around center_sec while
    keeping it completely inside the video.
    """

    if video_duration_sec <= 0:
        raise ValueError(
            "Video duration must be positive."
        )

    if video_duration_sec < clip_duration_sec:
        return (
            0.0,
            float(video_duration_sec),
        )

    half = clip_duration_sec / 2.0

    start = center_sec - half
    end = center_sec + half

    if start < 0:
        start = 0.0
        end = clip_duration_sec

    if end > video_duration_sec:
        end = video_duration_sec
        start = (
            video_duration_sec
            - clip_duration_sec
        )

    return (
        round(float(start), 3),
        round(float(end), 3),
    )


def event_inside_window(
    event_time_sec: float,
    start_sec: float,
    end_sec: float,
) -> bool:
    return (
        start_sec
        <= event_time_sec
        <= end_sec
    )


def negative_window_is_safe(
    start_sec: float,
    end_sec: float,
    event_times: list[float],
) -> bool:
    """
    Negative window must not overlap a known event
    and must keep EVENT_GUARD_SEC around the clip.
    """

    protected_start = (
        start_sec
        - EVENT_GUARD_SEC
    )

    protected_end = (
        end_sec
        + EVENT_GUARD_SEC
    )

    for event_time in event_times:
        if (
            protected_start
            <= event_time
            <= protected_end
        ):
            return False

    return True


def relative_video_path(
    filename: str,
) -> str:
    return str(
        Path("data")
        / "raw"
        / filename
    )


def check_video_exists(
    filename: str,
) -> None:
    video_path = (
        RAW_DIR / filename
    )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Raw video not found: "
            f"{video_path}"
        )


# =========================================================
# LOAD DATA
# =========================================================

def load_split(
    videos_file: Path,
    events_file: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    if not videos_file.exists():
        raise FileNotFoundError(
            f"Missing file: "
            f"{videos_file}"
        )

    if not events_file.exists():
        raise FileNotFoundError(
            f"Missing file: "
            f"{events_file}"
        )

    videos = pd.read_csv(
        videos_file
    )

    events = pd.read_csv(
        events_file
    )

    return videos, events


# =========================================================
# POSITIVE CLIPS
# =========================================================

def generate_positive_clips(
    videos: pd.DataFrame,
    events: pd.DataFrame,
    split_name: str,
) -> list[dict]:
    clips = []

    video_lookup = (
        videos
        .set_index("content_id")
        .to_dict("index")
    )

    for _, event in events.iterrows():
        content_id = event[
            "content_id"
        ]

        if content_id not in video_lookup:
            raise ValueError(
                f"Event {event['event_id']} "
                f"has no matching video."
            )

        video = video_lookup[
            content_id
        ]

        filename = str(
            video[
                "canonical_filename"
            ]
        )

        check_video_exists(
            filename
        )

        video_duration = float(
            video[
                "duration_sec"
            ]
        )

        event_time = float(
            event[
                "timecode_sec"
            ]
        )

        start_sec, end_sec = (
            clamp_window(
                center_sec=event_time,
                video_duration_sec=(
                    video_duration
                ),
                clip_duration_sec=(
                    CLIP_DURATION_SEC
                ),
            )
        )

        if not event_inside_window(
            event_time,
            start_sec,
            end_sec,
        ):
            raise RuntimeError(
                "Positive clip does not "
                "contain its event."
            )

        clip_id = (
            f"{split_name}"
            f"_positive_"
            f"{event['event_id']}"
        )

        clips.append(
            {
                "clip_id": clip_id,
                "split": split_name,
                "content_id": (
                    content_id
                ),
                "canonical_video_id": (
                    video[
                        "canonical_video_id"
                    ]
                ),
                "video_filename": (
                    filename
                ),
                "video_path": (
                    relative_video_path(
                        filename
                    )
                ),
                "label": 1,
                "clip_type": (
                    "positive_event"
                ),
                "event_id": (
                    event[
                        "event_id"
                    ]
                ),
                "event_timecode_sec": (
                    event_time
                ),
                "center_sec": (
                    event_time
                ),
                "start_sec": (
                    start_sec
                ),
                "end_sec": (
                    end_sec
                ),
                "clip_duration_sec": (
                    round(
                        end_sec
                        - start_sec,
                        3,
                    )
                ),
                "video_duration_sec": (
                    video_duration
                ),
            }
        )

    return clips


# =========================================================
# NEGATIVE CLIPS FROM CLEAN VIDEOS
# =========================================================

def generate_normal_negative_clips(
    videos: pd.DataFrame,
    split_name: str,
) -> list[dict]:
    clips = []

    normal_videos = videos[
        videos[
            "label"
        ].eq(0)
    ]

    for _, video in (
        normal_videos.iterrows()
    ):
        filename = str(
            video[
                "canonical_filename"
            ]
        )

        check_video_exists(
            filename
        )

        video_duration = float(
            video[
                "duration_sec"
            ]
        )

        for index, position in enumerate(
            NORMAL_CLIP_POSITIONS,
            start=1,
        ):
            center_sec = (
                video_duration
                * position
            )

            (
                start_sec,
                end_sec,
            ) = clamp_window(
                center_sec=center_sec,
                video_duration_sec=(
                    video_duration
                ),
                clip_duration_sec=(
                    CLIP_DURATION_SEC
                ),
            )

            clip_id = (
                f"{split_name}"
                f"_normal_"
                f"{video['canonical_video_id']}"
                f"_{index}"
            )

            clips.append(
                {
                    "clip_id": (
                        clip_id
                    ),
                    "split": (
                        split_name
                    ),
                    "content_id": (
                        video[
                            "content_id"
                        ]
                    ),
                    "canonical_video_id": (
                        video[
                            "canonical_video_id"
                        ]
                    ),
                    "video_filename": (
                        filename
                    ),
                    "video_path": (
                        relative_video_path(
                            filename
                        )
                    ),
                    "label": 0,
                    "clip_type": (
                        "negative_normal"
                    ),
                    "event_id": None,
                    "event_timecode_sec": (
                        None
                    ),
                    "center_sec": (
                        round(
                            center_sec,
                            3,
                        )
                    ),
                    "start_sec": (
                        start_sec
                    ),
                    "end_sec": (
                        end_sec
                    ),
                    "clip_duration_sec": (
                        round(
                            end_sec
                            - start_sec,
                            3,
                        )
                    ),
                    "video_duration_sec": (
                        video_duration
                    ),
                }
            )

    return clips


# =========================================================
# NEGATIVE CLIPS FROM VIOLATION VIDEOS
# =========================================================

def generate_same_video_negatives(
    videos: pd.DataFrame,
    events: pd.DataFrame,
    split_name: str,
) -> list[dict]:
    clips = []

    positive_videos = videos[
        videos[
            "label"
        ].eq(1)
    ]

    for _, video in (
        positive_videos.iterrows()
    ):
        content_id = video[
            "content_id"
        ]

        filename = str(
            video[
                "canonical_filename"
            ]
        )

        check_video_exists(
            filename
        )

        video_duration = float(
            video[
                "duration_sec"
            ]
        )

        event_times = (
            events.loc[
                events[
                    "content_id"
                ].eq(content_id),
                "timecode_sec",
            ]
            .astype(float)
            .tolist()
        )

        if not event_times:
            raise ValueError(
                "Violation video has no "
                "event annotations: "
                f"{video['canonical_video_id']}"
            )

        candidate_windows = []

        seen_windows = set()

        for position in (
            SAME_VIDEO_NEGATIVE_POSITIONS
        ):
            center_sec = (
                video_duration
                * position
            )

            (
                start_sec,
                end_sec,
            ) = clamp_window(
                center_sec=center_sec,
                video_duration_sec=(
                    video_duration
                ),
                clip_duration_sec=(
                    CLIP_DURATION_SEC
                ),
            )

            window_key = (
                start_sec,
                end_sec,
            )

            if window_key in seen_windows:
                continue

            seen_windows.add(
                window_key
            )

            if not negative_window_is_safe(
                start_sec,
                end_sec,
                event_times,
            ):
                continue

            center_actual = (
                start_sec
                + end_sec
            ) / 2.0

            distance_to_event = min(
                abs(
                    center_actual
                    - event_time
                )
                for event_time
                in event_times
            )

            candidate_windows.append(
                {
                    "start_sec": (
                        start_sec
                    ),
                    "end_sec": (
                        end_sec
                    ),
                    "center_sec": (
                        round(
                            center_actual,
                            3,
                        )
                    ),
                    "distance_to_event": (
                        distance_to_event
                    ),
                }
            )

        # Prefer the safest window that is
        # furthest from all known events.
        candidate_windows.sort(
            key=lambda item: (
                item[
                    "distance_to_event"
                ]
            ),
            reverse=True,
        )

        selected = (
            candidate_windows[
                :SAME_VIDEO_NEGATIVES_PER_VIDEO
            ]
        )

        for index, candidate in enumerate(
            selected,
            start=1,
        ):
            clip_id = (
                f"{split_name}"
                f"_same_video_negative_"
                f"{video['canonical_video_id']}"
                f"_{index}"
            )

            clips.append(
                {
                    "clip_id": (
                        clip_id
                    ),
                    "split": (
                        split_name
                    ),
                    "content_id": (
                        content_id
                    ),
                    "canonical_video_id": (
                        video[
                            "canonical_video_id"
                        ]
                    ),
                    "video_filename": (
                        filename
                    ),
                    "video_path": (
                        relative_video_path(
                            filename
                        )
                    ),
                    "label": 0,
                    "clip_type": (
                        "negative_same_video"
                    ),
                    "event_id": None,
                    "event_timecode_sec": (
                        None
                    ),
                    "center_sec": (
                        candidate[
                            "center_sec"
                        ]
                    ),
                    "start_sec": (
                        candidate[
                            "start_sec"
                        ]
                    ),
                    "end_sec": (
                        candidate[
                            "end_sec"
                        ]
                    ),
                    "clip_duration_sec": (
                        round(
                            candidate[
                                "end_sec"
                            ]
                            - candidate[
                                "start_sec"
                            ],
                            3,
                        )
                    ),
                    "video_duration_sec": (
                        video_duration
                    ),
                }
            )

    return clips


# =========================================================
# GENERATE ONE SPLIT
# =========================================================

def generate_split_clips(
    videos: pd.DataFrame,
    events: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    positive = (
        generate_positive_clips(
            videos,
            events,
            split_name,
        )
    )

    normal_negative = (
        generate_normal_negative_clips(
            videos,
            split_name,
        )
    )

    same_video_negative = (
        generate_same_video_negatives(
            videos,
            events,
            split_name,
        )
    )

    rows = (
        positive
        + normal_negative
        + same_video_negative
    )

    clips = pd.DataFrame(
        rows
    )

    clips["_video_number"] = (
        clips[
            "canonical_video_id"
        ]
        .apply(video_number)
    )

    clips = (
        clips
        .sort_values(
            [
                "_video_number",
                "start_sec",
                "label",
            ]
        )
        .drop(
            columns=[
                "_video_number"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return clips


# =========================================================
# VALIDATION
# =========================================================

def validate_clips(
    train_clips: pd.DataFrame,
    validation_clips: pd.DataFrame,
    train_events: pd.DataFrame,
    validation_events: pd.DataFrame,
):
    train_ids = set(
        train_clips[
            "content_id"
        ]
    )

    validation_ids = set(
        validation_clips[
            "content_id"
        ]
    )

    overlap = (
        train_ids
        & validation_ids
    )

    if overlap:
        raise RuntimeError(
            "Clip-level data leakage "
            "detected between train and "
            "validation."
        )

    all_clips = pd.concat(
        [
            train_clips,
            validation_clips,
        ],
        ignore_index=True,
    )

    invalid_start = all_clips[
        all_clips[
            "start_sec"
        ] < 0
    ]

    if not invalid_start.empty:
        raise RuntimeError(
            "Some clips start before 0."
        )

    invalid_end = all_clips[
        all_clips[
            "end_sec"
        ]
        >
        all_clips[
            "video_duration_sec"
        ]
        + 0.001
    ]

    if not invalid_end.empty:
        raise RuntimeError(
            "Some clips end after "
            "video duration."
        )

    invalid_duration = all_clips[
        all_clips[
            "end_sec"
        ]
        <=
        all_clips[
            "start_sec"
        ]
    ]

    if not invalid_duration.empty:
        raise RuntimeError(
            "Invalid clip durations found."
        )

    positive_clips = all_clips[
        all_clips[
            "label"
        ].eq(1)
    ]

    for _, clip in (
        positive_clips.iterrows()
    ):
        event_time = float(
            clip[
                "event_timecode_sec"
            ]
        )

        if not event_inside_window(
            event_time,
            float(
                clip[
                    "start_sec"
                ]
            ),
            float(
                clip[
                    "end_sec"
                ]
            ),
        ):
            raise RuntimeError(
                f"Positive clip "
                f"{clip['clip_id']} does "
                f"not contain its event."
            )

    expected_event_ids = set(
        pd.concat(
            [
                train_events,
                validation_events,
            ]
        )[
            "event_id"
        ]
    )

    generated_event_ids = set(
        positive_clips[
            "event_id"
        ].dropna()
    )

    if (
        expected_event_ids
        != generated_event_ids
    ):
        raise RuntimeError(
            "Not all events received "
            "positive clips."
        )

    if all_clips[
        "clip_id"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate clip_id detected."
        )


# =========================================================
# SUMMARY
# =========================================================

def split_statistics(
    clips: pd.DataFrame,
) -> dict:
    return {
        "total_clips": int(
            len(clips)
        ),

        "positive_clips": int(
            clips[
                "label"
            ].eq(1).sum()
        ),

        "negative_clips": int(
            clips[
                "label"
            ].eq(0).sum()
        ),

        "negative_normal": int(
            clips[
                "clip_type"
            ]
            .eq(
                "negative_normal"
            )
            .sum()
        ),

        "negative_same_video": int(
            clips[
                "clip_type"
            ]
            .eq(
                "negative_same_video"
            )
            .sum()
        ),

        "unique_video_contents": int(
            clips[
                "content_id"
            ].nunique()
        ),
    }


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 4 Clip Generation"
    )
    print("=" * 72)

    (
        train_videos,
        train_events,
    ) = load_split(
        TRAIN_VIDEOS_FILE,
        TRAIN_EVENTS_FILE,
    )

    (
        validation_videos,
        validation_events,
    ) = load_split(
        VALIDATION_VIDEOS_FILE,
        VALIDATION_EVENTS_FILE,
    )

    print()
    print("SOURCE SPLITS")
    print("-" * 72)

    print(
        f"Train videos:       "
        f"{len(train_videos)}"
    )

    print(
        f"Train events:       "
        f"{len(train_events)}"
    )

    print(
        f"Validation videos:  "
        f"{len(validation_videos)}"
    )

    print(
        f"Validation events:  "
        f"{len(validation_events)}"
    )

    # -----------------------------------------------------
    # Generate manifests
    # -----------------------------------------------------

    train_clips = (
        generate_split_clips(
            train_videos,
            train_events,
            "train",
        )
    )

    validation_clips = (
        generate_split_clips(
            validation_videos,
            validation_events,
            "validation",
        )
    )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    validate_clips(
        train_clips,
        validation_clips,
        train_events,
        validation_events,
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    train_clips.to_csv(
        TRAIN_CLIPS_FILE,
        index=False,
    )

    validation_clips.to_csv(
        VALIDATION_CLIPS_FILE,
        index=False,
    )

    train_stats = (
        split_statistics(
            train_clips
        )
    )

    validation_stats = (
        split_statistics(
            validation_clips
        )
    )

    summary = {
        "configuration": {
            "clip_duration_sec": (
                CLIP_DURATION_SEC
            ),
            "normal_clip_positions": (
                list(
                    NORMAL_CLIP_POSITIONS
                )
            ),
            "same_video_negative_positions": (
                list(
                    SAME_VIDEO_NEGATIVE_POSITIONS
                )
            ),
            "event_guard_sec": (
                EVENT_GUARD_SEC
            ),
            "same_video_negatives_per_video": (
                SAME_VIDEO_NEGATIVES_PER_VIDEO
            ),
            "storage_strategy": (
                "temporal_manifest_no_reencoding"
            ),
        },

        "train": (
            train_stats
        ),

        "validation": (
            validation_stats
        ),

        "validation_checks": {
            "train_validation_content_overlap": 0,
            "all_events_have_positive_clips": True,
            "all_clip_windows_valid": True,
            "duplicate_clip_ids": 0,
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
    # Console output
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print("TRAIN CLIPS")
    print("=" * 72)

    for key, value in (
        train_stats.items()
    ):
        print(
            f"{key:28} {value}"
        )

    print()
    print("=" * 72)
    print("VALIDATION CLIPS")
    print("=" * 72)

    for key, value in (
        validation_stats.items()
    ):
        print(
            f"{key:28} {value}"
        )

    print()
    print("=" * 72)
    print("CLIP TYPE DISTRIBUTION")
    print("=" * 72)

    print()
    print("TRAIN")
    print(
        train_clips[
            "clip_type"
        ].value_counts()
    )

    print()
    print("VALIDATION")
    print(
        validation_clips[
            "clip_type"
        ].value_counts()
    )

    print()
    print("=" * 72)
    print("SAMPLE POSITIVE CLIPS")
    print("=" * 72)

    print(
        train_clips[
            train_clips[
                "label"
            ].eq(1)
        ][
            [
                "clip_id",
                "canonical_video_id",
                "event_timecode_sec",
                "start_sec",
                "end_sec",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("SAMPLE NEGATIVE CLIPS")
    print("=" * 72)

    print(
        train_clips[
            train_clips[
                "label"
            ].eq(0)
        ][
            [
                "clip_id",
                "canonical_video_id",
                "clip_type",
                "start_sec",
                "end_sec",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print("VALIDATION CHECKS")
    print("=" * 72)

    print(
        "Train/validation content overlap: 0"
    )

    print(
        "All events have positive clips:   True"
    )

    print(
        "All clip windows valid:           True"
    )

    print(
        "Duplicate clip IDs:               0"
    )

    print()
    print(
        "✅ Clip manifests generated successfully."
    )

    print()
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    print(TRAIN_CLIPS_FILE)
    print(VALIDATION_CLIPS_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()