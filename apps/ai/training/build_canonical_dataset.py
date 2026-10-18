from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


AI_ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = (
    AI_ROOT
    / "data"
    / "processed"
    / "metadata.csv"
)

VIDEOS_FILE = (
    AI_ROOT
    / "data"
    / "processed"
    / "videos.csv"
)

EVENTS_FILE = (
    AI_ROOT
    / "data"
    / "processed"
    / "events.csv"
)

SUMMARY_FILE = (
    AI_ROOT
    / "data"
    / "processed"
    / "canonical_summary.json"
)


def video_number(video_id: str) -> int:
    """
    Extract numeric part from IDs such as:

    X-5  -> 5
    X-48 -> 48
    """

    match = re.search(r"(\d+)", str(video_id))

    if not match:
        return 999999

    return int(match.group(1))


def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 2A Canonical Dataset"
    )
    print("=" * 72)

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"metadata.csv not found: "
            f"{METADATA_FILE}"
        )

    df = pd.read_csv(METADATA_FILE)

    print()
    print(f"Metadata rows: {len(df)}")

    required_columns = [
        "video_id",
        "actual_filename",
        "label",
        "timecode_sec",
        "sha256",
        "duration_sec",
        "fps",
        "frame_count",
        "width",
        "height",
        "file_size_mb",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # -----------------------------------------------------
    # EVENTS TABLE
    # -----------------------------------------------------

    events = (
        df[
            df["label"].eq(1)
            & df["timecode_sec"].notna()
        ][
            [
                "sha256",
                "video_id",
                "actual_filename",
                "timecode_sec",
            ]
        ]
        .copy()
        .rename(
            columns={
                "sha256": "content_id",
                "video_id": "source_video_id",
                "actual_filename": (
                    "source_filename"
                ),
            }
        )
    )

    # Human-readable numeric sorting:
    # X-1, X-2, X-4, X-11...
    events["_video_number"] = (
        events["source_video_id"]
        .apply(video_number)
    )

    events = (
        events
        .sort_values(
            [
                "_video_number",
                "timecode_sec",
            ]
        )
        .drop(
            columns=["_video_number"]
        )
        .reset_index(drop=True)
    )

    events.insert(
        0,
        "event_id",
        [
            f"E-{index:03d}"
            for index in range(
                1,
                len(events) + 1,
            )
        ],
    )

    # -----------------------------------------------------
    # VIDEOS TABLE
    # -----------------------------------------------------

    video_rows = []
    label_conflict_groups = []

    for content_id, group in df.groupby(
        "sha256"
    ):
        group = group.copy()

        # Natural numeric ordering inside
        # duplicate groups.
        group["_video_number"] = (
            group["video_id"]
            .apply(video_number)
        )

        group = (
            group
            .sort_values(
                "_video_number"
            )
            .reset_index(drop=True)
        )

        labels = sorted(
            group["label"]
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        label_conflict = (
            len(labels) > 1
        )

        if label_conflict:
            label_conflict_groups.append(
                content_id
            )

        # If at least one annotation marks
        # this content as violation,
        # canonical video is positive.
        canonical_label = (
            1 if 1 in labels else 0
        )

        # Because group is already sorted
        # naturally, first row is the
        # canonical representation.
        representative = group.iloc[0]

        video_ids = (
            group["video_id"]
            .dropna()
            .astype(str)
            .tolist()
        )

        filenames = (
            group["actual_filename"]
            .dropna()
            .astype(str)
            .tolist()
        )

        event_rows = (
            events[
                events[
                    "content_id"
                ].eq(content_id)
            ]
            .sort_values(
                "timecode_sec"
            )
        )

        event_count = len(
            event_rows
        )

        event_timecodes = (
            event_rows[
                "timecode_sec"
            ]
            .astype(float)
            .tolist()
        )

        video_rows.append(
            {
                "content_id": (
                    content_id
                ),

                "canonical_video_id": (
                    representative[
                        "video_id"
                    ]
                ),

                "canonical_filename": (
                    representative[
                        "actual_filename"
                    ]
                ),

                "all_video_ids": "|".join(
                    video_ids
                ),

                "all_filenames": "|".join(
                    filenames
                ),

                "source_row_count": (
                    len(group)
                ),

                "label": (
                    canonical_label
                ),

                "label_conflict": (
                    label_conflict
                ),

                "event_count": (
                    event_count
                ),

                "event_timecodes": (
                    "|".join(
                        str(value)
                        for value
                        in event_timecodes
                    )
                ),

                "duration_sec": (
                    representative[
                        "duration_sec"
                    ]
                ),

                "fps": (
                    representative["fps"]
                ),

                "frame_count": (
                    representative[
                        "frame_count"
                    ]
                ),

                "width": (
                    representative["width"]
                ),

                "height": (
                    representative["height"]
                ),

                "file_size_mb": (
                    representative[
                        "file_size_mb"
                    ]
                ),
            }
        )

    videos = pd.DataFrame(
        video_rows
    )

    # -----------------------------------------------------
    # SORT VIDEOS NATURALLY
    # -----------------------------------------------------

    videos["_sort_id"] = (
        videos[
            "canonical_video_id"
        ]
        .apply(video_number)
    )

    videos = (
        videos
        .sort_values(
            "_sort_id"
        )
        .drop(
            columns=["_sort_id"]
        )
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    positive_videos = int(
        videos["label"]
        .eq(1)
        .sum()
    )

    negative_videos = int(
        videos["label"]
        .eq(0)
        .sum()
    )

    multi_event_videos = int(
        videos["event_count"]
        .gt(1)
        .sum()
    )

    duplicate_groups = int(
        videos["source_row_count"]
        .gt(1)
        .sum()
    )

    event_outside_video = []

    duration_lookup = (
        videos
        .set_index(
            "content_id"
        )[
            "duration_sec"
        ]
        .to_dict()
    )

    for _, event in events.iterrows():
        duration = (
            duration_lookup[
                event["content_id"]
            ]
        )

        if (
            event["timecode_sec"] < 0
            or
            event["timecode_sec"]
            > duration
        ):
            event_outside_video.append(
                event["event_id"]
            )

    # -----------------------------------------------------
    # SAVE FILES
    # -----------------------------------------------------

    videos.to_csv(
        VIDEOS_FILE,
        index=False,
    )

    events.to_csv(
        EVENTS_FILE,
        index=False,
    )

    summary = {
        "metadata_rows": int(
            len(df)
        ),

        "unique_video_contents": int(
            len(videos)
        ),

        "unique_positive_videos": (
            positive_videos
        ),

        "unique_negative_videos": (
            negative_videos
        ),

        "total_violation_events": int(
            len(events)
        ),

        "duplicate_content_groups": (
            duplicate_groups
        ),

        "multi_event_videos": (
            multi_event_videos
        ),

        "label_conflict_groups": int(
            len(
                label_conflict_groups
            )
        ),

        "events_outside_video": int(
            len(
                event_outside_video
            )
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # CONSOLE OUTPUT
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print(
        "CANONICAL DATASET SUMMARY"
    )
    print("=" * 72)

    for key, value in summary.items():
        print(
            f"{key:30} {value}"
        )

    # -----------------------------------------------------
    # DUPLICATES
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print(
        "DUPLICATE CONTENT GROUPS"
    )
    print("=" * 72)

    duplicate_rows = videos[
        videos[
            "source_row_count"
        ].gt(1)
    ]

    if duplicate_rows.empty:
        print(
            "No duplicate groups."
        )

    else:
        print(
            duplicate_rows[
                [
                    "canonical_video_id",
                    "all_video_ids",
                    "label",
                    "event_count",
                    "event_timecodes",
                ]
            ].to_string(
                index=False
            )
        )

    # -----------------------------------------------------
    # MULTI-EVENT VIDEOS
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print(
        "MULTI-EVENT VIDEOS"
    )
    print("=" * 72)

    multi_event_rows = videos[
        videos[
            "event_count"
        ].gt(1)
    ]

    if multi_event_rows.empty:
        print(
            "No multi-event videos."
        )

    else:
        print(
            multi_event_rows[
                [
                    "canonical_video_id",
                    "all_video_ids",
                    "event_count",
                    "event_timecodes",
                ]
            ].to_string(
                index=False
            )
        )

    # -----------------------------------------------------
    # FILES CREATED
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print(
        "FILES CREATED"
    )
    print("=" * 72)

    print(VIDEOS_FILE)
    print(EVENTS_FILE)
    print(SUMMARY_FILE)

    # -----------------------------------------------------
    # FINAL VALIDATION
    # -----------------------------------------------------

    print()

    if label_conflict_groups:
        print(
            "⚠️ Label conflicts detected "
            "inside duplicate groups."
        )
    else:
        print(
            "✅ No label conflicts detected."
        )

    if event_outside_video:
        print(
            "⚠️ Some events are outside "
            "video duration."
        )

        print(
            "Affected events:",
            event_outside_video,
        )
    else:
        print(
            "✅ All event timestamps are valid."
        )


if __name__ == "__main__":
    main()