from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.workplace import (
    WorkplaceAnalytics,
)

from app.workplace.zone_monitor import (
    NormalizedWorkZone,
)


AI_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RAW_DIR = (
    AI_ROOT
    / "data"
    / "raw"
)

REPORTS_DIR = (
    AI_ROOT
    / "reports"
)


def resolve_video(
    video_id: str,
) -> Path:
    candidates = [
        RAW_DIR
        / f"{video_id}.mov",

        RAW_DIR
        / (
            video_id.replace(
                "X",
                "Х",
            )
            + ".mov"
        ),
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    raise FileNotFoundError(
        f"Video not found: "
        f"{video_id}"
    )


def main():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--video-id",
        default="X-18",
    )

    args = (
        parser.parse_args()
    )

    video_path = (
        resolve_video(
            args.video_id
        )
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)

    print(
        "SecureVision — "
        "Workplace Analytics Phase 10B"
    )

    print("=" * 72)

    print()
    print(
        "Video:",
        video_path,
    )

    analytics = (
        WorkplaceAnalytics(
            sample_interval_sec=1.0,

            confidence_threshold=0.45,

            tracker_max_distance=0.22,

            tracker_max_missed=3,

            min_absence_sec=2.0,

            work_zone=(
                NormalizedWorkZone(
                    x1=0.18,
                    y1=0.42,
                    x2=0.72,
                    y2=0.72,
                )
            ),
        )
    )

    print(
        "Device:",
        analytics.detector.device,
    )

    print()
    print("ANALYZING")
    print("-" * 72)

    result = (
        analytics.analyze(
            video_path,
            show_progress=True,
        )
    )

    output = (
        REPORTS_DIR
        / (
            "workplace_tracking_"
            f"{args.video_id}.json"
        )
    )

    output.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("WORKER SELECTION")
    print("=" * 72)

    print(
        json.dumps(
            result[
                "worker_selection"
            ],
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        json.dumps(
            result[
                "summary"
            ],
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("TRACKS")
    print("-" * 72)

    for track in (
        result[
            "tracks"
        ]
    ):
        print(
            track
        )

    print()
    print("ABSENCE INTERVALS")
    print("-" * 72)

    if result["absence_intervals"]:
        for interval in result[
            "absence_intervals"
        ]:
            print(
                interval
            )
    else:
        print(
            "No significant absences."
        )

    print()
    print("TIMELINE")
    print("-" * 72)

    for item in result[
        "timeline"
    ]:
        print(
            f"{item['timecode_sec']:>5.1f}s | "
            f"{item['state']:<7} | "
            f"cashier_present="
            f"{item['cashier_present']} | "
            f"active_track="
            f"{item['active_cashier_track_id']} | "
            f"candidates="
            f"{item['cashier_candidate_track_ids']} | "
            f"people="
            f"{item['person_count']} | "
            f"tracks="
            f"{item['visible_track_ids']}"
        )

    print()
    print(
        "Processing time:",
        result[
            "processing_time_sec"
        ],
    )

    print()
    print("OUTPUT")
    print("-" * 72)

    print(
        output
    )

    print()
    print(
        "✅ Cashier-zone presence analysis complete."
    )

    print(
        "✅ ID-switch-resistant presence complete."
    )

    print(
        "✅ Absence analytics complete."
    )


if __name__ == "__main__":
    main()