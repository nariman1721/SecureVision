from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.workplace import (
    ProductionWorkplaceAnalytics,
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


def check_close(
    value_a: float,
    value_b: float,
    *,
    tolerance: float = 0.01,
) -> bool:
    return (
        abs(
            value_a
            - value_b
        )
        <= tolerance
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
        "Workplace Production Phase 10D"
    )
    print("=" * 72)

    print()
    print(
        "Video:",
        video_path,
    )

    analyzer = (
        ProductionWorkplaceAnalytics(
            sample_interval_sec=0.5,
            confidence_threshold=0.45,
            min_absence_sec=2.0,
        )
    )

    print(
        "Device:",
        analyzer.detector.device,
    )

    print()
    print("ANALYZING")
    print("-" * 72)

    result = (
        analyzer.analyze(
            video_path,
            show_progress=True,
        )
    )

    output = (
        REPORTS_DIR
        / (
            "workplace_production_"
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

    summary = (
        result[
            "summary"
        ]
    )

    duration = float(
        summary[
            "duration_sec"
        ]
    )

    presence_total = (
        float(
            summary[
                "present_sec"
            ]
        )
        + float(
            summary[
                "away_sec"
            ]
        )
        + float(
            summary[
                "unknown_sec"
            ]
        )
    )

    posture_total = (
        float(
            summary[
                "sitting_sec"
            ]
        )
        + float(
            summary[
                "standing_sec"
            ]
        )
        + float(
            summary[
                "unknown_posture_sec"
            ]
        )
    )

    presence_math_valid = (
        check_close(
            duration,
            presence_total,
        )
    )

    posture_math_valid = (
        check_close(
            float(
                summary[
                    "present_sec"
                ]
            ),
            posture_total,
        )
    )

    timeline_valid = (
        len(
            result[
                "timeline"
            ]
        )
        == int(
            summary[
                "samples"
            ]
        )
    )

    methodology_valid = (
        result[
            "methodology"
        ][
            "face_recognition"
        ]
        is False
        and result[
            "methodology"
        ][
            "biometric_identification"
        ]
        is False
        and result[
            "methodology"
        ][
            "id_switch_resistant_presence"
        ]
        is True
    )

    print()
    print("=" * 72)
    print("FINAL PRODUCTION SUMMARY")
    print("=" * 72)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("VALIDATION")
    print("-" * 72)

    print(
        "Presence time math:",
        presence_math_valid,
    )

    print(
        "Posture time math:",
        posture_math_valid,
    )

    print(
        "Timeline sample count:",
        timeline_valid,
    )

    print(
        "Privacy methodology:",
        methodology_valid,
    )

    print()
    print("METHODOLOGY")
    print("-" * 72)

    print(
        json.dumps(
            result[
                "methodology"
            ],
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("ABSENCE INTERVALS")
    print("-" * 72)

    if result[
        "absence_intervals"
    ]:
        for interval in (
            result[
                "absence_intervals"
            ]
        ):
            print(
                interval
            )

    else:
        print(
            "No significant absences."
        )

    print()
    print("TRACKING")
    print("-" * 72)

    print(
        "Primary track candidate:",
        result[
            "tracking"
        ][
            "primary_track_candidate"
        ],
    )

    for track in (
        result[
            "tracking"
        ][
            "tracks"
        ]
    ):
        print(
            track
        )

    print()
    print("TIMELINE")
    print("-" * 72)

    for item in (
        result[
            "timeline"
        ]
    ):
        print(
            f"{item['timecode_sec']:>5.1f}s | "
            f"{item['state']:<7} | "
            f"{item['posture']:<8} | "
            f"track="
            f"{item['active_cashier_track_id']} | "
            f"people="
            f"{item['person_count']} | "
            f"det="
            f"{item['detector_confidence']} | "
            f"posture_conf="
            f"{item['posture_confidence']}"
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

    if (
        presence_math_valid
        and posture_math_valid
        and timeline_valid
        and methodology_valid
    ):
        print(
            "✅ Unified production JSON valid."
        )

        print(
            "✅ Presence + Absence + "
            "Posture integrated."
        )

        print(
            "✅ Phase 10D complete."
        )

    else:
        raise RuntimeError(
            "Production JSON validation failed."
        )


if __name__ == "__main__":
    main()