from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from app.workplace.detector import (
    PersonDetector,
)

from app.workplace.posture import (
    PostureEstimator,
)

from app.workplace.zone_monitor import (
    NormalizedWorkZone,
    ZoneMonitor,
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
        f"Video not found: {video_id}"
    )


def choose_cashier_candidate(
    persons_in_zone: list[dict],
) -> dict | None:
    if not persons_in_zone:
        return None

    return max(
        persons_in_zone,
        key=lambda person: (
            person[
                "confidence"
            ]
        ),
    )


def main():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--video-id",
        default="X-18",
    )

    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.5,
    )

    args = (
        parser.parse_args()
    )

    video_path = (
        resolve_video(
            args.video_id
        )
    )

    detector = (
        PersonDetector(
            confidence_threshold=0.45,
        )
    )

    cashier_zone = (
        ZoneMonitor(
            NormalizedWorkZone(
                x1=0.18,
                y1=0.42,
                x2=0.72,
                y2=0.72,
            )
        )
    )

    posture_estimator = (
        PostureEstimator()
    )

    capture = cv2.VideoCapture(
        str(
            video_path
        )
    )

    if not capture.isOpened():
        raise RuntimeError(
            "Unable to open video."
        )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    frame_width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    frame_height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    if fps <= 0:
        capture.release()

        raise RuntimeError(
            "Invalid FPS."
        )

    duration_sec = (
        frame_count
        / fps
    )

    sample_interval = float(
        args.sample_interval
    )

    sample_times = []

    current = 0.0

    while current < duration_sec:
        sample_times.append(
            current
        )

        current += (
            sample_interval
        )

    timeline = []

    print("=" * 72)

    print(
        "SecureVision — "
        "Workplace Posture Phase 10C"
    )

    print("=" * 72)

    print()
    print(
        "Video:",
        video_path,
    )

    print(
        "Device:",
        detector.device,
    )

    print(
        "Sample interval:",
        sample_interval,
    )

    print()
    print("ANALYZING")
    print("-" * 72)

    try:
        for index, timestamp in enumerate(
            sample_times,
            start=1,
        ):
            print(
                f"\rposture scan: "
                f"{index}/"
                f"{len(sample_times)}",
                end="",
                flush=True,
            )

            capture.set(
                cv2.CAP_PROP_POS_MSEC,
                timestamp
                * 1000.0,
            )

            success, frame = (
                capture.read()
            )

            if (
                not success
                or frame is None
            ):
                timeline.append(
                    {
                        "timecode_sec": float(
                            round(
                                timestamp,
                                3,
                            )
                        ),

                        "state": (
                            "unknown"
                        ),

                        "posture": (
                            "unknown"
                        ),

                        "cashier_present": (
                            False
                        ),

                        "detector_confidence": (
                            None
                        ),

                        "posture_confidence": (
                            None
                        ),

                        "features": None,
                    }
                )

                continue

            detections = (
                detector.detect(
                    frame
                )
            )

            persons_in_zone = [
                detection
                for detection
                in detections
                if (
                    cashier_zone
                    .detection_inside_zone(
                        detection,

                        frame_width=(
                            frame_width
                        ),

                        frame_height=(
                            frame_height
                        ),
                    )
                )
            ]

            cashier = (
                choose_cashier_candidate(
                    persons_in_zone
                )
            )

            if cashier is None:
                timeline.append(
                    {
                        "timecode_sec": float(
                            round(
                                timestamp,
                                3,
                            )
                        ),

                        "state": (
                            "away"
                        ),

                        "posture": (
                            "away"
                        ),

                        "cashier_present": (
                            False
                        ),

                        "detector_confidence": (
                            None
                        ),

                        "posture_confidence": (
                            None
                        ),

                        "features": None,
                    }
                )

                continue

            posture_result = (
                posture_estimator.classify(
                    cashier,

                    frame_width=(
                        frame_width
                    ),

                    frame_height=(
                        frame_height
                    ),
                )
            )

            timeline.append(
                {
                    "timecode_sec": float(
                        round(
                            timestamp,
                            3,
                        )
                    ),

                    "state": (
                        "present"
                    ),

                    "posture": (
                        posture_result[
                            "posture"
                        ]
                    ),

                    "cashier_present": (
                        True
                    ),

                    "detector_confidence": float(
                        round(
                            cashier[
                                "confidence"
                            ],
                            4,
                        )
                    ),

                    "posture_confidence": (
                        posture_result[
                            "heuristic_confidence"
                        ]
                    ),

                    "features": (
                        posture_result[
                            "features"
                        ]
                    ),
                }
            )

    finally:
        capture.release()

        print()

    # -----------------------------------------------------
    # Convert sampled states to time durations
    # -----------------------------------------------------

    sitting_sec = 0.0
    standing_sec = 0.0
    away_sec = 0.0
    unknown_sec = 0.0

    for index, item in enumerate(
        timeline
    ):
        start = float(
            item[
                "timecode_sec"
            ]
        )

        if (
            index + 1
            < len(
                timeline
            )
        ):
            end = float(
                timeline[
                    index + 1
                ][
                    "timecode_sec"
                ]
            )

        else:
            end = float(
                duration_sec
            )

        duration = max(
            0.0,
            end - start,
        )

        item[
            "sample_duration_sec"
        ] = float(
            round(
                duration,
                3,
            )
        )

        posture = (
            item[
                "posture"
            ]
        )

        if posture == "sitting":
            sitting_sec += duration

        elif posture == "standing":
            standing_sec += duration

        elif posture == "away":
            away_sec += duration

        else:
            unknown_sec += duration

    present_sec = (
        sitting_sec
        + standing_sec
        + unknown_sec
    )

    posture_known_sec = (
        sitting_sec
        + standing_sec
    )

    if posture_known_sec > 0:
        sitting_percent = (
            sitting_sec
            / posture_known_sec
            * 100.0
        )

        standing_percent = (
            standing_sec
            / posture_known_sec
            * 100.0
        )

    else:
        sitting_percent = 0.0
        standing_percent = 0.0

    summary = {
        "duration_sec": float(
            round(
                duration_sec,
                3,
            )
        ),

        "present_sec": float(
            round(
                present_sec,
                3,
            )
        ),

        "sitting_sec": float(
            round(
                sitting_sec,
                3,
            )
        ),

        "standing_sec": float(
            round(
                standing_sec,
                3,
            )
        ),

        "away_sec": float(
            round(
                away_sec,
                3,
            )
        ),

        "unknown_posture_sec": float(
            round(
                unknown_sec,
                3,
            )
        ),

        "sitting_percent_of_known_posture": float(
            round(
                sitting_percent,
                2,
            )
        ),

        "standing_percent_of_known_posture": float(
            round(
                standing_percent,
                2,
            )
        ),

        "samples": int(
            len(
                timeline
            )
        ),
    }

    result = {
        "status": (
            "done"
        ),

        "file_name": (
            video_path.name
        ),

        "method": (
            "camera_calibrated_bbox_geometry"
        ),

        "configuration": {
            "sample_interval_sec": (
                sample_interval
            ),

            "cashier_zone": {
                "x1": 0.18,
                "y1": 0.42,
                "x2": 0.72,
                "y2": 0.72,
            },

            "posture_thresholds": {
                "standing_height_ratio": (
                    posture_estimator
                    .thresholds
                    .standing_height_ratio
                ),

                "sitting_height_ratio": (
                    posture_estimator
                    .thresholds
                    .sitting_height_ratio
                ),

                "standing_top_y_ratio": (
                    posture_estimator
                    .thresholds
                    .standing_top_y_ratio
                ),

                "sitting_top_y_ratio": (
                    posture_estimator
                    .thresholds
                    .sitting_top_y_ratio
                ),
            },
        },

        "summary": (
            summary
        ),

        "timeline": (
            timeline
        ),
    }

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        REPORTS_DIR
        / (
            "workplace_posture_"
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
    print("SUMMARY")
    print("=" * 72)

    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("TIMELINE")
    print("-" * 72)

    for item in timeline:
        features = (
            item[
                "features"
            ]
            or {}
        )

        print(
            f"{item['timecode_sec']:>5.1f}s | "
            f"{item['posture']:<8} | "
            f"det="
            f"{item['detector_confidence']} | "
            f"posture_conf="
            f"{item['posture_confidence']} | "
            f"height="
            f"{features.get('height_ratio')} | "
            f"top_y="
            f"{features.get('top_y_ratio')}"
        )

    print()
    print("OUTPUT")
    print("-" * 72)

    print(
        output
    )

    print()
    print(
        "✅ Sitting / standing "
        "analysis complete."
    )


if __name__ == "__main__":
    main()