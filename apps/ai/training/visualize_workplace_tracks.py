from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from app.workplace.detector import (
    PersonDetector,
)

from app.workplace.tracker import (
    PersonTracker,
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
    / "workplace_calibration"
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


def draw_zone(
    frame,
    monitor: ZoneMonitor,
):
    height, width = (
        frame.shape[:2]
    )

    bounds = (
        monitor.pixel_bounds(
            frame_width=width,
            frame_height=height,
        )
    )

    cv2.rectangle(
        frame,
        (
            bounds["x1"],
            bounds["y1"],
        ),
        (
            bounds["x2"],
            bounds["y2"],
        ),
        (
            0,
            255,
            255,
        ),
        2,
    )

    cv2.putText(
        frame,
        "CURRENT WORK ZONE",
        (
            bounds["x1"] + 5,
            max(
                20,
                bounds["y1"] + 20,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (
            0,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )


def draw_person(
    frame,
    person: dict,
    *,
    in_zone: bool,
):
    bbox = person[
        "bbox"
    ]

    x1 = int(
        bbox["x1"]
    )

    y1 = int(
        bbox["y1"]
    )

    x2 = int(
        bbox["x2"]
    )

    y2 = int(
        bbox["y2"]
    )

    track_id = int(
        person[
            "track_id"
        ]
    )

    confidence = float(
        person[
            "confidence"
        ]
    )

    if in_zone:
        color = (
            0,
            255,
            0,
        )
    else:
        color = (
            0,
            0,
            255,
        )

    cv2.rectangle(
        frame,
        (
            x1,
            y1,
        ),
        (
            x2,
            y2,
        ),
        color,
        2,
    )

    label = (
        f"Track {track_id} "
        f"{confidence:.2f} "
        f"{'IN' if in_zone else 'OUT'}"
    )

    cv2.putText(
        frame,
        label,
        (
            x1,
            max(
                20,
                y1 - 8,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )

    center = person[
        "center"
    ]

    cv2.circle(
        frame,
        (
            int(
                center["x"]
            ),
            int(
                center["y"]
            ),
        ),
        5,
        color,
        -1,
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
        default=1.0,
    )

    args = (
        parser.parse_args()
    )

    video_path = (
        resolve_video(
            args.video_id
        )
    )

    output_dir = (
        REPORTS_DIR
        / args.video_id
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = (
        PersonDetector(
            confidence_threshold=0.45,
        )
    )

    tracker = (
        PersonTracker(
            max_distance=0.22,
            max_missed=3,
        )
    )

    zone_monitor = (
        ZoneMonitor(
            NormalizedWorkZone(
                x1=0.18,
                y1=0.42,
                x2=0.72,
                y2=0.72,
            )
        )
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

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    duration_sec = (
        frame_count
        / fps
    )

    sample_times = []

    current = 0.0

    while (
        current
        < duration_sec
    ):
        sample_times.append(
            current
        )

        current += (
            args.sample_interval
        )

    # Save several representative frames.
    desired_times = {
        0.0,
        5.0,
        10.0,
        12.0,
        15.0,
        18.0,
        22.0,
    }

    saved_frames = []

    print("=" * 72)
    print(
        "SecureVision — "
        "Workplace Zone Calibration"
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

    try:
        for index, timestamp in enumerate(
            sample_times,
            start=1,
        ):
            print(
                f"\rtracking: "
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
                continue

            detections = (
                detector.detect(
                    frame
                )
            )

            tracked_people = (
                tracker.update(
                    detections,
                    timestamp_sec=timestamp,
                    frame_width=width,
                    frame_height=height,
                )
            )

            should_save = any(
                abs(
                    timestamp
                    - target
                )
                < 0.01
                for target
                in desired_times
            )

            if not should_save:
                continue

            annotated = (
                frame.copy()
            )

            draw_zone(
                annotated,
                zone_monitor,
            )

            for person in (
                tracked_people
            ):
                in_zone = (
                    zone_monitor
                    .detection_inside_zone(
                        person,
                        frame_width=width,
                        frame_height=height,
                    )
                )

                draw_person(
                    annotated,
                    person,
                    in_zone=in_zone,
                )

            cv2.putText(
                annotated,
                (
                    f"Time: "
                    f"{timestamp:.1f}s"
                ),
                (
                    15,
                    30,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (
                    255,
                    255,
                    255,
                ),
                2,
                cv2.LINE_AA,
            )

            filename = (
                output_dir
                / (
                    f"frame_"
                    f"{timestamp:05.1f}s.jpg"
                )
            )

            cv2.imwrite(
                str(
                    filename
                ),
                annotated,
            )

            saved_frames.append(
                filename
            )

    finally:
        capture.release()

    print()
    print()

    print(
        "Saved frames:",
        len(
            saved_frames
        ),
    )

    for path in (
        saved_frames
    ):
        print(
            path
        )

    print()
    print(
        "✅ Calibration frames created."
    )


if __name__ == "__main__":
    main()