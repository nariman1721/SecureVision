from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2


DEFAULT_OFFSETS_SEC = (
    -0.75,
    0.0,
    0.75,
)

DEFAULT_JPEG_QUALITY = 85
DEFAULT_MAX_WIDTH = 960


def resize_frame(
    frame,
    max_width: int = DEFAULT_MAX_WIDTH,
):
    height, width = frame.shape[:2]

    if width <= max_width:
        return frame

    scale = (
        max_width
        / float(width)
    )

    new_width = int(
        round(
            width * scale
        )
    )

    new_height = int(
        round(
            height * scale
        )
    )

    return cv2.resize(
        frame,
        (
            new_width,
            new_height,
        ),
        interpolation=cv2.INTER_AREA,
    )


def frame_to_base64_jpeg(
    frame,
    *,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    max_width: int = DEFAULT_MAX_WIDTH,
) -> str:
    frame = resize_frame(
        frame,
        max_width=max_width,
    )

    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            int(
                jpeg_quality
            ),
        ],
    )

    if not success:
        raise RuntimeError(
            "Failed to encode JPEG frame."
        )

    return base64.b64encode(
        buffer.tobytes()
    ).decode(
        "ascii"
    )


class EvidenceExtractor:
    def __init__(
        self,
        *,
        offsets_sec: tuple[
            float,
            ...,
        ] = DEFAULT_OFFSETS_SEC,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        max_width: int = DEFAULT_MAX_WIDTH,
    ):
        self.offsets_sec = tuple(
            float(value)
            for value
            in offsets_sec
        )

        self.jpeg_quality = int(
            jpeg_quality
        )

        self.max_width = int(
            max_width
        )

    def extract(
        self,
        video_path: str | Path,
        timecode_sec: float,
    ) -> list[dict[str, Any]]:
        video_path = Path(
            video_path
        ).resolve()

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: "
                f"{video_path}"
            )

        capture = cv2.VideoCapture(
            str(
                video_path
            )
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Unable to open video: "
                f"{video_path}"
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

        if fps > 0:
            duration_sec = (
                frame_count
                / fps
            )
        else:
            duration_sec = 0.0

        results = []

        seen_times = set()

        try:
            for offset in (
                self.offsets_sec
            ):
                target_sec = float(
                    timecode_sec
                    + offset
                )

                if duration_sec > 0:
                    target_sec = min(
                        max(
                            target_sec,
                            0.0,
                        ),
                        max(
                            duration_sec
                            - 0.001,
                            0.0,
                        ),
                    )
                else:
                    target_sec = max(
                        target_sec,
                        0.0,
                    )

                rounded_target = round(
                    target_sec,
                    3,
                )

                if (
                    rounded_target
                    in seen_times
                ):
                    continue

                seen_times.add(
                    rounded_target
                )

                capture.set(
                    cv2.CAP_PROP_POS_MSEC,
                    target_sec
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

                actual_msec = float(
                    capture.get(
                        cv2.CAP_PROP_POS_MSEC
                    )
                )

                if actual_msec > 0:
                    actual_sec = (
                        actual_msec
                        / 1000.0
                    )
                else:
                    actual_sec = (
                        target_sec
                    )

                encoded = (
                    frame_to_base64_jpeg(
                        frame,
                        jpeg_quality=(
                            self.jpeg_quality
                        ),
                        max_width=(
                            self.max_width
                        ),
                    )
                )

                results.append(
                    {
                        "timecode_sec": float(
                            round(
                                actual_sec,
                                3,
                            )
                        ),

                        "offset_sec": float(
                            offset
                        ),

                        "mime_type": (
                            "image/jpeg"
                        ),

                        "base64": (
                            encoded
                        ),
                    }
                )

        finally:
            capture.release()

        if not results:
            raise RuntimeError(
                "Unable to extract "
                "evidence frames."
            )

        return results[:3]