from __future__ import annotations

import math
from pathlib import Path

import cv2
import torch


def validate_time_window(
    start_sec: float,
    end_sec: float,
) -> None:
    if start_sec < 0:
        raise ValueError(
            "start_sec cannot be negative"
        )

    if end_sec <= start_sec:
        raise ValueError(
            "end_sec must be greater "
            "than start_sec"
        )


def select_centered_subwindow(
    outer_start_sec: float,
    outer_end_sec: float,
    center_sec: float,
    duration_sec: float,
) -> tuple[float, float]:
    """
    Select a smaller temporal window inside
    a larger manifest window.

    Example:

        manifest:
        7.0 -------- 13.0

        event:
                10.0

        X3D window:
              9.2 -- 10.8

    The result is always kept inside the
    outer manifest window.
    """

    validate_time_window(
        outer_start_sec,
        outer_end_sec,
    )

    if duration_sec <= 0:
        raise ValueError(
            "duration_sec must be positive"
        )

    outer_duration = (
        outer_end_sec
        - outer_start_sec
    )

    if outer_duration <= duration_sec:
        return (
            round(
                outer_start_sec,
                3,
            ),
            round(
                outer_end_sec,
                3,
            ),
        )

    half_duration = (
        duration_sec / 2.0
    )

    start_sec = (
        center_sec
        - half_duration
    )

    end_sec = (
        center_sec
        + half_duration
    )

    if start_sec < outer_start_sec:
        start_sec = outer_start_sec
        end_sec = (
            start_sec
            + duration_sec
        )

    if end_sec > outer_end_sec:
        end_sec = outer_end_sec
        start_sec = (
            end_sec
            - duration_sec
        )

    start_sec = max(
        start_sec,
        outer_start_sec,
    )

    end_sec = min(
        end_sec,
        outer_end_sec,
    )

    return (
        round(
            float(start_sec),
            3,
        ),
        round(
            float(end_sec),
            3,
        ),
    )


def decode_video_clip(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
) -> torch.Tensor:
    """
    Decode a temporal section of a video.

    Returns:
        torch.Tensor
        shape: [C, T, H, W]
        dtype: torch.uint8
        color: RGB
    """

    validate_time_window(
        start_sec,
        end_sec,
    )

    video_path = Path(
        video_path
    )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: "
            f"{video_path}"
        )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise ValueError(
            f"Unable to open video: "
            f"{video_path}"
        )

    try:
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

        if fps <= 0:
            raise ValueError(
                f"Invalid FPS for video: "
                f"{video_path}"
            )

        if frame_count <= 0:
            raise ValueError(
                "Video contains no frames: "
                f"{video_path}"
            )

        video_duration = (
            frame_count / fps
        )

        if start_sec >= video_duration:
            raise ValueError(
                f"start_sec={start_sec} "
                f"is outside video duration "
                f"{video_duration:.3f}"
            )

        end_sec = min(
            float(end_sec),
            video_duration,
        )

        start_frame = max(
            0,
            int(
                math.floor(
                    start_sec * fps
                )
            ),
        )

        end_frame = min(
            frame_count,
            int(
                math.ceil(
                    end_sec * fps
                )
            ),
        )

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame,
        )

        frames = []

        current_frame = (
            start_frame
        )

        while (
            current_frame
            < end_frame
        ):
            success, frame = (
                capture.read()
            )

            if not success:
                break

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            frame_tensor = (
                torch.from_numpy(
                    frame
                )
            )

            frames.append(
                frame_tensor
            )

            current_frame += 1

    finally:
        capture.release()

    if not frames:
        raise ValueError(
            "No frames decoded from "
            f"{video_path} between "
            f"{start_sec}s and {end_sec}s"
        )

    # [T, H, W, C]
    video = torch.stack(
        frames,
        dim=0,
    )

    # [C, T, H, W]
    video = (
        video
        .permute(
            3,
            0,
            1,
            2,
        )
        .contiguous()
    )

    return video