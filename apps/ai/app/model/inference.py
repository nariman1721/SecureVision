from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch

from app.action_recognition.preprocessing import (
    decode_video_clip,
    select_centered_subwindow,
)

from app.model.preprocessing import (
    MODEL_CLIP_DURATION_SEC,
    preprocess_for_x3d,
)


def optional_float(
    value: Any,
) -> float | None:
    if value is None:
        return None

    try:
        number = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if math.isnan(number):
        return None

    return number


def resolve_video_path(
    path_value: str,
    ai_root: str | Path,
) -> Path:
    path = Path(
        str(path_value)
    )

    if path.is_absolute():
        resolved = path
    else:
        resolved = (
            Path(ai_root)
            / path
        )

    resolved = (
        resolved.resolve()
    )

    if not resolved.exists():
        raise FileNotFoundError(
            f"Video not found: "
            f"{resolved}"
        )

    return resolved


def prepare_model_input(
    video_path: str | Path,
    manifest_start_sec: float,
    manifest_end_sec: float,
    center_sec: float,
    device: (
        torch.device
        | str
        | None
    ) = None,
) -> tuple[
    torch.Tensor,
    dict[str, Any],
]:
    """
    Prepare one manifest window for X3D.

    The manifest window can be 6 seconds,
    while X3D-XS consumes a smaller
    1.6-second subwindow centered around
    the event or negative center.
    """

    (
        model_start_sec,
        model_end_sec,
    ) = select_centered_subwindow(
        outer_start_sec=(
            float(
                manifest_start_sec
            )
        ),
        outer_end_sec=(
            float(
                manifest_end_sec
            )
        ),
        center_sec=float(
            center_sec
        ),
        duration_sec=(
            MODEL_CLIP_DURATION_SEC
        ),
    )

    raw_video = (
        decode_video_clip(
            video_path=video_path,
            start_sec=(
                model_start_sec
            ),
            end_sec=(
                model_end_sec
            ),
        )
    )

    raw_shape = tuple(
        raw_video.shape
    )

    processed = (
        preprocess_for_x3d(
            raw_video
        )
    )

    # Add batch dimension:
    #
    # [C,T,H,W]
    #       ↓
    # [N,C,T,H,W]
    batch = (
        processed
        .unsqueeze(0)
    )

    if device is not None:
        batch = batch.to(
            device
        )

    metadata = {
        "video_path": (
            str(video_path)
        ),

        "manifest_start_sec": (
            float(
                manifest_start_sec
            )
        ),

        "manifest_end_sec": (
            float(
                manifest_end_sec
            )
        ),

        "model_start_sec": (
            model_start_sec
        ),

        "model_end_sec": (
            model_end_sec
        ),

        "model_center_sec": (
            float(
                center_sec
            )
        ),

        "model_duration_sec": (
            round(
                model_end_sec
                - model_start_sec,
                3,
            )
        ),

        "raw_tensor_shape": (
            raw_shape
        ),

        "processed_tensor_shape": (
            tuple(
                processed.shape
            )
        ),

        "batch_shape": (
            tuple(
                batch.shape
            )
        ),
    }

    return (
        batch,
        metadata,
    )


def prepare_model_input_from_manifest(
    row: Mapping[str, Any],
    ai_root: str | Path,
    device: (
        torch.device
        | str
        | None
    ) = None,
) -> tuple[
    torch.Tensor,
    dict[str, Any],
]:
    """
    Prepare one row from train_clips.csv
    or validation_clips.csv.
    """

    required = [
        "video_path",
        "start_sec",
        "end_sec",
        "center_sec",
        "label",
        "clip_type",
    ]

    missing = [
        key
        for key in required
        if key not in row
    ]

    if missing:
        raise ValueError(
            "Manifest row is missing: "
            f"{missing}"
        )

    video_path = (
        resolve_video_path(
            path_value=str(
                row[
                    "video_path"
                ]
            ),
            ai_root=ai_root,
        )
    )

    event_time = optional_float(
        row.get(
            "event_timecode_sec"
        )
    )

    center_sec = optional_float(
        row.get(
            "center_sec"
        )
    )

    if center_sec is None:
        raise ValueError(
            "Manifest center_sec "
            "is missing."
        )

    label = int(
        row["label"]
    )

    # Positive examples should center
    # the X3D temporal window on the
    # annotated event.
    if (
        label == 1
        and event_time is not None
    ):
        model_center_sec = (
            event_time
        )
    else:
        model_center_sec = (
            center_sec
        )

    batch, metadata = (
        prepare_model_input(
            video_path=video_path,
            manifest_start_sec=float(
                row["start_sec"]
            ),
            manifest_end_sec=float(
                row["end_sec"]
            ),
            center_sec=(
                model_center_sec
            ),
            device=device,
        )
    )

    metadata.update(
        {
            "clip_id": (
                str(
                    row.get(
                        "clip_id",
                        "",
                    )
                )
            ),

            "clip_type": (
                str(
                    row.get(
                        "clip_type",
                        "",
                    )
                )
            ),

            "label": (
                label
            ),

            "event_timecode_sec": (
                event_time
            ),
        }
    )

    return (
        batch,
        metadata,
    )


def forward_model(
    model: torch.nn.Module,
    batch: torch.Tensor,
) -> torch.Tensor:
    """
    Generic X3D forward helper.

    IMPORTANT:
    Current loader still uses
    pretrained=False, so output values
    are NOT meaningful predictions yet.
    """

    model.eval()

    with torch.inference_mode():
        output = model(
            batch
        )

    return output