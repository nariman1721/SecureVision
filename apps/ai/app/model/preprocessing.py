from __future__ import annotations

import torch
import torch.nn.functional as F


# =========================================================
# X3D-XS OFFICIAL-LIKE INPUT CONFIG
# =========================================================

MODEL_NAME = "x3d_xs"

NUM_FRAMES = 4

SAMPLING_RATE = 12

ASSUMED_FPS = 30.0

SIDE_SIZE = 182

CROP_SIZE = 182

RGB_MEAN = (
    0.45,
    0.45,
    0.45,
)

RGB_STD = (
    0.225,
    0.225,
    0.225,
)

MODEL_CLIP_DURATION_SEC = (
    NUM_FRAMES
    * SAMPLING_RATE
    / ASSUMED_FPS
)

EXPECTED_INPUT_SHAPE = (
    3,
    NUM_FRAMES,
    CROP_SIZE,
    CROP_SIZE,
)


# =========================================================
# VALIDATION
# =========================================================

def validate_video_tensor(
    video: torch.Tensor,
) -> None:
    if not isinstance(
        video,
        torch.Tensor,
    ):
        raise TypeError(
            "video must be a torch.Tensor"
        )

    if video.ndim != 4:
        raise ValueError(
            "Video tensor must have "
            "shape [C, T, H, W]. "
            f"Received: {tuple(video.shape)}"
        )

    channels = (
        video.shape[0]
    )

    if channels != 3:
        raise ValueError(
            "Expected 3 RGB channels. "
            f"Received: {channels}"
        )

    if video.shape[1] <= 0:
        raise ValueError(
            "Video contains no frames."
        )


# =========================================================
# TEMPORAL SUBSAMPLING
# =========================================================

def uniform_temporal_subsample(
    video: torch.Tensor,
    num_frames: int = NUM_FRAMES,
) -> torch.Tensor:
    """
    Uniformly select num_frames from
    [C, T, H, W].
    """

    validate_video_tensor(
        video
    )

    total_frames = int(
        video.shape[1]
    )

    if num_frames <= 0:
        raise ValueError(
            "num_frames must be positive"
        )

    indices = torch.linspace(
        0,
        total_frames - 1,
        steps=num_frames,
        device=video.device,
    )

    indices = (
        indices
        .round()
        .long()
    )

    return video.index_select(
        1,
        indices,
    )


# =========================================================
# NORMALIZATION
# =========================================================

def convert_to_float(
    video: torch.Tensor,
) -> torch.Tensor:
    if video.dtype == torch.uint8:
        return (
            video.float()
            / 255.0
        )

    video = video.float()

    # Defensive fallback if floating
    # tensor still contains 0..255 values.
    if video.numel() > 0:
        max_value = float(
            video.max().item()
        )

        if max_value > 1.5:
            video = (
                video / 255.0
            )

    return video


def normalize_rgb(
    video: torch.Tensor,
) -> torch.Tensor:
    mean = torch.tensor(
        RGB_MEAN,
        dtype=video.dtype,
        device=video.device,
    ).view(
        3,
        1,
        1,
        1,
    )

    std = torch.tensor(
        RGB_STD,
        dtype=video.dtype,
        device=video.device,
    ).view(
        3,
        1,
        1,
        1,
    )

    return (
        video - mean
    ) / std


# =========================================================
# SPATIAL TRANSFORMS
# =========================================================

def short_side_scale(
    video: torch.Tensor,
    size: int = SIDE_SIZE,
) -> torch.Tensor:
    """
    Resize video while preserving aspect ratio.

    Input:
        [C, T, H, W]

    Output:
        [C, T, new_H, new_W]

    Shortest spatial side becomes `size`.
    """

    validate_video_tensor(
        video
    )

    _, _, height, width = (
        video.shape
    )

    if height <= 0 or width <= 0:
        raise ValueError(
            "Invalid spatial dimensions."
        )

    if height < width:
        new_height = size

        new_width = int(
            round(
                width
                * size
                / height
            )
        )

    else:
        new_width = size

        new_height = int(
            round(
                height
                * size
                / width
            )
        )

    # F.interpolate expects:
    # [N, C, H, W]
    frames = video.permute(
        1,
        0,
        2,
        3,
    )

    frames = F.interpolate(
        frames,
        size=(
            new_height,
            new_width,
        ),
        mode="bilinear",
        align_corners=False,
    )

    return (
        frames
        .permute(
            1,
            0,
            2,
            3,
        )
        .contiguous()
    )


def center_crop(
    video: torch.Tensor,
    crop_size: int = CROP_SIZE,
) -> torch.Tensor:
    validate_video_tensor(
        video
    )

    _, _, height, width = (
        video.shape
    )

    if (
        height < crop_size
        or width < crop_size
    ):
        raise ValueError(
            "Video is smaller than "
            "requested crop size. "
            f"Video: {height}x{width}, "
            f"crop: {crop_size}"
        )

    top = (
        height - crop_size
    ) // 2

    left = (
        width - crop_size
    ) // 2

    return video[
        :,
        :,
        top:top + crop_size,
        left:left + crop_size,
    ].contiguous()


# =========================================================
# FULL X3D PREPROCESSING
# =========================================================

def preprocess_for_x3d(
    video: torch.Tensor,
) -> torch.Tensor:
    """
    Convert decoded RGB video tensor into
    X3D-XS input.

    Input:
        [3, T, H, W]
        uint8 or float

    Output:
        [3, 4, 182, 182]
        float32
    """

    validate_video_tensor(
        video
    )

    video = (
        uniform_temporal_subsample(
            video,
            NUM_FRAMES,
        )
    )

    video = convert_to_float(
        video
    )

    video = normalize_rgb(
        video
    )

    video = short_side_scale(
        video,
        SIDE_SIZE,
    )

    video = center_crop(
        video,
        CROP_SIZE,
    )

    video = (
        video
        .float()
        .contiguous()
    )

    actual_shape = tuple(
        video.shape
    )

    if (
        actual_shape
        != EXPECTED_INPUT_SHAPE
    ):
        raise RuntimeError(
            "Unexpected X3D tensor shape. "
            f"Expected "
            f"{EXPECTED_INPUT_SHAPE}, "
            f"received {actual_shape}"
        )

    return video