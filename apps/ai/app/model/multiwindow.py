from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

import torch

from app.model.features import (
    X3D_FEATURE_DIM,
    extract_x3d_features,
)

from app.model.inference import (
    prepare_model_input,
    resolve_video_path,
)

from app.model.preprocessing import (
    MODEL_CLIP_DURATION_SEC,
)


DEFAULT_NUM_WINDOWS = 3

PoolingMode = Literal[
    "mean",
    "max",
]


# =========================================================
# WINDOW POSITIONS
# =========================================================

def compute_window_centers(
    manifest_start_sec: float,
    manifest_end_sec: float,
    num_windows: int = DEFAULT_NUM_WINDOWS,
    model_duration_sec: float = MODEL_CLIP_DURATION_SEC,
) -> list[float]:
    """
    Create evenly spaced X3D window centers inside
    the full manifest interval.

    Example for a 6-second manifest:

        manifest: 54.0 -> 60.0
        X3D duration: 1.6 sec

        centers:
            54.8
            57.0
            59.2

        windows:
            54.0 -> 55.6
            56.2 -> 57.8
            58.4 -> 60.0

    IMPORTANT:
    This function does not use label or event timestamp.
    Positive and negative samples are processed identically.
    """

    start_sec = float(
        manifest_start_sec
    )

    end_sec = float(
        manifest_end_sec
    )

    model_duration_sec = float(
        model_duration_sec
    )

    if start_sec < 0:
        raise ValueError(
            "manifest_start_sec "
            "cannot be negative."
        )

    if end_sec <= start_sec:
        raise ValueError(
            "manifest_end_sec must be "
            "greater than manifest_start_sec."
        )

    if num_windows <= 0:
        raise ValueError(
            "num_windows must be positive."
        )

    if model_duration_sec <= 0:
        raise ValueError(
            "model_duration_sec must be positive."
        )

    manifest_duration = (
        end_sec
        - start_sec
    )

    midpoint = (
        start_sec
        + manifest_duration / 2.0
    )

    # If the manifest itself is shorter than
    # one X3D temporal window, all windows
    # point to the same center.
    if (
        manifest_duration
        <= model_duration_sec
    ):
        return [
            round(
                midpoint,
                3,
            )
            for _ in range(
                num_windows
            )
        ]

    half_window = (
        model_duration_sec / 2.0
    )

    first_center = (
        start_sec
        + half_window
    )

    last_center = (
        end_sec
        - half_window
    )

    if num_windows == 1:
        return [
            round(
                midpoint,
                3,
            )
        ]

    step = (
        last_center
        - first_center
    ) / (
        num_windows - 1
    )

    centers = [
        round(
            first_center
            + index * step,
            3,
        )
        for index in range(
            num_windows
        )
    ]

    return centers


# =========================================================
# PREPARE MULTI-WINDOW BATCH
# =========================================================

def prepare_multiwindow_batch_from_manifest(
    row: Mapping[str, Any],
    ai_root: str | Path,
    device: (
        torch.device
        | str
        | None
    ) = None,
    num_windows: int = DEFAULT_NUM_WINDOWS,
) -> tuple[
    torch.Tensor,
    dict[str, Any],
]:
    """
    Convert one 6-second manifest row into multiple
    X3D inputs.

    Output batch shape:

        [W, C, T, H, W]

    For the current configuration:

        [3, 3, 4, 182, 182]
    """

    required = [
        "video_path",
        "start_sec",
        "end_sec",
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

    manifest_start_sec = float(
        row[
            "start_sec"
        ]
    )

    manifest_end_sec = float(
        row[
            "end_sec"
        ]
    )

    video_path = resolve_video_path(
        path_value=str(
            row[
                "video_path"
            ]
        ),
        ai_root=ai_root,
    )

    centers = compute_window_centers(
        manifest_start_sec=(
            manifest_start_sec
        ),
        manifest_end_sec=(
            manifest_end_sec
        ),
        num_windows=(
            num_windows
        ),
    )

    batches = []

    windows_metadata = []

    for index, center_sec in enumerate(
        centers,
        start=1,
    ):
        batch, metadata = (
            prepare_model_input(
                video_path=video_path,
                manifest_start_sec=(
                    manifest_start_sec
                ),
                manifest_end_sec=(
                    manifest_end_sec
                ),
                center_sec=(
                    center_sec
                ),
                device=device,
            )
        )

        batches.append(
            batch
        )

        windows_metadata.append(
            {
                "window_index": (
                    index
                ),

                "center_sec": float(
                    center_sec
                ),

                "start_sec": float(
                    metadata[
                        "model_start_sec"
                    ]
                ),

                "end_sec": float(
                    metadata[
                        "model_end_sec"
                    ]
                ),

                "duration_sec": float(
                    metadata[
                        "model_duration_sec"
                    ]
                ),
            }
        )

    multiwindow_batch = torch.cat(
        batches,
        dim=0,
    )

    expected_batch_size = (
        len(
            centers
        )
    )

    if (
        multiwindow_batch.ndim
        != 5
    ):
        raise RuntimeError(
            "Expected multi-window batch "
            "[W,C,T,H,W]. "
            f"Received "
            f"{tuple(multiwindow_batch.shape)}"
        )

    if (
        multiwindow_batch.shape[0]
        != expected_batch_size
    ):
        raise RuntimeError(
            "Unexpected number of "
            "temporal windows."
        )

    metadata = {
        "clip_id": str(
            row.get(
                "clip_id",
                "",
            )
        ),

        "canonical_video_id": str(
            row.get(
                "canonical_video_id",
                "",
            )
        ),

        "clip_type": str(
            row.get(
                "clip_type",
                "",
            )
        ),

        "label": int(
            row.get(
                "label",
                -1,
            )
        ),

        "video_path": str(
            video_path
        ),

        "manifest_start_sec": (
            manifest_start_sec
        ),

        "manifest_end_sec": (
            manifest_end_sec
        ),

        "manifest_duration_sec": (
            round(
                manifest_end_sec
                - manifest_start_sec,
                3,
            )
        ),

        "num_windows": (
            expected_batch_size
        ),

        "model_window_duration_sec": (
            MODEL_CLIP_DURATION_SEC
        ),

        "window_centers_sec": (
            centers
        ),

        "windows": (
            windows_metadata
        ),

        "batch_shape": tuple(
            multiwindow_batch.shape
        ),
    }

    return (
        multiwindow_batch,
        metadata,
    )


# =========================================================
# FEATURE POOLING
# =========================================================

def pool_window_features(
    window_features: torch.Tensor,
    pooling: PoolingMode = "mean",
) -> torch.Tensor:
    """
    Input:

        [num_windows, 2048]

    Output:

        [1, 2048]
    """

    if window_features.ndim != 2:
        raise ValueError(
            "Expected window features "
            "with shape [W, 2048]. "
            f"Received "
            f"{tuple(window_features.shape)}"
        )

    if (
        window_features.shape[1]
        != X3D_FEATURE_DIM
    ):
        raise ValueError(
            "Unexpected feature dimension."
        )

    if pooling == "mean":
        pooled = (
            window_features
            .mean(
                dim=0,
                keepdim=True,
            )
        )

    elif pooling == "max":
        pooled = (
            window_features
            .max(
                dim=0,
                keepdim=True,
            )
            .values
        )

    else:
        raise ValueError(
            f"Unsupported pooling: "
            f"{pooling}"
        )

    if not torch.isfinite(
        pooled
    ).all():
        raise RuntimeError(
            "Pooled features contain "
            "NaN or Inf."
        )

    return pooled


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_multiwindow_feature_set(
    model: torch.nn.Module,
    multiwindow_batch: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Run all temporal windows through X3D only once.

    Returns:

        windows : [W, 2048]
        mean    : [1, 2048]
        max     : [1, 2048]

    We save both mean and max so later CV can compare
    pooling strategies without running X3D again.
    """

    window_features = (
        extract_x3d_features(
            model,
            multiwindow_batch,
        )
    )

    mean_features = (
        pool_window_features(
            window_features,
            pooling="mean",
        )
    )

    max_features = (
        pool_window_features(
            window_features,
            pooling="max",
        )
    )

    return {
        "windows": (
            window_features
        ),

        "mean": (
            mean_features
        ),

        "max": (
            max_features
        ),
    }