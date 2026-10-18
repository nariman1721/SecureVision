from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from app.model.loader import (
    load_model,
)

from app.model.multiwindow import (
    DEFAULT_NUM_WINDOWS,
    extract_multiwindow_feature_set,
    prepare_multiwindow_batch_from_manifest,
)

from app.model.preprocessing import (
    EXPECTED_INPUT_SHAPE,
)


AI_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

TRAIN_CLIPS_FILE = (
    AI_ROOT
    / "data"
    / "processed"
    / "train_clips.csv"
)


def test_sample(
    sample,
    model,
    device,
):
    print()
    print("=" * 72)
    print(
        f"SAMPLE: {sample['clip_type']}"
    )
    print("=" * 72)

    print(
        "clip_id:",
        sample[
            "clip_id"
        ],
    )

    print(
        "video:",
        sample[
            "canonical_video_id"
        ],
    )

    print(
        "label:",
        sample[
            "label"
        ],
    )

    print(
        "manifest:",
        sample[
            "start_sec"
        ],
        "→",
        sample[
            "end_sec"
        ],
    )

    batch, metadata = (
        prepare_multiwindow_batch_from_manifest(
            row=sample,
            ai_root=AI_ROOT,
            device=device,
        )
    )

    print()
    print("TEMPORAL WINDOWS")
    print("-" * 72)

    for window in (
        metadata[
            "windows"
        ]
    ):
        print(
            f"window "
            f"{window['window_index']}: "
            f"{window['start_sec']} "
            f"→ "
            f"{window['end_sec']} "
            f"(center "
            f"{window['center_sec']})"
        )

    expected_shape = (
        DEFAULT_NUM_WINDOWS,
        *EXPECTED_INPUT_SHAPE,
    )

    actual_shape = tuple(
        batch.shape
    )

    print()
    print(
        "Expected batch:",
        expected_shape,
    )

    print(
        "Actual batch:  ",
        actual_shape,
    )

    print(
        "dtype:",
        batch.dtype,
    )

    print(
        "device:",
        batch.device,
    )

    if (
        actual_shape
        != expected_shape
    ):
        raise RuntimeError(
            "Invalid multi-window "
            "batch shape."
        )

    feature_set = (
        extract_multiwindow_feature_set(
            model,
            batch,
        )
    )

    windows = (
        feature_set[
            "windows"
        ]
    )

    mean_features = (
        feature_set[
            "mean"
        ]
    )

    max_features = (
        feature_set[
            "max"
        ]
    )

    print()
    print("FEATURES")
    print("-" * 72)

    print(
        "Per-window:",
        tuple(
            windows.shape
        ),
    )

    print(
        "Mean pooled:",
        tuple(
            mean_features.shape
        ),
    )

    print(
        "Max pooled:",
        tuple(
            max_features.shape
        ),
    )

    print(
        "Mean feature std:",
        float(
            mean_features.std().item()
        ),
    )

    print(
        "Max feature std:",
        float(
            max_features.std().item()
        ),
    )

    if tuple(
        windows.shape
    ) != (
        DEFAULT_NUM_WINDOWS,
        2048,
    ):
        raise RuntimeError(
            "Invalid per-window "
            "feature shape."
        )

    if tuple(
        mean_features.shape
    ) != (
        1,
        2048,
    ):
        raise RuntimeError(
            "Invalid mean feature shape."
        )

    if tuple(
        max_features.shape
    ) != (
        1,
        2048,
    ):
        raise RuntimeError(
            "Invalid max feature shape."
        )

    if not torch.isfinite(
        windows
    ).all():
        raise RuntimeError(
            "Window features contain "
            "NaN or Inf."
        )


def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 7B "
        "Multi-window X3D Test"
    )
    print("=" * 72)

    clips = pd.read_csv(
        TRAIN_CLIPS_FILE
    )

    positive = (
        clips[
            clips[
                "label"
            ].eq(1)
        ]
        .iloc[0]
    )

    negative = (
        clips[
            clips[
                "label"
            ].eq(0)
        ]
        .iloc[0]
    )

    model, device = (
        load_model(
            pretrained=True
        )
    )

    print()
    print(
        "Device:",
        device,
    )

    print(
        "Number of windows:",
        DEFAULT_NUM_WINDOWS,
    )

    test_sample(
        positive,
        model,
        device,
    )

    test_sample(
        negative,
        model,
        device,
    )

    print()
    print("=" * 72)

    print(
        "✅ Positive multi-window "
        "preprocessing passed."
    )

    print(
        "✅ Negative multi-window "
        "preprocessing passed."
    )

    print(
        "✅ Per-window X3D features passed."
    )

    print(
        "✅ Mean pooling passed."
    )

    print(
        "✅ Max pooling passed."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()