from __future__ import annotations
from pathlib import Path
import pandas as pd
import torch
from app.model.inference import (
    forward_model,
    prepare_model_input_from_manifest,
)

from app.model.loader import (
    load_model,
)

from app.model.preprocessing import (
    EXPECTED_INPUT_SHAPE,
    MODEL_CLIP_DURATION_SEC,
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


def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 5 "
        "X3D Preprocessing Test"
    )
    print("=" * 72)

    if not TRAIN_CLIPS_FILE.exists():
        raise FileNotFoundError(
            f"Missing file: "
            f"{TRAIN_CLIPS_FILE}"
        )

    clips = pd.read_csv(
        TRAIN_CLIPS_FILE
    )

    positive_clips = clips[
        clips[
            "label"
        ].eq(1)
    ]

    if positive_clips.empty:
        raise RuntimeError(
            "No positive clips found."
        )

    sample = (
        positive_clips
        .iloc[0]
    )

    print()
    print("SAMPLE MANIFEST")
    print("-" * 72)

    print(
        "clip_id:       ",
        sample[
            "clip_id"
        ],
    )

    print(
        "video:         ",
        sample[
            "canonical_video_id"
        ],
    )

    print(
        "video_path:    ",
        sample[
            "video_path"
        ],
    )

    print(
        "manifest:      ",
        f"{sample['start_sec']} "
        f"→ {sample['end_sec']}",
    )

    print(
        "event:         ",
        sample[
            "event_timecode_sec"
        ],
    )

    print()
    print(
        "X3D model window duration:",
        MODEL_CLIP_DURATION_SEC,
    )

    # -----------------------------------------------------
    # Load architecture/device
    # -----------------------------------------------------

    model, device = (
        load_model()
    )

    print()
    print("DEVICE")
    print("-" * 72)

    print(
        "Device:",
        device,
    )

    # -----------------------------------------------------
    # Preprocess
    # -----------------------------------------------------

    batch, metadata = (
        prepare_model_input_from_manifest(
            row=sample,
            ai_root=AI_ROOT,
            device=device,
        )
    )

    print()
    print("PREPROCESSING")
    print("-" * 72)

    for key, value in (
        metadata.items()
    ):
        print(
            f"{key:28} {value}"
        )

    expected_batch_shape = (
        1,
        *EXPECTED_INPUT_SHAPE,
    )

    actual_batch_shape = (
        tuple(
            batch.shape
        )
    )

    print()
    print(
        "Expected batch shape:",
        expected_batch_shape,
    )

    print(
        "Actual batch shape:  ",
        actual_batch_shape,
    )

    print(
        "dtype:               ",
        batch.dtype,
    )

    print(
        "device:              ",
        batch.device,
    )

    if (
        actual_batch_shape
        != expected_batch_shape
    ):
        raise RuntimeError(
            "Unexpected model input shape."
        )

    if (
        batch.dtype
        != torch.float32
    ):
        raise RuntimeError(
            "Expected float32 tensor."
        )

    if not torch.isfinite(
        batch
    ).all():
        raise RuntimeError(
            "Tensor contains NaN or Inf."
        )

    # -----------------------------------------------------
    # X3D compatibility test
    # -----------------------------------------------------

    print()
    print("X3D FORWARD TEST")
    print("-" * 72)

    output = forward_model(
        model,
        batch,
    )

    print(
        "Output shape:",
        tuple(
            output.shape
        ),
    )

    print(
        "Output dtype:",
        output.dtype,
    )

    print(
        "Output device:",
        output.device,
    )

    if output.ndim != 2:
        raise RuntimeError(
            "Unexpected X3D output rank."
        )

    if output.shape[0] != 1:
        raise RuntimeError(
            "Unexpected X3D batch size."
        )

    print()
    print("=" * 72)

    print(
        "✅ Video decoding passed."
    )

    print(
        "✅ Temporal preprocessing passed."
    )

    print(
        "✅ X3D tensor preprocessing passed."
    )

    print(
        "✅ Device transfer passed."
    )

    print(
        "✅ X3D forward compatibility passed."
    )

    print("=" * 72)

    print()
    print(
        "NOTE: model weights are currently "
        "pretrained=False."
    )

    print(
        "The forward pass verifies compatibility "
        "only, not violation detection."
    )


if __name__ == "__main__":
    main()