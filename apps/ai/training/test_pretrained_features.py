from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from app.model.features import (
    X3D_FEATURE_DIM,
    extract_x3d_features,
)

from app.model.inference import (
    forward_model,
    prepare_model_input_from_manifest,
)

from app.model.loader import (
    calculate_weights_sha256,
    get_weights_path,
    load_model,
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
        "SecureVision — Phase 5B "
        "Pretrained Feature Test"
    )
    print("=" * 72)

    if not TRAIN_CLIPS_FILE.exists():
        raise FileNotFoundError(
            TRAIN_CLIPS_FILE
        )

    print()
    print("PRETRAINED WEIGHTS")
    print("-" * 72)

    weights_path = (
        get_weights_path()
    )

    print(
        "Weights:",
        weights_path,
    )

    print(
        "Exists:",
        weights_path.exists(),
    )

    if not weights_path.exists():
        raise FileNotFoundError(
            "\nRun first:\n"
            "python -m "
            "training.download_pretrained"
        )

    print(
        "SHA256:",
        calculate_weights_sha256(),
    )

    clips = pd.read_csv(
        TRAIN_CLIPS_FILE
    )

    sample = (
        clips[
            clips[
                "label"
            ].eq(1)
        ]
        .iloc[0]
    )

    print()
    print("SAMPLE")
    print("-" * 72)

    print(
        "clip_id:",
        sample["clip_id"],
    )

    print(
        "video:",
        sample[
            "canonical_video_id"
        ],
    )

    print(
        "event:",
        sample[
            "event_timecode_sec"
        ],
    )

    model, device = (
        load_model(
            pretrained=True
        )
    )

    print()
    print("MODEL")
    print("-" * 72)

    print(
        "Device:",
        device,
    )

    print(
        "Pretrained:",
        True,
    )

    batch, metadata = (
        prepare_model_input_from_manifest(
            row=sample,
            ai_root=AI_ROOT,
            device=device,
        )
    )

    print(
        "Batch shape:",
        tuple(
            batch.shape
        ),
    )

    # ---------------------------------------------
    # Original Kinetics output
    # ---------------------------------------------

    kinetics_output = (
        forward_model(
            model,
            batch,
        )
    )

    print()
    print("KINETICS HEAD")
    print("-" * 72)

    print(
        "Output shape:",
        tuple(
            kinetics_output.shape
        ),
    )

    # ---------------------------------------------
    # SecureVision features
    # ---------------------------------------------

    features = (
        extract_x3d_features(
            model,
            batch,
        )
    )

    print()
    print("FEATURE VECTOR")
    print("-" * 72)

    print(
        "Shape:",
        tuple(
            features.shape
        ),
    )

    print(
        "Expected dimension:",
        X3D_FEATURE_DIM,
    )

    print(
        "dtype:",
        features.dtype,
    )

    print(
        "device:",
        features.device,
    )

    print(
        "mean:",
        float(
            features.mean().item()
        ),
    )

    print(
        "std:",
        float(
            features.std().item()
        ),
    )

    print(
        "L2 norm:",
        float(
            torch.linalg.vector_norm(
                features
            ).item()
        ),
    )

    if tuple(
        features.shape
    ) != (
        1,
        X3D_FEATURE_DIM,
    ):
        raise RuntimeError(
            "Invalid feature shape."
        )

    if not torch.isfinite(
        features
    ).all():
        raise RuntimeError(
            "Features contain "
            "NaN or Inf."
        )

    if float(
        features.std().item()
    ) <= 0:
        raise RuntimeError(
            "Feature vector has "
            "zero variance."
        )

    print()
    print("=" * 72)

    print(
        "✅ Local pretrained "
        "checkpoint loaded."
    )

    print(
        "✅ X3D forward passed."
    )

    print(
        "✅ 2048-dimensional "
        "features extracted."
    )

    print(
        "✅ Features are finite."
    )

    print(
        "✅ Feature variance > 0."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()