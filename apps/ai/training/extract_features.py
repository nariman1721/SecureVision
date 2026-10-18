from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from app.model.features import (
    X3D_FEATURE_DIM,
    extract_x3d_features,
)

from app.model.inference import (
    prepare_model_input_from_manifest,
)

from app.model.loader import (
    calculate_weights_sha256,
    get_weights_path,
    load_model,
)


# =========================================================
# PATHS
# =========================================================

AI_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

PROCESSED_DIR = (
    AI_ROOT
    / "data"
    / "processed"
)

FEATURES_DIR = (
    PROCESSED_DIR
    / "features"
)

TRAIN_CLIPS_FILE = (
    PROCESSED_DIR
    / "train_clips.csv"
)

VALIDATION_CLIPS_FILE = (
    PROCESSED_DIR
    / "validation_clips.csv"
)

TRAIN_FEATURES_FILE = (
    FEATURES_DIR
    / "train_x3d_features.npz"
)

VALIDATION_FEATURES_FILE = (
    FEATURES_DIR
    / "validation_x3d_features.npz"
)

TRAIN_METADATA_FILE = (
    FEATURES_DIR
    / "train_features_metadata.csv"
)

VALIDATION_METADATA_FILE = (
    FEATURES_DIR
    / "validation_features_metadata.csv"
)

SUMMARY_FILE = (
    FEATURES_DIR
    / "features_summary.json"
)


# =========================================================
# HELPERS
# =========================================================

def load_manifest(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing manifest: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    required = [
        "clip_id",
        "content_id",
        "canonical_video_id",
        "video_path",
        "label",
        "clip_type",
        "start_sec",
        "end_sec",
        "center_sec",
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return dataframe


def extract_split(
    manifest: pd.DataFrame,
    model: torch.nn.Module,
    device: torch.device,
    split_name: str,
    output_npz: Path,
    output_metadata: Path,
) -> dict:
    features_list = []
    metadata_rows = []

    print()
    print(
        f"Extracting {split_name} features..."
    )

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc=split_name,
    ):
        batch, inference_metadata = (
            prepare_model_input_from_manifest(
                row=row,
                ai_root=AI_ROOT,
                device=device,
            )
        )

        features = (
            extract_x3d_features(
                model,
                batch,
            )
        )

        vector = (
            features[
                0
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        if (
            vector.shape
            != (
                X3D_FEATURE_DIM,
            )
        ):
            raise RuntimeError(
                "Unexpected feature shape "
                f"for {row['clip_id']}: "
                f"{vector.shape}"
            )

        if not np.isfinite(
            vector
        ).all():
            raise RuntimeError(
                "Non-finite features for "
                f"{row['clip_id']}"
            )

        features_list.append(
            vector
        )

        metadata_rows.append(
            {
                "clip_id": (
                    str(
                        row[
                            "clip_id"
                        ]
                    )
                ),

                "content_id": (
                    str(
                        row[
                            "content_id"
                        ]
                    )
                ),

                "canonical_video_id": (
                    str(
                        row[
                            "canonical_video_id"
                        ]
                    )
                ),

                "label": int(
                    row[
                        "label"
                    ]
                ),

                "clip_type": (
                    str(
                        row[
                            "clip_type"
                        ]
                    )
                ),

                "start_sec": float(
                    row[
                        "start_sec"
                    ]
                ),

                "end_sec": float(
                    row[
                        "end_sec"
                    ]
                ),

                "model_start_sec": (
                    inference_metadata[
                        "model_start_sec"
                    ]
                ),

                "model_end_sec": (
                    inference_metadata[
                        "model_end_sec"
                    ]
                ),
            }
        )

        del batch
        del features

    if not features_list:
        raise RuntimeError(
            f"No {split_name} "
            "features extracted."
        )

    X = np.stack(
        features_list,
        axis=0,
    )

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    y = (
        metadata_df[
            "label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    clip_ids = (
        metadata_df[
            "clip_id"
        ]
        .astype(str)
        .to_numpy(
            dtype=np.str_
        )
    )

    content_ids = (
        metadata_df[
            "content_id"
        ]
        .astype(str)
        .to_numpy(
            dtype=np.str_
        )
    )

    video_ids = (
        metadata_df[
            "canonical_video_id"
        ]
        .astype(str)
        .to_numpy(
            dtype=np.str_
        )
    )

    clip_types = (
        metadata_df[
            "clip_type"
        ]
        .astype(str)
        .to_numpy(
            dtype=np.str_
        )
    )

    np.savez_compressed(
        output_npz,
        X=X,
        y=y,
        clip_ids=clip_ids,
        content_ids=content_ids,
        video_ids=video_ids,
        clip_types=clip_types,
    )

    metadata_df.to_csv(
        output_metadata,
        index=False,
    )

    statistics = {
        "clips": int(
            len(metadata_df)
        ),

        "feature_dim": int(
            X.shape[1]
        ),

        "positive": int(
            (y == 1).sum()
        ),

        "negative": int(
            (y == 0).sum()
        ),

        "unique_contents": int(
            metadata_df[
                "content_id"
            ].nunique()
        ),

        "feature_mean": float(
            X.mean()
        ),

        "feature_std": float(
            X.std()
        ),

        "all_finite": bool(
            np.isfinite(
                X
            ).all()
        ),
    }

    print()
    print(
        f"{split_name} shape:",
        X.shape,
    )

    print(
        f"{split_name} positive:",
        statistics[
            "positive"
        ],
    )

    print(
        f"{split_name} negative:",
        statistics[
            "negative"
        ],
    )

    return statistics


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 5B "
        "X3D Feature Extraction"
    )
    print("=" * 72)

    FEATURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_manifest = (
        load_manifest(
            TRAIN_CLIPS_FILE
        )
    )

    validation_manifest = (
        load_manifest(
            VALIDATION_CLIPS_FILE
        )
    )

    print()
    print("MANIFESTS")
    print("-" * 72)

    print(
        "Train clips:",
        len(train_manifest),
    )

    print(
        "Validation clips:",
        len(
            validation_manifest
        ),
    )

    weights_path = (
        get_weights_path()
    )

    if not weights_path.exists():
        raise FileNotFoundError(
            "\nPretrained weights missing.\n"
            "Run:\n"
            "python -m "
            "training.download_pretrained"
        )

    print()
    print("MODEL")
    print("-" * 72)

    print(
        "Weights:",
        weights_path,
    )

    print(
        "SHA256:",
        calculate_weights_sha256(),
    )

    model, device = (
        load_model(
            pretrained=True
        )
    )

    print(
        "Device:",
        device,
    )

    print(
        "Feature dimension:",
        X3D_FEATURE_DIM,
    )

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    train_stats = (
        extract_split(
            manifest=(
                train_manifest
            ),
            model=model,
            device=device,
            split_name="train",
            output_npz=(
                TRAIN_FEATURES_FILE
            ),
            output_metadata=(
                TRAIN_METADATA_FILE
            ),
        )
    )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    validation_stats = (
        extract_split(
            manifest=(
                validation_manifest
            ),
            model=model,
            device=device,
            split_name="validation",
            output_npz=(
                VALIDATION_FEATURES_FILE
            ),
            output_metadata=(
                VALIDATION_METADATA_FILE
            ),
        )
    )

    # -----------------------------------------------------
    # LEAKAGE CHECK
    # -----------------------------------------------------

    train_content_ids = set(
        train_manifest[
            "content_id"
        ]
    )

    validation_content_ids = set(
        validation_manifest[
            "content_id"
        ]
    )

    overlap = (
        train_content_ids
        & validation_content_ids
    )

    if overlap:
        raise RuntimeError(
            "Train/validation feature "
            "content leakage detected."
        )

    summary = {
        "model": {
            "name": (
                "x3d_xs"
            ),

            "pretrained": True,

            "pretraining_dataset": (
                "Kinetics-400"
            ),

            "feature_dim": (
                X3D_FEATURE_DIM
            ),

            "weights_file": (
                str(
                    weights_path
                )
            ),

            "weights_sha256": (
                calculate_weights_sha256()
            ),

            "device": (
                str(device)
            ),
        },

        "train": (
            train_stats
        ),

        "validation": (
            validation_stats
        ),

        "validation_checks": {
            "content_overlap": (
                len(overlap)
            ),

            "all_train_features_finite": (
                train_stats[
                    "all_finite"
                ]
            ),

            "all_validation_features_finite": (
                validation_stats[
                    "all_finite"
                ]
            ),
        },
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 72)

    print(
        "Train features:",
        TRAIN_FEATURES_FILE,
    )

    print(
        "Validation features:",
        VALIDATION_FEATURES_FILE,
    )

    print(
        "Summary:",
        SUMMARY_FILE,
    )

    print()
    print(
        "Train/validation "
        "content overlap:",
        len(overlap),
    )

    print()

    print(
        "✅ Pretrained X3D features "
        "successfully extracted."
    )


if __name__ == "__main__":
    main()