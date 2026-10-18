from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from app.model.loader import (
    calculate_weights_sha256,
    get_weights_path,
    load_model,
)

from app.model.multiwindow import (
    DEFAULT_NUM_WINDOWS,
    extract_multiwindow_feature_set,
    prepare_multiwindow_batch_from_manifest,
)

from app.model.preprocessing import (
    MODEL_CLIP_DURATION_SEC,
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
    / "train_x3d_multiwindow_features.npz"
)

VALIDATION_FEATURES_FILE = (
    FEATURES_DIR
    / "validation_x3d_multiwindow_features.npz"
)

TRAIN_METADATA_FILE = (
    FEATURES_DIR
    / "train_multiwindow_metadata.csv"
)

VALIDATION_METADATA_FILE = (
    FEATURES_DIR
    / "validation_multiwindow_metadata.csv"
)

SUMMARY_FILE = (
    FEATURES_DIR
    / "multiwindow_features_summary.json"
)


# =========================================================
# MANIFEST
# =========================================================

def load_manifest(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing manifest: "
            f"{path}"
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
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Missing manifest "
            f"columns: {missing}"
        )

    return dataframe


# =========================================================
# EXTRACT ONE SPLIT
# =========================================================

def extract_split(
    manifest: pd.DataFrame,
    model,
    device,
    split_name: str,
    output_npz: Path,
    output_metadata: Path,
) -> dict:
    mean_features_list = []

    max_features_list = []

    window_features_list = []

    metadata_rows = []

    print()
    print(
        f"Extracting {split_name} "
        f"multi-window features..."
    )

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(
            manifest
        ),
        desc=(
            split_name
        ),
    ):
        (
            multiwindow_batch,
            metadata,
        ) = (
            prepare_multiwindow_batch_from_manifest(
                row=row,
                ai_root=AI_ROOT,
                device=device,
                num_windows=(
                    DEFAULT_NUM_WINDOWS
                ),
            )
        )

        feature_set = (
            extract_multiwindow_feature_set(
                model,
                multiwindow_batch,
            )
        )

        windows = (
            feature_set[
                "windows"
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        mean_features = (
            feature_set[
                "mean"
            ][0]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        max_features = (
            feature_set[
                "max"
            ][0]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        if (
            windows.shape
            != (
                DEFAULT_NUM_WINDOWS,
                2048,
            )
        ):
            raise RuntimeError(
                "Unexpected window "
                "feature shape for "
                f"{row['clip_id']}: "
                f"{windows.shape}"
            )

        if (
            mean_features.shape
            != (
                2048,
            )
        ):
            raise RuntimeError(
                "Unexpected mean feature "
                "shape."
            )

        if (
            max_features.shape
            != (
                2048,
            )
        ):
            raise RuntimeError(
                "Unexpected max feature "
                "shape."
            )

        if not np.isfinite(
            windows
        ).all():
            raise RuntimeError(
                "Non-finite window "
                "features detected."
            )

        if not np.isfinite(
            mean_features
        ).all():
            raise RuntimeError(
                "Non-finite mean "
                "features detected."
            )

        if not np.isfinite(
            max_features
        ).all():
            raise RuntimeError(
                "Non-finite max "
                "features detected."
            )

        window_features_list.append(
            windows
        )

        mean_features_list.append(
            mean_features
        )

        max_features_list.append(
            max_features
        )

        metadata_row = {
            "clip_id": str(
                row[
                    "clip_id"
                ]
            ),

            "content_id": str(
                row[
                    "content_id"
                ]
            ),

            "canonical_video_id": str(
                row[
                    "canonical_video_id"
                ]
            ),

            "label": int(
                row[
                    "label"
                ]
            ),

            "clip_type": str(
                row[
                    "clip_type"
                ]
            ),

            "manifest_start_sec": float(
                row[
                    "start_sec"
                ]
            ),

            "manifest_end_sec": float(
                row[
                    "end_sec"
                ]
            ),

            "num_windows": (
                metadata[
                    "num_windows"
                ]
            ),
        }

        for window in (
            metadata[
                "windows"
            ]
        ):
            index = int(
                window[
                    "window_index"
                ]
            )

            metadata_row[
                f"window_{index}_start_sec"
            ] = float(
                window[
                    "start_sec"
                ]
            )

            metadata_row[
                f"window_{index}_center_sec"
            ] = float(
                window[
                    "center_sec"
                ]
            )

            metadata_row[
                f"window_{index}_end_sec"
            ] = float(
                window[
                    "end_sec"
                ]
            )

        metadata_rows.append(
            metadata_row
        )

        del multiwindow_batch
        del feature_set

    if not mean_features_list:
        raise RuntimeError(
            f"No {split_name} "
            "features extracted."
        )

    X_mean = np.stack(
        mean_features_list,
        axis=0,
    )

    X_max = np.stack(
        max_features_list,
        axis=0,
    )

    X_windows = np.stack(
        window_features_list,
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

    # X is intentionally equal to mean pooling.
    # This keeps compatibility with the existing
    # classifier/CV code.
    np.savez_compressed(
        output_npz,

        X=X_mean,

        X_mean=X_mean,

        X_max=X_max,

        X_windows=X_windows,

        y=y,

        clip_ids=(
            clip_ids
        ),

        content_ids=(
            content_ids
        ),

        video_ids=(
            video_ids
        ),

        clip_types=(
            clip_types
        ),
    )

    metadata_df.to_csv(
        output_metadata,
        index=False,
    )

    statistics = {
        "clips": int(
            len(
                metadata_df
            )
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

        "num_windows": (
            DEFAULT_NUM_WINDOWS
        ),

        "window_feature_shape": [
            int(value)
            for value in (
                X_windows.shape
            )
        ],

        "mean_feature_shape": [
            int(value)
            for value in (
                X_mean.shape
            )
        ],

        "max_feature_shape": [
            int(value)
            for value in (
                X_max.shape
            )
        ],

        "mean_pool_feature_mean": float(
            X_mean.mean()
        ),

        "mean_pool_feature_std": float(
            X_mean.std()
        ),

        "max_pool_feature_mean": float(
            X_max.mean()
        ),

        "max_pool_feature_std": float(
            X_max.std()
        ),

        "all_window_features_finite": bool(
            np.isfinite(
                X_windows
            ).all()
        ),

        "all_mean_features_finite": bool(
            np.isfinite(
                X_mean
            ).all()
        ),

        "all_max_features_finite": bool(
            np.isfinite(
                X_max
            ).all()
        ),
    }

    print()
    print(
        f"{split_name} mean shape:",
        X_mean.shape,
    )

    print(
        f"{split_name} max shape:",
        X_max.shape,
    )

    print(
        f"{split_name} windows shape:",
        X_windows.shape,
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
        "SecureVision — Phase 7B "
        "Multi-window Feature Extraction"
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
    print("CONFIGURATION")
    print("-" * 72)

    print(
        "Temporal windows per clip:",
        DEFAULT_NUM_WINDOWS,
    )

    print(
        "X3D window duration:",
        MODEL_CLIP_DURATION_SEC,
    )

    print(
        "Saved pooling strategies:",
        "mean + max",
    )

    print(
        "Default X array:",
        "mean pooled features",
    )

    print()
    print("MANIFESTS")
    print("-" * 72)

    print(
        "Train clips:",
        len(
            train_manifest
        ),
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
            "Pretrained X3D weights "
            "are missing."
        )

    print()
    print("MODEL")
    print("-" * 72)

    print(
        "Weights:",
        weights_path,
    )

    weights_sha256 = (
        calculate_weights_sha256()
    )

    print(
        "SHA256:",
        weights_sha256,
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

    # -----------------------------------------------------
    # Extract
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
    # Leakage
    # -----------------------------------------------------

    train_content_ids = set(
        train_manifest[
            "content_id"
        ].astype(str)
    )

    validation_content_ids = set(
        validation_manifest[
            "content_id"
        ].astype(str)
    )

    overlap = (
        train_content_ids
        & validation_content_ids
    )

    if overlap:
        raise RuntimeError(
            "Train/validation leakage "
            "detected in multi-window "
            "features."
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary = {
        "model": {
            "name": (
                "x3d_xs"
            ),

            "pretrained": True,

            "pretraining_dataset": (
                "Kinetics-400"
            ),

            "feature_dim": 2048,

            "weights_file": str(
                weights_path
            ),

            "weights_sha256": (
                weights_sha256
            ),

            "device": str(
                device
            ),
        },

        "multiwindow": {
            "num_windows": (
                DEFAULT_NUM_WINDOWS
            ),

            "model_window_duration_sec": (
                MODEL_CLIP_DURATION_SEC
            ),

            "window_position_strategy": (
                "evenly_spaced_inside_manifest"
            ),

            "uses_event_timestamp": (
                False
            ),

            "saved_pooling_strategies": [
                "mean",
                "max",
            ],

            "default_X_pooling": (
                "mean"
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
                len(
                    overlap
                )
            ),

            "all_train_mean_features_finite": (
                train_stats[
                    "all_mean_features_finite"
                ]
            ),

            "all_validation_mean_features_finite": (
                validation_stats[
                    "all_mean_features_finite"
                ]
            ),

            "all_train_window_features_finite": (
                train_stats[
                    "all_window_features_finite"
                ]
            ),

            "all_validation_window_features_finite": (
                validation_stats[
                    "all_window_features_finite"
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
    print(
        "MULTI-WINDOW EXTRACTION COMPLETE"
    )
    print("=" * 72)

    print(
        "Train:",
        TRAIN_FEATURES_FILE,
    )

    print(
        "Validation:",
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
        len(
            overlap
        ),
    )

    print()

    print(
        "✅ Multi-window X3D "
        "features extracted."
    )


if __name__ == "__main__":
    main()