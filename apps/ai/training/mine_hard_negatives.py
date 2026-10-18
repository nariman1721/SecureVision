from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from app.model.loader import load_model
from app.model.multiwindow import (
    extract_multiwindow_feature_set,
    prepare_multiwindow_batch_from_manifest,
)

from training.diagnose_temporal_signal import (
    load_train_manifest,
)


AI_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = AI_ROOT / "reports"
PROCESSED_DIR = AI_ROOT / "data" / "processed"

NEGATIVE_WINDOWS_FILE = (
    REPORTS_DIR
    / "negative_temporal_windows.csv"
)

MANIFEST_FILE = (
    PROCESSED_DIR
    / "hard_negative_clips.csv"
)

FEATURES_FILE = (
    PROCESSED_DIR
    / "hard_negative_multiwindow_features.npz"
)

SUMMARY_FILE = (
    REPORTS_DIR
    / "hard_negative_mining_summary.json"
)


MIN_PROBABILITY = 0.50
MAX_WINDOWS_PER_VIDEO = 2
MIN_CENTER_DISTANCE_SEC = 4.0


def resolve_video_path(
    value: str,
) -> Path:
    path = Path(
        str(value)
    )

    if not path.is_absolute():
        path = (
            AI_ROOT
            / path
        )

    path = path.resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Video not found: {path}"
        )

    return path


def build_video_mapping(
    manifest: pd.DataFrame,
) -> dict[str, dict]:
    normal = (
        manifest[
            manifest[
                "clip_type"
            ]
            .astype(str)
            .eq(
                "negative_normal"
            )
        ]
        .copy()
    )

    mapping = {}

    for _, row in (
        normal.iterrows()
    ):
        video_id = str(
            row[
                "canonical_video_id"
            ]
        )

        if video_id in mapping:
            continue

        mapping[
            video_id
        ] = {
            "content_id": str(
                row[
                    "content_id"
                ]
            ),

            "video_path": str(
                resolve_video_path(
                    row[
                        "video_path"
                    ]
                )
            ),
        }

    return mapping


def select_hard_windows(
    windows: pd.DataFrame,
) -> pd.DataFrame:
    selected_rows = []

    for video_id, group in (
        windows.groupby(
            "video_id"
        )
    ):
        group = (
            group[
                group[
                    "raw_probability"
                ]
                >= MIN_PROBABILITY
            ]
            .sort_values(
                [
                    "raw_probability",
                    "candidate_score",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .copy()
        )

        selected_centers = []

        for _, row in (
            group.iterrows()
        ):
            center = float(
                row[
                    "center_sec"
                ]
            )

            sufficiently_far = all(
                abs(
                    center
                    - existing
                )
                >= MIN_CENTER_DISTANCE_SEC
                for existing
                in selected_centers
            )

            if not sufficiently_far:
                continue

            selected_rows.append(
                row.to_dict()
            )

            selected_centers.append(
                center
            )

            if (
                len(
                    selected_centers
                )
                >= MAX_WINDOWS_PER_VIDEO
            ):
                break

    if not selected_rows:
        raise RuntimeError(
            "No hard negative windows selected."
        )

    return (
        pd.DataFrame(
            selected_rows
        )
        .reset_index(
            drop=True
        )
    )


def main():
    print("=" * 72)
    print(
        "SecureVision — Phase 8G-A "
        "Hard Negative Mining"
    )
    print("=" * 72)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not NEGATIVE_WINDOWS_FILE.exists():
        raise FileNotFoundError(
            f"Missing:\n"
            f"{NEGATIVE_WINDOWS_FILE}"
        )

    windows = pd.read_csv(
        NEGATIVE_WINDOWS_FILE
    )

    train_manifest = (
        load_train_manifest()
    )

    video_mapping = (
        build_video_mapping(
            train_manifest
        )
    )

    hard = (
        select_hard_windows(
            windows
        )
    )

    print()
    print("SELECTION")
    print("-" * 72)

    print(
        "Normal scan windows:",
        len(
            windows
        ),
    )

    print(
        "Minimum probability:",
        MIN_PROBABILITY,
    )

    print(
        "Maximum per video:",
        MAX_WINDOWS_PER_VIDEO,
    )

    print(
        "Minimum center distance:",
        MIN_CENTER_DISTANCE_SEC,
    )

    print(
        "Selected hard negatives:",
        len(
            hard
        ),
    )

    print(
        "Videos represented:",
        hard[
            "video_id"
        ].nunique(),
    )

    manifest_rows = []

    for index, row in (
        hard.iterrows()
    ):
        video_id = str(
            row[
                "video_id"
            ]
        )

        if (
            video_id
            not in video_mapping
        ):
            raise RuntimeError(
                f"No train mapping for "
                f"{video_id}"
            )

        mapping = (
            video_mapping[
                video_id
            ]
        )

        manifest_rows.append(
            {
                "clip_id": (
                    f"hardneg_"
                    f"{video_id}_"
                    f"{index + 1:03d}"
                ),

                "content_id": (
                    mapping[
                        "content_id"
                    ]
                ),

                "canonical_video_id": (
                    video_id
                ),

                "video_path": (
                    mapping[
                        "video_path"
                    ]
                ),

                "label": 0,

                "clip_type": (
                    "hard_negative_normal"
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

                "center_sec": float(
                    row[
                        "center_sec"
                    ]
                ),

                "old_probability": float(
                    row[
                        "raw_probability"
                    ]
                ),

                "old_candidate_score": float(
                    row[
                        "candidate_score"
                    ]
                ),
            }
        )

    hard_manifest = pd.DataFrame(
        manifest_rows
    )

    hard_manifest.to_csv(
        MANIFEST_FILE,
        index=False,
    )

    print()
    print("SELECTED WINDOWS")
    print("-" * 72)

    print(
        hard_manifest[
            [
                "canonical_video_id",
                "start_sec",
                "end_sec",
                "center_sec",
                "old_probability",
            ]
        ]
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print()
    print("LOADING X3D")
    print("-" * 72)

    model, device = (
        load_model(
            pretrained=True
        )
    )

    print(
        "Device:",
        device,
    )

    features = []

    for _, row in tqdm(
        hard_manifest.iterrows(),
        total=len(
            hard_manifest
        ),
        desc="hard-negative features",
    ):
        batch, _ = (
            prepare_multiwindow_batch_from_manifest(
                row=row.to_dict(),
                ai_root=AI_ROOT,
                device=device,
                num_windows=3,
            )
        )

        feature_set = (
            extract_multiwindow_feature_set(
                model,
                batch,
            )
        )

        feature = (
            feature_set[
                "mean"
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
            .reshape(
                -1
            )
        )

        if feature.shape != (
            2048,
        ):
            raise RuntimeError(
                "Unexpected feature shape: "
                f"{feature.shape}"
            )

        features.append(
            feature
        )

    X = np.stack(
        features,
        axis=0,
    )

    y = np.zeros(
        len(
            hard_manifest
        ),
        dtype=np.int64,
    )

    np.savez_compressed(
        FEATURES_FILE,
        X=X,
        y=y,

        clip_ids=(
            hard_manifest[
                "clip_id"
            ]
            .astype(str)
            .to_numpy()
        ),

        content_ids=(
            hard_manifest[
                "content_id"
            ]
            .astype(str)
            .to_numpy()
        ),

        video_ids=(
            hard_manifest[
                "canonical_video_id"
            ]
            .astype(str)
            .to_numpy()
        ),

        old_probabilities=(
            hard_manifest[
                "old_probability"
            ]
            .to_numpy(
                dtype=np.float32
            )
        ),
    )

    report = {
        "configuration": {
            "minimum_probability": (
                MIN_PROBABILITY
            ),

            "maximum_windows_per_video": (
                MAX_WINDOWS_PER_VIDEO
            ),

            "minimum_center_distance_sec": (
                MIN_CENTER_DISTANCE_SEC
            ),

            "development_validation_used": (
                False
            ),
        },

        "selection": {
            "source_windows": int(
                len(
                    windows
                )
            ),

            "hard_negative_windows": int(
                len(
                    hard_manifest
                )
            ),

            "videos_represented": int(
                hard_manifest[
                    "canonical_video_id"
                ].nunique()
            ),

            "mean_old_probability": float(
                hard_manifest[
                    "old_probability"
                ].mean()
            ),

            "max_old_probability": float(
                hard_manifest[
                    "old_probability"
                ].max()
            ),
        },

        "features": {
            "shape": [
                int(value)
                for value
                in X.shape
            ],

            "finite": bool(
                np.isfinite(
                    X
                ).all()
            ),
        },
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)

    print(
        "Feature shape:",
        X.shape,
    )

    print(
        "Finite:",
        np.isfinite(
            X
        ).all(),
    )

    print()
    print("FILES CREATED")
    print("-" * 72)

    print(
        MANIFEST_FILE
    )

    print(
        FEATURES_FILE
    )

    print(
        SUMMARY_FILE
    )

    print()
    print(
        "✅ Hard negative mining complete."
    )


if __name__ == "__main__":
    main()