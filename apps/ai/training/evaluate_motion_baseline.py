from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.model.motion_features import (
    extract_motion_features,
    motion_feature_names,
)


RANDOM_STATE = 42
N_SPLITS = 4

AI_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = AI_ROOT / "data" / "processed"
FEATURES_DIR = PROCESSED_DIR / "features"
REPORTS_DIR = AI_ROOT / "reports"
MODELS_DIR = AI_ROOT / "models"

TRAIN_MANIFEST = PROCESSED_DIR / "train_clips.csv"
VALIDATION_MANIFEST = PROCESSED_DIR / "validation_clips.csv"
TRAIN_X3D_FILE = FEATURES_DIR / "train_x3d_multiwindow_features.npz"
VALIDATION_X3D_FILE = FEATURES_DIR / "validation_x3d_multiwindow_features.npz"
TRAIN_MOTION_FILE = FEATURES_DIR / "train_motion_features.npz"
VALIDATION_MOTION_FILE = FEATURES_DIR / "validation_motion_features.npz"

REPORT_FILE = REPORTS_DIR / "motion_fusion_evaluation.json"
OOF_FILE = REPORTS_DIR / "motion_fusion_oof_predictions.csv"
VALIDATION_FILE = REPORTS_DIR / "motion_fusion_validation_predictions.csv"
MODEL_FILE = MODELS_DIR / "securevision_motion_clip_classifier.joblib"


def resolve_video_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = AI_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    return path


def extract_manifest_features(
    manifest_path: Path,
    output_path: Path,
) -> dict[str, np.ndarray]:
    manifest = pd.read_csv(manifest_path)
    names = motion_feature_names()
    rows = []

    for index, row in manifest.iterrows():
        rows.append(
            extract_motion_features(
                resolve_video_path(str(row["video_path"])),
                float(row["start_sec"]),
                float(row["end_sec"]),
            )
        )
        if (index + 1) % 10 == 0 or index + 1 == len(manifest):
            print(f"Motion features: {index + 1}/{len(manifest)}")

    matrix = np.stack(rows).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=matrix,
        y=manifest["label"].to_numpy(dtype=np.int64),
        clip_ids=manifest["clip_id"].to_numpy(dtype=str),
        content_ids=manifest["content_id"].to_numpy(dtype=str),
        video_ids=manifest["canonical_video_id"].to_numpy(dtype=str),
        feature_names=np.asarray(names),
    )
    return {
        "X": matrix,
        "y": manifest["label"].to_numpy(dtype=np.int64),
        "clip_ids": manifest["clip_id"].to_numpy(dtype=str),
        "content_ids": manifest["content_id"].to_numpy(dtype=str),
        "video_ids": manifest["canonical_video_id"].to_numpy(dtype=str),
    }


def load_or_extract_motion(
    manifest_path: Path,
    feature_path: Path,
) -> dict[str, np.ndarray]:
    if not feature_path.exists():
        return extract_manifest_features(manifest_path, feature_path)

    data = np.load(feature_path, allow_pickle=False)
    return {key: data[key] for key in ("X", "y", "clip_ids", "content_ids", "video_ids")}


def load_x3d(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {
        "X": data["X_mean"].astype(np.float32),
        "clip_ids": data["clip_ids"].astype(str),
    }


def align_x3d(
    x3d: dict[str, np.ndarray],
    target_clip_ids: np.ndarray,
) -> np.ndarray:
    index = {clip_id: row for row, clip_id in enumerate(x3d["clip_ids"])}
    missing = [clip_id for clip_id in target_clip_ids if clip_id not in index]
    if missing:
        raise ValueError(f"X3D features missing clip ids: {missing[:5]}")
    return np.stack([x3d["X"][index[clip_id]] for clip_id in target_clip_ids])


def build_x3d_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=16, random_state=RANDOM_STATE)),
            (
                "classifier",
                LogisticRegression(
                    C=0.0003,
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_motion_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=8, random_state=RANDOM_STATE)),
            (
                "classifier",
                LogisticRegression(
                    C=0.01,
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_fusion_pipeline(x3d_dim: int, motion_dim: int) -> Pipeline:
    features = ColumnTransformer(
        [
            (
                "x3d",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=16, random_state=RANDOM_STATE)),
                    ]
                ),
                slice(0, x3d_dim),
            ),
            (
                "motion",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=8, random_state=RANDOM_STATE)),
                    ]
                ),
                slice(x3d_dim, x3d_dim + motion_dim),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    C=0.01,
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def inverse_group_weights(groups: np.ndarray) -> np.ndarray:
    values, counts = np.unique(groups, return_counts=True)
    count_by_group = dict(zip(values, counts))
    weights = np.asarray([1.0 / count_by_group[group] for group in groups])
    return weights / weights.mean()


def positive_probabilities(model: Pipeline, X: np.ndarray) -> np.ndarray:
    classes = model.named_steps["classifier"].classes_
    positive_index = int(np.where(classes == 1)[0][0])
    return model.predict_proba(X)[:, positive_index]


def aggregate_videos(
    y: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    video_ids: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "content_id": groups.astype(str),
            "video_id": video_ids.astype(str),
            "label": y.astype(int),
            "probability": probabilities.astype(float),
        }
    )
    return (
        frame.groupby("content_id", as_index=False)
        .agg(
            video_id=("video_id", "first"),
            label=("label", "max"),
            probability=("probability", "max"),
            mean_probability=("probability", "mean"),
            clip_count=("label", "size"),
        )
    )


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    prediction = (probability >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "samples": int(len(y)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "pr_auc": float(average_precision_score(y, probability)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    candidates = []
    for threshold in np.arange(0.10, 0.91, 0.05):
        score = metrics(y, probability, float(threshold))
        if score["recall"] >= 0.80:
            candidates.append(score)
    if not candidates:
        return 0.50
    best = max(
        candidates,
        key=lambda item: (
            item["balanced_accuracy"],
            item["specificity"],
            item["pr_auc"],
        ),
    )
    return float(best["threshold"])


def grouped_oof(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    factory: Callable[[], Pipeline],
) -> np.ndarray:
    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    output = np.zeros(len(y), dtype=np.float64)
    for train_indices, validation_indices in splitter.split(X, y, groups):
        model = factory()
        weights = inverse_group_weights(groups[train_indices])
        model.fit(
            X[train_indices],
            y[train_indices],
            classifier__sample_weight=weights,
        )
        output[validation_indices] = positive_probabilities(model, X[validation_indices])
    return output


def main() -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_motion = load_or_extract_motion(TRAIN_MANIFEST, TRAIN_MOTION_FILE)
    validation_motion = load_or_extract_motion(VALIDATION_MANIFEST, VALIDATION_MOTION_FILE)
    train_x3d = align_x3d(load_x3d(TRAIN_X3D_FILE), train_motion["clip_ids"])
    validation_x3d = align_x3d(
        load_x3d(VALIDATION_X3D_FILE), validation_motion["clip_ids"]
    )

    train_fusion = np.concatenate([train_x3d, train_motion["X"]], axis=1)
    validation_fusion = np.concatenate(
        [validation_x3d, validation_motion["X"]], axis=1
    )
    y = train_motion["y"]
    groups = train_motion["content_ids"]

    experiments = {
        "x3d_reference": (train_x3d, build_x3d_pipeline),
        "motion_only": (train_motion["X"], build_motion_pipeline),
        "x3d_motion_fusion": (
            train_fusion,
            lambda: build_fusion_pipeline(train_x3d.shape[1], train_motion["X"].shape[1]),
        ),
    }

    summaries: dict[str, Any] = {}
    oof_rows = []
    for name, (matrix, factory) in experiments.items():
        print(f"Grouped OOF: {name}")
        probability = grouped_oof(matrix, y, groups, factory)
        videos = aggregate_videos(
            y,
            probability,
            groups,
            train_motion["video_ids"],
        )
        threshold = choose_threshold(
            videos["label"].to_numpy(dtype=np.int64),
            videos["probability"].to_numpy(dtype=np.float64),
        )
        summary = metrics(
            videos["label"].to_numpy(dtype=np.int64),
            videos["probability"].to_numpy(dtype=np.float64),
            threshold,
        )
        summaries[name] = summary
        for row_index, clip_id in enumerate(train_motion["clip_ids"]):
            oof_rows.append(
                {
                    "experiment": name,
                    "clip_id": clip_id,
                    "content_id": groups[row_index],
                    "video_id": train_motion["video_ids"][row_index],
                    "label": int(y[row_index]),
                    "probability": float(probability[row_index]),
                }
            )

    eligible = [
        (name, summary)
        for name, summary in summaries.items()
        if summary["recall"] >= 0.80
    ]
    selected_name, selected_oof = max(
        eligible,
        key=lambda item: (
            item[1]["balanced_accuracy"],
            item[1]["specificity"],
            item[1]["roc_auc"],
        ),
    )

    selected_matrix, selected_factory = experiments[selected_name]
    selected_model = selected_factory()
    selected_model.fit(
        selected_matrix,
        y,
        classifier__sample_weight=inverse_group_weights(groups),
    )

    if selected_name == "x3d_reference":
        validation_matrix = validation_x3d
    elif selected_name == "motion_only":
        validation_matrix = validation_motion["X"]
    else:
        validation_matrix = validation_fusion

    validation_probability = positive_probabilities(selected_model, validation_matrix)
    validation_videos = aggregate_videos(
        validation_motion["y"],
        validation_probability,
        validation_motion["content_ids"],
        validation_motion["video_ids"],
    )
    validation_summary = metrics(
        validation_videos["label"].to_numpy(dtype=np.int64),
        validation_videos["probability"].to_numpy(dtype=np.float64),
        float(selected_oof["threshold"]),
    )

    payload = {
        "pipeline": selected_model,
        "threshold": float(selected_oof["threshold"]),
        "feature_strategy": selected_name,
        "x3d_feature_dim": int(train_x3d.shape[1]),
        "motion_feature_dim": int(train_motion["X"].shape[1]),
        "motion_num_frames": 16,
        "development_validation_used_for_selection": False,
    }
    joblib.dump(payload, MODEL_FILE)

    pd.DataFrame(oof_rows).to_csv(OOF_FILE, index=False)
    validation_videos.to_csv(VALIDATION_FILE, index=False)
    report = {
        "methodology": {
            "selection": "four-fold stratified grouped OOF on training videos",
            "video_aggregation": "maximum clip probability",
            "minimum_selection_recall": 0.80,
            "development_validation_used_for_selection": False,
        },
        "features": {
            "motion_dimension": int(train_motion["X"].shape[1]),
            "motion_frames_per_six_second_clip": 16,
            "x3d_dimension": int(train_x3d.shape[1]),
        },
        "oof_experiments": summaries,
        "selected_experiment": selected_name,
        "selected_oof": selected_oof,
        "development_validation": validation_summary,
        "model_file": str(MODEL_FILE),
    }
    REPORT_FILE.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
