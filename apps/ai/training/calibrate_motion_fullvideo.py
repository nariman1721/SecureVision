from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score

from app.video.temporal_localizer import TemporalLocalizer


AI_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = AI_ROOT / "data" / "processed"
REPORTS_DIR = AI_ROOT / "reports"
SOURCE_MODEL_PATH = AI_ROOT / "models" / "securevision_motion_clip_classifier.joblib"
MODEL_PATH = AI_ROOT / "models" / "securevision_motion_classifier.joblib"

SPLITS_FILE = PROCESSED_DIR / "split_assignments.csv"
EVENTS_FILE = PROCESSED_DIR / "events.csv"
OUTPUT_JSON = REPORTS_DIR / "motion_fullvideo_calibration.json"
OUTPUT_CSV = REPORTS_DIR / "motion_fullvideo_scores.csv"

EVENT_RADIUS_SEC = 10.0
MINIMUM_VIDEO_RECALL = 0.80
MINIMUM_EVENT_RECALL = 0.80


def score_threshold(
    frame: pd.DataFrame,
    event_scores: list[float],
    threshold: float,
) -> dict[str, Any]:
    y = frame["label"].to_numpy(dtype=np.int64)
    prediction = (frame["video_score"].to_numpy(dtype=float) >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    event_recall = float(np.mean(np.asarray(event_scores) >= threshold))
    return {
        "threshold": float(threshold),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "video_recall": float(recall_score(y, prediction, zero_division=0)),
        "event_recall": event_recall,
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    assignments = pd.read_csv(SPLITS_FILE)
    train = assignments[assignments["split"].eq("train")].copy()
    events = pd.read_csv(EVENTS_FILE)
    localizer = TemporalLocalizer(classifier_path=SOURCE_MODEL_PATH)

    rows = []
    all_event_scores: list[float] = []
    for index, item in train.iterrows():
        video_id = str(item["canonical_video_id"])
        video_path = (AI_ROOT / "data" / "raw" / str(item["canonical_filename"])).resolve()
        result = localizer.localize(video_path, show_progress=False)
        candidates = result.get("top_candidates", [])
        video_score = max(
            (float(candidate["peak_probability"]) for candidate in candidates),
            default=0.0,
        )
        event_times = (
            events[events["content_id"].astype(str).eq(str(item["content_id"]))][
                "timecode_sec"
            ]
            .astype(float)
            .tolist()
        )
        event_scores = []
        for event_time in event_times:
            nearby = [
                float(candidate["peak_probability"])
                for candidate in candidates
                if abs(float(candidate["timecode_sec"]) - event_time) <= EVENT_RADIUS_SEC
            ]
            score = max(nearby, default=0.0)
            event_scores.append(score)
            all_event_scores.append(score)

        rows.append(
            {
                "video_id": video_id,
                "content_id": str(item["content_id"]),
                "label": int(item["label"]),
                "video_score": float(video_score),
                "event_scores": json.dumps(event_scores),
                "candidate_count": int(len(candidates)),
                "windows_scanned": int(result["windows_scanned"]),
            }
        )
        print(
            f"{len(rows)}/{len(train)}",
            video_id,
            f"label={int(item['label'])}",
            f"score={video_score:.4f}",
        )

    frame = pd.DataFrame(rows)
    observed = frame["video_score"].to_numpy(dtype=float)
    thresholds = sorted(
        {
            *np.round(np.arange(0.10, 0.951, 0.025), 3).tolist(),
            *np.round(observed, 6).tolist(),
        }
    )
    sweep = [
        score_threshold(frame, all_event_scores, float(threshold))
        for threshold in thresholds
    ]
    eligible = [
        item
        for item in sweep
        if item["video_recall"] >= MINIMUM_VIDEO_RECALL
        and item["event_recall"] >= MINIMUM_EVENT_RECALL
    ]
    if not eligible:
        raise RuntimeError("No threshold satisfies the recall constraints.")

    selected = max(
        eligible,
        key=lambda item: (
            item["balanced_accuracy"],
            item["specificity"],
            item["event_recall"],
            item["threshold"],
        ),
    )

    payload = joblib.load(SOURCE_MODEL_PATH)
    payload["threshold"] = float(selected["threshold"])
    payload["fullvideo_calibration"] = {
        "split": "train",
        "development_validation_used": False,
        "event_radius_sec": EVENT_RADIUS_SEC,
        "minimum_video_recall": MINIMUM_VIDEO_RECALL,
        "minimum_event_recall": MINIMUM_EVENT_RECALL,
        "selected": selected,
    }
    joblib.dump(payload, MODEL_PATH)

    report = {
        "methodology": {
            "split": "train full-video scans",
            "development_validation_used": False,
            "event_radius_sec": EVENT_RADIUS_SEC,
            "minimum_video_recall": MINIMUM_VIDEO_RECALL,
            "minimum_event_recall": MINIMUM_EVENT_RECALL,
        },
        "dataset": {
            "videos": int(len(frame)),
            "positive_videos": int(frame["label"].sum()),
            "negative_videos": int(frame["label"].eq(0).sum()),
            "events": int(len(all_event_scores)),
        },
        "selected": selected,
        "sweep": sweep,
        "model_file": str(MODEL_PATH),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "sweep"}, indent=2))


if __name__ == "__main__":
    main()
