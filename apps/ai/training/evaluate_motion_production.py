from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.video.production_inference import (
    ProductionVideoInference,
)


AI_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = AI_ROOT / "data" / "processed"
REPORTS_DIR = AI_ROOT / "reports"
MODEL_PATH = AI_ROOT / "models" / "securevision_motion_classifier.joblib"

SPLITS_FILE = PROCESSED_DIR / "split_assignments.csv"
EVENTS_FILE = PROCESSED_DIR / "events.csv"
OUTPUT_JSON = REPORTS_DIR / "motion_production_validation.json"
OUTPUT_CSV = REPORTS_DIR / "motion_production_validation.csv"


def main() -> None:
    assignments = pd.read_csv(SPLITS_FILE)
    validation = assignments[assignments["split"].eq("validation")].copy()
    events = pd.read_csv(EVENTS_FILE)
    inference = ProductionVideoInference(
        classifier_path=MODEL_PATH,
        max_violations=3,
    )

    rows = []
    localization_errors = []
    for _, item in validation.iterrows():
        video_id = str(item["canonical_video_id"])
        video_path = (AI_ROOT / "data" / "raw" / str(item["canonical_filename"])).resolve()
        result = inference.analyze(video_path, include_evidence=False)
        prediction = int(bool(result["detected"]))
        label = int(item["label"])
        predicted_times = [
            float(violation["timecode_sec"])
            for violation in result["violations"]
        ]

        event_times = (
            events[events["content_id"].astype(str).eq(str(item["content_id"]))][
                "timecode_sec"
            ]
            .astype(float)
            .tolist()
        )
        event_errors = []
        for event_time in event_times:
            if predicted_times:
                error = min(abs(value - event_time) for value in predicted_times)
                event_errors.append(float(error))
                localization_errors.append(float(error))

        rows.append(
            {
                "video_id": video_id,
                "label": label,
                "prediction": prediction,
                "correct": bool(label == prediction),
                "predicted_timecodes": json.dumps(predicted_times),
                "event_timecodes": json.dumps(event_times),
                "event_errors_sec": json.dumps(event_errors),
                "processing_time_sec": float(result["analysis"]["processing_time_sec"]),
                "windows_scanned": int(result["analysis"]["windows_scanned"]),
            }
        )
        print(
            video_id,
            f"label={label}",
            f"prediction={prediction}",
            f"times={predicted_times}",
        )

    frame = pd.DataFrame(rows)
    y = frame["label"].to_numpy(dtype=np.int64)
    prediction = frame["prediction"].to_numpy(dtype=np.int64)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()

    report = {
        "methodology": {
            "split": "development validation",
            "used_for_model_or_threshold_selection": False,
            "classifier": MODEL_PATH.name,
            "feature_strategy": inference.localizer.feature_strategy,
            "threshold": float(inference.localizer.threshold),
            "scan_window_sec": float(inference.localizer.scan_window_sec),
            "scan_stride_sec": float(inference.localizer.scan_stride_sec),
        },
        "classification": {
            "videos": int(len(frame)),
            "accuracy": float(accuracy_score(y, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "f1": float(f1_score(y, prediction, zero_division=0)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "localization": {
            "matched_events": int(len(localization_errors)),
            "mean_error_sec": float(np.mean(localization_errors)) if localization_errors else None,
            "median_error_sec": float(np.median(localization_errors)) if localization_errors else None,
            "within_5_sec": int(sum(error <= 5.0 for error in localization_errors)),
            "within_10_sec": int(sum(error <= 10.0 for error in localization_errors)),
        },
        "runtime": {
            "mean_processing_time_sec": float(frame["processing_time_sec"].mean()),
            "max_processing_time_sec": float(frame["processing_time_sec"].max()),
            "all_under_300_sec": bool(frame["processing_time_sec"].lt(300.0).all()),
        },
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
