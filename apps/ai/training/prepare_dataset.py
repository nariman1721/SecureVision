from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import cv2
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

AI_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = AI_ROOT / "data" / "raw"
LABELS_FILE = AI_ROOT / "data" / "labels" / "open_labels.xlsx"
PROCESSED_DIR = AI_ROOT / "data" / "processed"

METADATA_FILE = PROCESSED_DIR / "metadata.csv"
ISSUES_FILE = PROCESSED_DIR / "dataset_issues.csv"
UNLABELED_FILE = PROCESSED_DIR / "unlabeled_raw_files.csv"
SUMMARY_FILE = PROCESSED_DIR / "dataset_summary.json"


# ---------------------------------------------------------
# Filename normalization
# ---------------------------------------------------------

def normalize_video_id(value: Any) -> str | None:
    """
    Normalize labels such as:

    X-2
    Х-3  <- Cyrillic Х
    x-4
    X–5  <- different dash

    into:

    X-2
    X-3
    X-4
    X-5
    """

    if value is None or pd.isna(value):
        return None

    text = str(value).strip()

    # Remove extension if one exists.
    text = Path(text).stem

    # Cyrillic Х/х -> Latin X/x.
    text = text.replace("Х", "X")
    text = text.replace("х", "x")

    # Normalize different Unicode dashes.
    for dash in ["–", "—", "−", "-"]:
        text = text.replace(dash, "-")

    # Remove whitespace.
    text = re.sub(r"\s+", "", text)

    text = text.upper()

    # Make X1 / X-1 / X_1 consistent.
    match = re.fullmatch(r"X[-_]?(\d+)", text)

    if match:
        number = int(match.group(1))
        return f"X-{number}"

    return text


# ---------------------------------------------------------
# Timestamp parser
# ---------------------------------------------------------

def parse_timecode(value: Any) -> float | None:
    """
    Supported examples:

    22 секунда
    10 секунд
    5 сек
    1минута 8 секунда
    1 минута 8 секунд
    68
    00:10
    01:08
    """

    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().lower()

    if not text:
        return None

    text = text.replace(",", ".")

    # HH:MM:SS or MM:SS
    if ":" in text:
        parts = text.split(":")

        try:
            numbers = [float(part) for part in parts]

            if len(numbers) == 2:
                minutes, seconds = numbers
                return minutes * 60 + seconds

            if len(numbers) == 3:
                hours, minutes, seconds = numbers
                return hours * 3600 + minutes * 60 + seconds

        except ValueError:
            pass

    minute_match = re.search(
        r"(\d+(?:\.\d+)?)\s*мин",
        text,
    )

    second_match = re.search(
        r"(\d+(?:\.\d+)?)\s*сек",
        text,
    )

    if minute_match or second_match:
        minutes = (
            float(minute_match.group(1))
            if minute_match
            else 0.0
        )

        seconds = (
            float(second_match.group(1))
            if second_match
            else 0.0
        )

        return minutes * 60 + seconds

    # Final fallback: first numeric value.
    number_match = re.search(
        r"\d+(?:\.\d+)?",
        text,
    )

    if number_match:
        return float(number_match.group(0))

    raise ValueError(
        f"Unable to parse timestamp: {value!r}"
    )


# ---------------------------------------------------------
# Label parser
# ---------------------------------------------------------

def normalize_label(value: Any) -> int | None:
    """
    violation      -> 1
    no violation   -> 0
    """

    if value is None or pd.isna(value):
        return None

    text = str(value).strip().lower()

    if "без" in text and "наруш" in text:
        return 0

    if "наруш" in text:
        return 1

    return None


# ---------------------------------------------------------
# File hashing
# ---------------------------------------------------------

def calculate_sha256(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


# ---------------------------------------------------------
# Video inspection
# ---------------------------------------------------------

def inspect_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        capture.release()

        return {
            "readable": False,
            "fps": None,
            "frame_count": None,
            "duration_sec": None,
            "width": None,
            "height": None,
        }

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    # Actually try reading one frame.
    success, _ = capture.read()

    capture.release()

    readable = bool(
        success
        and fps > 0
        and frame_count > 0
    )

    duration = (
        frame_count / fps
        if fps > 0
        else None
    )

    return {
        "readable": readable,
        "fps": round(fps, 3)
        if fps > 0
        else None,
        "frame_count": frame_count,
        "duration_sec": round(duration, 3)
        if duration is not None
        else None,
        "width": width,
        "height": height,
    }


# ---------------------------------------------------------
# Excel helpers
# ---------------------------------------------------------

def find_column(
    dataframe: pd.DataFrame,
    keyword: str,
) -> str:
    keyword = keyword.lower()

    for column in dataframe.columns:
        if keyword in str(column).strip().lower():
            return column

    raise ValueError(
        f"Could not find column containing {keyword!r}. "
        f"Available columns: {list(dataframe.columns)}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("SecureVision — Phase 1 Dataset Preparation")
    print("=" * 72)

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not LABELS_FILE.exists():
        raise FileNotFoundError(
            f"Labels file not found:\n{LABELS_FILE}"
        )

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw video directory not found:\n{RAW_DIR}"
        )

    # -----------------------------------------------------
    # Read labels
    # -----------------------------------------------------

    labels_df = pd.read_excel(
        LABELS_FILE,
        sheet_name="labels",
    )

    file_column = find_column(
        labels_df,
        "наименование",
    )

    time_column = find_column(
        labels_df,
        "тайм",
    )

    label_column = find_column(
        labels_df,
        "нарушение",
    )

    print()
    print(f"Labels loaded: {len(labels_df)}")
    print(f"File column:  {file_column}")
    print(f"Time column:  {time_column}")
    print(f"Label column: {label_column}")

    # -----------------------------------------------------
    # Index raw videos
    # -----------------------------------------------------

    raw_video_paths = sorted(
        RAW_DIR.glob("*.mov")
    )

    print()
    print(
        f"Raw .mov files found: "
        f"{len(raw_video_paths)}"
    )

    raw_index: dict[str, Path] = {}
    duplicate_normalized_names: list[str] = []

    for path in raw_video_paths:
        normalized_id = normalize_video_id(
            path.stem
        )

        if normalized_id is None:
            continue

        if normalized_id in raw_index:
            duplicate_normalized_names.append(
                normalized_id
            )
        else:
            raw_index[normalized_id] = path

    # -----------------------------------------------------
    # Build dataset rows
    # -----------------------------------------------------

    records: list[dict[str, Any]] = []

    for _, row in labels_df.iterrows():
        label_filename = row[file_column]

        video_id = normalize_video_id(
            label_filename
        )

        label = normalize_label(
            row[label_column]
        )

        timestamp_error = None

        try:
            timecode_sec = parse_timecode(
                row[time_column]
            )
        except ValueError as error:
            timecode_sec = None
            timestamp_error = str(error)

        path = (
            raw_index.get(video_id)
            if video_id
            else None
        )

        file_found = path is not None

        metadata = {
            "readable": False,
            "fps": None,
            "frame_count": None,
            "duration_sec": None,
            "width": None,
            "height": None,
        }

        actual_filename = None
        file_size_mb = None
        sha256 = None

        if path is not None:
            actual_filename = path.name

            file_size_mb = round(
                path.stat().st_size
                / (1024 * 1024),
                3,
            )

            metadata = inspect_video(path)

            sha256 = calculate_sha256(path)

        issues: list[str] = []

        if video_id is None:
            issues.append(
                "invalid_video_id"
            )

        if not file_found:
            issues.append(
                "video_file_not_found"
            )

        if label is None:
            issues.append(
                "invalid_label"
            )

        if timestamp_error:
            issues.append(
                "invalid_timestamp_format"
            )

        if (
            label == 1
            and timecode_sec is None
        ):
            issues.append(
                "violation_missing_timestamp"
            )

        if (
            label == 0
            and timecode_sec is not None
        ):
            issues.append(
                "no_violation_has_timestamp"
            )

        if (
            file_found
            and not metadata["readable"]
        ):
            issues.append(
                "video_unreadable"
            )

        duration = metadata[
            "duration_sec"
        ]

        if (
            timecode_sec is not None
            and duration is not None
            and (
                timecode_sec < 0
                or timecode_sec > duration
            )
        ):
            issues.append(
                "timestamp_outside_video"
            )

        # Dataset expectations from the task description.
        duration_expected = (
            duration is not None
            and 10 <= duration <= 145
        )

        fps = metadata["fps"]

        fps_expected = (
            fps is not None
            and fps >= 29
        )

        if (
            file_found
            and metadata["readable"]
            and not duration_expected
        ):
            issues.append(
                "duration_outside_expected_range"
            )

        if (
            file_found
            and metadata["readable"]
            and not fps_expected
        ):
            issues.append(
                "fps_below_expected_range"
            )

        records.append(
            {
                "video_id": video_id,
                "label_filename": str(
                    label_filename
                ),
                "actual_filename": actual_filename,
                "label_text": str(
                    row[label_column]
                ),
                "label": label,
                "timecode_sec": timecode_sec,
                "file_found": file_found,
                "readable": metadata[
                    "readable"
                ],
                "duration_sec": duration,
                "fps": fps,
                "frame_count": metadata[
                    "frame_count"
                ],
                "width": metadata[
                    "width"
                ],
                "height": metadata[
                    "height"
                ],
                "file_size_mb": file_size_mb,
                "sha256": sha256,
                "duration_expected": (
                    duration_expected
                ),
                "fps_expected": (
                    fps_expected
                ),
                "issues": ";".join(issues),
            }
        )

    dataset = pd.DataFrame(records)

    if not dataset.empty:
        dataset["label"] = (
            dataset["label"]
            .astype("Int64")
        )

    # -----------------------------------------------------
    # Duplicate-content detection
    # -----------------------------------------------------

    dataset["duplicate_content"] = False

    valid_hash_mask = dataset[
        "sha256"
    ].notna()

    dataset.loc[
        valid_hash_mask,
        "duplicate_content",
    ] = (
        dataset.loc[
            valid_hash_mask,
            "sha256",
        ]
        .duplicated(keep=False)
        .to_numpy()
    )

    duplicate_mask = dataset[
        "duplicate_content"
    ]

    dataset.loc[
        duplicate_mask,
        "issues",
    ] = dataset.loc[
        duplicate_mask,
        "issues",
    ].apply(
        lambda value:
        (
            value + ";duplicate_video_content"
            if value
            else "duplicate_video_content"
        )
    )

    # -----------------------------------------------------
    # Find raw videos that have no label
    # -----------------------------------------------------

    labelled_ids = set(
        dataset[
            "video_id"
        ].dropna()
    )

    unlabeled_records = []

    for normalized_id, path in raw_index.items():
        if normalized_id not in labelled_ids:
            unlabeled_records.append(
                {
                    "video_id": normalized_id,
                    "actual_filename": path.name,
                }
            )

    unlabeled_df = pd.DataFrame(
        unlabeled_records
    )

    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------

    dataset = dataset.sort_values(
        by="video_id",
        key=lambda series: (
            series.str.extract(
                r"(\d+)",
                expand=False,
            )
            .astype(float)
        ),
        na_position="last",
    )

    dataset.to_csv(
        METADATA_FILE,
        index=False,
    )

    issues_df = dataset[
        dataset["issues"].fillna("") != ""
    ].copy()

    issues_df.to_csv(
        ISSUES_FILE,
        index=False,
    )

    unlabeled_df.to_csv(
        UNLABELED_FILE,
        index=False,
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    violation_count = int(
        (dataset["label"] == 1).sum()
    )

    no_violation_count = int(
        (dataset["label"] == 0).sum()
    )

    summary = {
        "labels_total": int(
            len(dataset)
        ),
        "raw_videos_total": int(
            len(raw_video_paths)
        ),
        "matched_videos": int(
            dataset[
                "file_found"
            ].sum()
        ),
        "missing_videos": int(
            (
                ~dataset[
                    "file_found"
                ]
            ).sum()
        ),
        "readable_videos": int(
            dataset[
                "readable"
            ].sum()
        ),
        "violations": violation_count,
        "no_violations": (
            no_violation_count
        ),
        "issue_rows": int(
            len(issues_df)
        ),
        "unlabeled_raw_videos": int(
            len(unlabeled_df)
        ),
        "duplicate_content_files": int(
            dataset[
                "duplicate_content"
            ].sum()
        ),
        "duplicate_normalized_names": (
            duplicate_normalized_names
        ),
    }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -----------------------------------------------------
    # Console report
    # -----------------------------------------------------

    print()
    print("=" * 72)
    print("DATASET SUMMARY")
    print("=" * 72)

    for key, value in summary.items():
        print(
            f"{key:30} {value}"
        )

    print()
    print("=" * 72)
    print("LABEL DISTRIBUTION")
    print("=" * 72)

    print(
        dataset[
            "label_text"
        ].value_counts(
            dropna=False
        )
    )

    print()
    print("=" * 72)
    print("DATAFRAME INFO")
    print("=" * 72)

    dataset.info()

    print()
    print("=" * 72)
    print("NUMERIC DESCRIPTION")
    print("=" * 72)

    numeric_columns = [
        "timecode_sec",
        "duration_sec",
        "fps",
        "frame_count",
        "width",
        "height",
        "file_size_mb",
    ]

    print(
        dataset[
            numeric_columns
        ].describe().round(3)
    )

    print()
    print("=" * 72)
    print("FILES CREATED")
    print("=" * 72)

    print(METADATA_FILE)
    print(ISSUES_FILE)
    print(UNLABELED_FILE)
    print(SUMMARY_FILE)

    if len(issues_df) == 0:
        print()
        print(
            "✅ No dataset issues detected."
        )
    else:
        print()
        print(
            f"⚠️ {len(issues_df)} labelled "
            f"rows need attention."
        )

        print(
            issues_df[
                [
                    "video_id",
                    "actual_filename",
                    "issues",
                ]
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()