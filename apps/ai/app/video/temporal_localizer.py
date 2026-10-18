from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np

from app.model.loader import load_model
from app.model.motion_features import (
    extract_motion_features,
)
from app.model.multiwindow import (
    extract_multiwindow_feature_set,
    prepare_multiwindow_batch_from_manifest,
)


AI_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DEFAULT_CLASSIFIER_PATH = (
    AI_ROOT
    / "models"
    / "securevision_final_classifier.joblib"
)

DEFAULT_SCAN_WINDOW_SEC = 6.0
DEFAULT_SCAN_STRIDE_SEC = 2.0
DEFAULT_SMOOTHING_RADIUS = 1

DEFAULT_SUPPORT_RADIUS = 1
DEFAULT_TOP_CANDIDATES = 5


# =========================================================
# VIDEO METADATA
# =========================================================

def get_video_duration_sec(
    video_path: str | Path,
) -> float:
    path = Path(video_path).resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Video not found: {path}"
        )

    capture = cv2.VideoCapture(
        str(path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {path}"
        )

    fps = float(
        capture.get(
            cv2.CAP_PROP_FPS
        )
    )

    frame_count = float(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    capture.release()

    if fps <= 0:
        raise RuntimeError(
            f"Invalid FPS: {path}"
        )

    if frame_count <= 0:
        raise RuntimeError(
            f"Invalid frame count: {path}"
        )

    return float(
        frame_count / fps
    )


# =========================================================
# SCAN WINDOWS
# =========================================================

def build_scan_windows(
    duration_sec: float,
    window_sec: float = DEFAULT_SCAN_WINDOW_SEC,
    stride_sec: float = DEFAULT_SCAN_STRIDE_SEC,
) -> list[tuple[float, float]]:
    duration_sec = float(
        duration_sec
    )

    window_sec = float(
        window_sec
    )

    stride_sec = float(
        stride_sec
    )

    if duration_sec <= 0:
        raise ValueError(
            "duration_sec must be > 0"
        )

    if window_sec <= 0:
        raise ValueError(
            "window_sec must be > 0"
        )

    if stride_sec <= 0:
        raise ValueError(
            "stride_sec must be > 0"
        )

    if duration_sec <= window_sec:
        return [
            (
                0.0,
                round(
                    duration_sec,
                    3,
                ),
            )
        ]

    last_start = (
        duration_sec
        - window_sec
    )

    starts = []

    current = 0.0

    while (
        current
        <= last_start + 1e-9
    ):
        starts.append(
            current
        )

        current += stride_sec

    if (
        not starts
        or abs(
            starts[-1]
            - last_start
        )
        > 1e-6
    ):
        starts.append(
            last_start
        )

    unique_starts = []

    for start in starts:
        start = round(
            max(
                0.0,
                start,
            ),
            3,
        )

        if (
            not unique_starts
            or abs(
                start
                - unique_starts[-1]
            )
            > 1e-6
        ):
            unique_starts.append(
                start
            )

    return [
        (
            start,
            round(
                min(
                    start
                    + window_sec,
                    duration_sec,
                ),
                3,
            ),
        )
        for start in unique_starts
    ]


# =========================================================
# LEGACY SMOOTHING
# =========================================================

def smooth_probabilities(
    probabilities: list[float],
    radius: int = 1,
) -> list[float]:
    values = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if len(values) == 0:
        return []

    if radius < 0:
        raise ValueError(
            "radius cannot be negative"
        )

    if radius == 0:
        return (
            values
            .astype(float)
            .tolist()
        )

    result = np.empty_like(
        values
    )

    for index in range(
        len(values)
    ):
        left = max(
            0,
            index - radius,
        )

        right = min(
            len(values),
            index + radius + 1,
        )

        result[index] = (
            values[
                left:right
            ]
            .mean()
        )

    return (
        result
        .astype(float)
        .tolist()
    )


# =========================================================
# IMPROVED TEMPORAL SCORING
# =========================================================

def build_baseline_normalized_scores(
    probabilities: list[float],
    support_radius: int = DEFAULT_SUPPORT_RADIUS,
) -> dict[str, Any]:
    values = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if len(values) == 0:
        raise ValueError(
            "Empty probability timeline."
        )

    baseline = float(
        np.median(
            values
        )
    )

    centered = (
        values
        - baseline
    )

    mad = float(
        np.median(
            np.abs(
                centered
            )
        )
    )

    # Avoid unstable division on almost-flat timelines.
    robust_scale = max(
        1.4826 * mad,
        0.02,
    )

    normalized = (
        centered
        / robust_scale
    )

    support_scores = np.zeros_like(
        normalized
    )

    persistence = np.zeros_like(
        normalized
    )

    candidate_scores = np.zeros_like(
        normalized
    )

    for index in range(
        len(values)
    ):
        left = max(
            0,
            index - support_radius,
        )

        right = min(
            len(values),
            index + support_radius + 1,
        )

        neighborhood = (
            normalized[
                left:right
            ]
        )

        centered_neighborhood = (
            centered[
                left:right
            ]
        )

        support = float(
            np.mean(
                neighborhood
            )
        )

        persistent_fraction = float(
            np.mean(
                centered_neighborhood
                > 0
            )
        )

        # 35% current window
        # 65% neighboring support
        score = (
            0.35
            * float(
                normalized[index]
            )
            + 0.65
            * support
        )

        # Penalize isolated spikes.
        if persistent_fraction < 0.5:
            score *= 0.60

        support_scores[
            index
        ] = support

        persistence[
            index
        ] = persistent_fraction

        candidate_scores[
            index
        ] = score

    return {
        "baseline_probability": (
            baseline
        ),

        "robust_scale": (
            robust_scale
        ),

        "centered_probability": (
            centered
            .astype(float)
            .tolist()
        ),

        "normalized_score": (
            normalized
            .astype(float)
            .tolist()
        ),

        "support_score": (
            support_scores
            .astype(float)
            .tolist()
        ),

        "persistence": (
            persistence
            .astype(float)
            .tolist()
        ),

        "candidate_score": (
            candidate_scores
            .astype(float)
            .tolist()
        ),
    }


# =========================================================
# CLASSIFIER
# =========================================================

def get_positive_probability(
    pipeline,
    feature_vector: np.ndarray,
) -> float:
    feature_vector = np.asarray(
        feature_vector,
        dtype=np.float32,
    )

    if (
        feature_vector.ndim != 2
        or feature_vector.shape[0] != 1
    ):
        raise ValueError(
            "Expected one 2D feature row, got "
            f"{feature_vector.shape}"
        )

    expected_features = getattr(
        pipeline,
        "n_features_in_",
        None,
    )

    if (
        expected_features is not None
        and feature_vector.shape[1]
        != int(expected_features)
    ):
        raise ValueError(
            "Feature dimension mismatch: "
            f"model expects {expected_features}, "
            f"received {feature_vector.shape[1]}."
        )

    probabilities = (
        pipeline.predict_proba(
            feature_vector
        )
    )

    classifier = (
        pipeline.named_steps[
            "classifier"
        ]
    )

    positive_indices = np.where(
        classifier.classes_
        == 1
    )[0]

    if len(
        positive_indices
    ) != 1:
        raise RuntimeError(
            "Positive class 1 not found."
        )

    positive_index = int(
        positive_indices[0]
    )

    return float(
        probabilities[
            0,
            positive_index,
        ]
    )


# =========================================================
# TEMPORAL LOCALIZER
# =========================================================

class TemporalLocalizer:
    def __init__(
        self,
        classifier_path: (
            str
            | Path
        ) = DEFAULT_CLASSIFIER_PATH,
        scan_window_sec: float = DEFAULT_SCAN_WINDOW_SEC,
        scan_stride_sec: float = DEFAULT_SCAN_STRIDE_SEC,
        smoothing_radius: int = DEFAULT_SMOOTHING_RADIUS,
        support_radius: int = DEFAULT_SUPPORT_RADIUS,
        top_candidates: int = DEFAULT_TOP_CANDIDATES,
    ):
        self.classifier_path = Path(
            classifier_path
        ).resolve()

        if not self.classifier_path.exists():
            raise FileNotFoundError(
                f"Classifier not found: "
                f"{self.classifier_path}"
            )

        self.scan_window_sec = float(
            scan_window_sec
        )

        self.scan_stride_sec = float(
            scan_stride_sec
        )

        self.smoothing_radius = int(
            smoothing_radius
        )

        self.support_radius = int(
            support_radius
        )

        self.top_candidates = int(
            top_candidates
        )

        payload = joblib.load(
            self.classifier_path
        )

        required = [
            "pipeline",
            "threshold",
            "feature_strategy",
        ]

        missing = [
            key
            for key in required
            if key not in payload
        ]

        if missing:
            raise RuntimeError(
                f"Classifier payload missing: "
                f"{missing}"
            )

        self.feature_strategy = str(
            payload[
                "feature_strategy"
            ]
        )

        self.pipeline = (
            payload[
                "pipeline"
            ]
        )

        self.threshold = float(
            payload[
                "threshold"
            ]
        )

        self.motion_num_frames = int(
            payload.get(
                "motion_num_frames",
                16,
            )
        )

        if self.feature_strategy == "multi_mean":
            if int(
                payload.get(
                    "x3d_windows",
                    0,
                )
            ) != 3:
                raise RuntimeError(
                    "Expected 3 X3D windows."
                )

            (
                self.model,
                self.device,
            ) = load_model(
                pretrained=True
            )

        elif self.feature_strategy == "motion_only":
            self.model = None
            self.device = "cpu"

        else:
            raise RuntimeError(
                "Unsupported feature strategy: "
                f"{self.feature_strategy}"
            )

    # =====================================================
    # ONE WINDOW
    # =====================================================

    def analyze_window(
        self,
        video_path: str | Path,
        start_sec: float,
        end_sec: float,
        window_index: int,
    ) -> dict[str, Any]:
        if self.feature_strategy == "motion_only":
            motion_features = (
                extract_motion_features(
                    video_path,
                    start_sec,
                    end_sec,
                    num_frames=(
                        self.motion_num_frames
                    ),
                )
                .reshape(1, -1)
                .astype(np.float32)
            )

            probability = (
                get_positive_probability(
                    self.pipeline,
                    motion_features,
                )
            )

            center_sec = (
                float(start_sec)
                + float(end_sec)
            ) / 2.0

            return {
                "window_index": int(
                    window_index
                ),
                "start_sec": float(
                    start_sec
                ),
                "end_sec": float(
                    end_sec
                ),
                "center_sec": float(
                    round(
                        center_sec,
                        3,
                    )
                ),
                "probability": float(
                    probability
                ),
                "internal_window_centers_sec": [],
            }

        row = {
            "clip_id": (
                f"scan_{window_index:04d}"
            ),

            "canonical_video_id": (
                Path(
                    video_path
                ).stem
            ),

            "clip_type": (
                "inference_scan"
            ),

            "label": -1,

            "video_path": str(
                Path(
                    video_path
                ).resolve()
            ),

            "start_sec": float(
                start_sec
            ),

            "end_sec": float(
                end_sec
            ),
        }

        (
            batch,
            metadata,
        ) = (
            prepare_multiwindow_batch_from_manifest(
                row=row,
                ai_root=AI_ROOT,
                device=self.device,
                num_windows=3,
            )
        )

        feature_set = (
            extract_multiwindow_feature_set(
                self.model,
                batch,
            )
        )

        mean_features = (
            feature_set[
                "mean"
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        probability = (
            get_positive_probability(
                self.pipeline,
                mean_features,
            )
        )

        center_sec = (
            float(
                start_sec
            )
            + float(
                end_sec
            )
        ) / 2.0

        return {
            "window_index": int(
                window_index
            ),

            "start_sec": float(
                start_sec
            ),

            "end_sec": float(
                end_sec
            ),

            "center_sec": float(
                round(
                    center_sec,
                    3,
                )
            ),

            "probability": float(
                probability
            ),

            "internal_window_centers_sec": (
                metadata[
                    "window_centers_sec"
                ]
            ),
        }

    # =====================================================
    # LEGACY EVENTS
    # =====================================================

    def _build_events(
        self,
        scan_results: list[
            dict[str, Any]
        ],
    ) -> list[dict[str, Any]]:
        positive_indices = [
            index
            for index, item
            in enumerate(
                scan_results
            )
            if (
                item[
                    "smoothed_probability"
                ]
                >= self.threshold
            )
        ]

        if not positive_indices:
            return []

        groups = []

        current = [
            positive_indices[0]
        ]

        for index in (
            positive_indices[1:]
        ):
            if (
                index
                == current[-1] + 1
            ):
                current.append(
                    index
                )
            else:
                groups.append(
                    current
                )

                current = [
                    index
                ]

        groups.append(
            current
        )

        events = []

        for event_index, group in enumerate(
            groups,
            start=1,
        ):
            items = [
                scan_results[index]
                for index in group
            ]

            peak = max(
                items,
                key=lambda item: (
                    item[
                        "smoothed_probability"
                    ]
                ),
            )

            events.append(
                {
                    "event_index": (
                        event_index
                    ),

                    "start_sec": float(
                        items[0][
                            "start_sec"
                        ]
                    ),

                    "end_sec": float(
                        items[-1][
                            "end_sec"
                        ]
                    ),

                    "timecode_sec": float(
                        peak[
                            "center_sec"
                        ]
                    ),

                    "probability": float(
                        peak[
                            "smoothed_probability"
                        ]
                    ),

                    "windows_in_region": int(
                        len(
                            items
                        )
                    ),
                }
            )

        return events

    # =====================================================
    # IMPROVED CANDIDATE REGIONS
    # =====================================================

    def _build_candidate_regions(
        self,
        scan_results: list[
            dict[str, Any]
        ],
    ) -> list[dict[str, Any]]:
        stable_indices = [
            index
            for index, item
            in enumerate(
                scan_results
            )
            if (
                item[
                    "candidate_score"
                ]
                > 0
                and item[
                    "persistence"
                ]
                >= 0.50
            )
        ]

        # Fallback:
        # if nothing is above baseline,
        # keep the strongest window.
        if not stable_indices:
            strongest_index = int(
                np.argmax(
                    [
                        item[
                            "candidate_score"
                        ]
                        for item
                        in scan_results
                    ]
                )
            )

            stable_indices = [
                strongest_index
            ]

        groups = []

        current = [
            stable_indices[0]
        ]

        for index in (
            stable_indices[1:]
        ):
            if (
                index
                == current[-1] + 1
            ):
                current.append(
                    index
                )
            else:
                groups.append(
                    current
                )

                current = [
                    index
                ]

        groups.append(
            current
        )

        candidates = []

        for candidate_index, group in enumerate(
            groups,
            start=1,
        ):
            items = [
                scan_results[index]
                for index in group
            ]

            candidate_scores = np.asarray(
                [
                    max(
                        float(
                            item[
                                "candidate_score"
                            ]
                        ),
                        0.0,
                    )
                    for item in items
                ],
                dtype=np.float64,
            )

            centers = np.asarray(
                [
                    float(
                        item[
                            "center_sec"
                        ]
                    )
                    for item in items
                ],
                dtype=np.float64,
            )

            if (
                candidate_scores.sum()
                > 0
            ):
                timecode_sec = float(
                    np.average(
                        centers,
                        weights=(
                            candidate_scores
                        ),
                    )
                )
            else:
                timecode_sec = float(
                    centers.mean()
                )

            max_score = float(
                max(
                    item[
                        "candidate_score"
                    ]
                    for item in items
                )
            )

            mean_score = float(
                np.mean(
                    [
                        item[
                            "candidate_score"
                        ]
                        for item in items
                    ]
                )
            )

            region_score = (
                0.55
                * max_score
                + 0.45
                * mean_score
            )

            peak_probability = float(
                max(
                    item[
                        "probability"
                    ]
                    for item in items
                )
            )

            mean_probability = float(
                np.mean(
                    [
                        item[
                            "probability"
                        ]
                        for item in items
                    ]
                )
            )

            persistence_mean = float(
                np.mean(
                    [
                        item[
                            "persistence"
                        ]
                        for item in items
                    ]
                )
            )

            candidates.append(
                {
                    "candidate_index": (
                        candidate_index
                    ),

                    "start_sec": float(
                        items[0][
                            "start_sec"
                        ]
                    ),

                    "end_sec": float(
                        items[-1][
                            "end_sec"
                        ]
                    ),

                    "timecode_sec": float(
                        round(
                            timecode_sec,
                            3,
                        )
                    ),

                    "score": float(
                        region_score
                    ),

                    "max_candidate_score": (
                        max_score
                    ),

                    "mean_candidate_score": (
                        mean_score
                    ),

                    "peak_probability": (
                        peak_probability
                    ),

                    "mean_probability": (
                        mean_probability
                    ),

                    "persistence": (
                        persistence_mean
                    ),

                    "windows_in_region": int(
                        len(
                            items
                        )
                    ),
                }
            )

        candidates.sort(
            key=lambda item: (
                item[
                    "score"
                ]
            ),
            reverse=True,
        )

        for rank, candidate in enumerate(
            candidates,
            start=1,
        ):
            candidate[
                "rank"
            ] = int(
                rank
            )

        return (
            candidates[
                :self.top_candidates
            ]
        )

    # =====================================================
    # FULL VIDEO
    # =====================================================

    def localize(
        self,
        video_path: str | Path,
        show_progress: bool = False,
    ) -> dict[str, Any]:
        video_path = Path(
            video_path
        ).resolve()

        duration_sec = (
            get_video_duration_sec(
                video_path
            )
        )

        scan_windows = (
            build_scan_windows(
                duration_sec=(
                    duration_sec
                ),
                window_sec=(
                    self.scan_window_sec
                ),
                stride_sec=(
                    self.scan_stride_sec
                ),
            )
        )

        iterator = enumerate(
            scan_windows,
            start=1,
        )

        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(
                iterator,
                total=len(
                    scan_windows
                ),
                desc=(
                    f"scan "
                    f"{video_path.stem}"
                ),
            )

        scan_results = []

        for window_index, (
            start_sec,
            end_sec,
        ) in iterator:
            result = (
                self.analyze_window(
                    video_path=(
                        video_path
                    ),
                    start_sec=(
                        start_sec
                    ),
                    end_sec=(
                        end_sec
                    ),
                    window_index=(
                        window_index
                    ),
                )
            )

            scan_results.append(
                result
            )

        raw_probabilities = [
            item[
                "probability"
            ]
            for item in scan_results
        ]

        # Legacy smoothing
        smoothed = (
            smooth_probabilities(
                raw_probabilities,
                radius=(
                    self.smoothing_radius
                ),
            )
        )

        # New normalized temporal scoring
        temporal_scores = (
            build_baseline_normalized_scores(
                raw_probabilities,
                support_radius=(
                    self.support_radius
                ),
            )
        )

        for index, item in enumerate(
            scan_results
        ):
            item[
                "smoothed_probability"
            ] = float(
                smoothed[index]
            )

            item[
                "above_threshold"
            ] = bool(
                smoothed[index]
                >= self.threshold
            )

            item[
                "centered_probability"
            ] = float(
                temporal_scores[
                    "centered_probability"
                ][index]
            )

            item[
                "normalized_score"
            ] = float(
                temporal_scores[
                    "normalized_score"
                ][index]
            )

            item[
                "support_score"
            ] = float(
                temporal_scores[
                    "support_score"
                ][index]
            )

            item[
                "persistence"
            ] = float(
                temporal_scores[
                    "persistence"
                ][index]
            )

            item[
                "candidate_score"
            ] = float(
                temporal_scores[
                    "candidate_score"
                ][index]
            )

        # -------------------------------------------------
        # Legacy peak
        # -------------------------------------------------

        legacy_peak = max(
            scan_results,
            key=lambda item: (
                item[
                    "smoothed_probability"
                ]
            ),
        )

        legacy_events = (
            self._build_events(
                scan_results
            )
        )

        # -------------------------------------------------
        # Improved candidate regions
        # -------------------------------------------------

        top_candidates = (
            self._build_candidate_regions(
                scan_results
            )
        )

        best_candidate = (
            top_candidates[0]
        )

        improved_peak = {
            "timecode_sec": float(
                best_candidate[
                    "timecode_sec"
                ]
            ),

            "score": float(
                best_candidate[
                    "score"
                ]
            ),

            "probability": float(
                best_candidate[
                    "peak_probability"
                ]
            ),

            "start_sec": float(
                best_candidate[
                    "start_sec"
                ]
            ),

            "end_sec": float(
                best_candidate[
                    "end_sec"
                ]
            ),

            "windows_in_region": int(
                best_candidate[
                    "windows_in_region"
                ]
            ),
        }

        return {
            "video_path": str(
                video_path
            ),

            "video_id": (
                video_path.stem
            ),

            "duration_sec": float(
                duration_sec
            ),

            "scan_window_sec": (
                self.scan_window_sec
            ),

            "scan_stride_sec": (
                self.scan_stride_sec
            ),

            "threshold": (
                self.threshold
            ),

            "windows_scanned": int(
                len(
                    scan_results
                )
            ),

            "baseline_probability": float(
                temporal_scores[
                    "baseline_probability"
                ]
            ),

            "robust_scale": float(
                temporal_scores[
                    "robust_scale"
                ]
            ),

            # Legacy output kept for compatibility
            "detected": bool(
                legacy_peak[
                    "smoothed_probability"
                ]
                >= self.threshold
            ),

            "peak": {
                "timecode_sec": float(
                    legacy_peak[
                        "center_sec"
                    ]
                ),

                "probability": float(
                    legacy_peak[
                        "smoothed_probability"
                    ]
                ),

                "raw_probability": float(
                    legacy_peak[
                        "probability"
                    ]
                ),

                "window_start_sec": float(
                    legacy_peak[
                        "start_sec"
                    ]
                ),

                "window_end_sec": float(
                    legacy_peak[
                        "end_sec"
                    ]
                ),
            },

            # Improved output
            "improved_detected": bool(
                best_candidate[
                    "peak_probability"
                ]
                >= self.threshold
            ),

            "improved_peak": (
                improved_peak
            ),

            "top_candidates": (
                top_candidates
            ),

            "events": (
                legacy_events
            ),

            "timeline": (
                scan_results
            ),
        }
