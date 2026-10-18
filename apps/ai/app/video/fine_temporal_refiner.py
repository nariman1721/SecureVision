from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.video.temporal_localizer import TemporalLocalizer


DEFAULT_FINE_WINDOW_SEC = 6.0
DEFAULT_FINE_STRIDE_SEC = 0.5
DEFAULT_SEARCH_RADIUS_SEC = 5.0
DEFAULT_SUPPORT_RADIUS_WINDOWS = 2


# =========================================================
# ROBUST STATISTICS
# =========================================================

def robust_baseline_and_scale(
    probabilities: list[float] | np.ndarray,
) -> tuple[float, float]:
    values = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "probabilities must be 1D"
        )

    if len(values) == 0:
        raise ValueError(
            "probabilities cannot be empty"
        )

    baseline = float(
        np.median(
            values
        )
    )

    mad = float(
        np.median(
            np.abs(
                values
                - baseline
            )
        )
    )

    robust_scale = max(
        1.4826 * mad,
        0.02,
    )

    return (
        baseline,
        robust_scale,
    )


# =========================================================
# FINE WINDOWS
# =========================================================

def build_fine_scan_windows(
    *,
    duration_sec: float,
    candidate_time_sec: float,
    search_radius_sec: float = DEFAULT_SEARCH_RADIUS_SEC,
    window_sec: float = DEFAULT_FINE_WINDOW_SEC,
    stride_sec: float = DEFAULT_FINE_STRIDE_SEC,
) -> list[tuple[float, float, float]]:
    duration_sec = float(
        duration_sec
    )

    candidate_time_sec = float(
        candidate_time_sec
    )

    search_radius_sec = float(
        search_radius_sec
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

    if search_radius_sec <= 0:
        raise ValueError(
            "search_radius_sec must be > 0"
        )

    # Very short video.
    if duration_sec <= window_sec:
        center = (
            duration_sec
            / 2.0
        )

        return [
            (
                0.0,
                round(
                    duration_sec,
                    3,
                ),
                round(
                    center,
                    3,
                ),
            )
        ]

    half_window = (
        window_sec
        / 2.0
    )

    minimum_center = (
        half_window
    )

    maximum_center = (
        duration_sec
        - half_window
    )

    left_center = max(
        minimum_center,
        candidate_time_sec
        - search_radius_sec,
    )

    right_center = min(
        maximum_center,
        candidate_time_sec
        + search_radius_sec,
    )

    if right_center < left_center:
        clipped_center = min(
            max(
                candidate_time_sec,
                minimum_center,
            ),
            maximum_center,
        )

        left_center = (
            clipped_center
        )

        right_center = (
            clipped_center
        )

    centers = []

    current = (
        left_center
    )

    while (
        current
        <= right_center + 1e-9
    ):
        centers.append(
            current
        )

        current += (
            stride_sec
        )

    # Always include the end of the
    # local search interval.
    if (
        not centers
        or abs(
            centers[-1]
            - right_center
        )
        > 1e-6
    ):
        centers.append(
            right_center
        )

    windows = []

    seen_starts = set()

    for center in centers:
        start_sec = (
            center
            - half_window
        )

        end_sec = (
            start_sec
            + window_sec
        )

        start_sec = max(
            0.0,
            start_sec,
        )

        end_sec = min(
            duration_sec,
            end_sec,
        )

        # Preserve six-second duration
        # near the right edge.
        if (
            end_sec
            - start_sec
            < window_sec
            and duration_sec
            >= window_sec
        ):
            start_sec = max(
                0.0,
                end_sec
                - window_sec,
            )

        start_sec = round(
            start_sec,
            3,
        )

        end_sec = round(
            end_sec,
            3,
        )

        center_sec = round(
            (
                start_sec
                + end_sec
            )
            / 2.0,
            3,
        )

        if start_sec in seen_starts:
            continue

        seen_starts.add(
            start_sec
        )

        windows.append(
            (
                start_sec,
                end_sec,
                center_sec,
            )
        )

    return windows


# =========================================================
# LOCAL SUPPORT
# =========================================================

def calculate_support_probabilities(
    probabilities: np.ndarray,
    radius: int,
) -> np.ndarray:
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    result = np.empty_like(
        probabilities
    )

    for index in range(
        len(
            probabilities
        )
    ):
        left = max(
            0,
            index - radius,
        )

        right = min(
            len(
                probabilities
            ),
            index + radius + 1,
        )

        result[
            index
        ] = float(
            probabilities[
                left:right
            ].mean()
        )

    return result


# =========================================================
# FINE TEMPORAL REFINER
# =========================================================

class FineTemporalRefiner:
    def __init__(
        self,
        localizer: TemporalLocalizer,
        fine_window_sec: float = DEFAULT_FINE_WINDOW_SEC,
        fine_stride_sec: float = DEFAULT_FINE_STRIDE_SEC,
        search_radius_sec: float = DEFAULT_SEARCH_RADIUS_SEC,
        support_radius_windows: int = DEFAULT_SUPPORT_RADIUS_WINDOWS,
    ):
        self.localizer = (
            localizer
        )

        self.fine_window_sec = float(
            fine_window_sec
        )

        self.fine_stride_sec = float(
            fine_stride_sec
        )

        self.search_radius_sec = float(
            search_radius_sec
        )

        self.support_radius_windows = int(
            support_radius_windows
        )

    # =====================================================
    # REFINE ONE CANDIDATE
    # =====================================================

    def refine_candidate(
        self,
        *,
        video_path: str | Path,
        duration_sec: float,
        candidate: dict[str, Any],
        global_baseline: float,
        global_scale: float,
        original_rank: int,
        show_progress: bool = False,
    ) -> dict[str, Any]:
        video_path = Path(
            video_path
        ).resolve()

        candidate_time = float(
            candidate[
                "timecode_sec"
            ]
        )

        fine_windows = (
            build_fine_scan_windows(
                duration_sec=(
                    duration_sec
                ),
                candidate_time_sec=(
                    candidate_time
                ),
                search_radius_sec=(
                    self.search_radius_sec
                ),
                window_sec=(
                    self.fine_window_sec
                ),
                stride_sec=(
                    self.fine_stride_sec
                ),
            )
        )

        iterator = enumerate(
            fine_windows,
            start=1,
        )

        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(
                iterator,
                total=len(
                    fine_windows
                ),
                desc=(
                    f"fine "
                    f"{video_path.stem} "
                    f"#{original_rank}"
                ),
            )

        fine_results = []

        for local_index, (
            start_sec,
            end_sec,
            center_sec,
        ) in iterator:
            result = (
                self.localizer.analyze_window(
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
                        100000
                        + original_rank
                        * 1000
                        + local_index
                    ),
                )
            )

            result[
                "center_sec"
            ] = float(
                center_sec
            )

            fine_results.append(
                result
            )

        probabilities = np.asarray(
            [
                item[
                    "probability"
                ]
                for item
                in fine_results
            ],
            dtype=np.float64,
        )

        centers = np.asarray(
            [
                item[
                    "center_sec"
                ]
                for item
                in fine_results
            ],
            dtype=np.float64,
        )

        (
            local_baseline,
            local_scale,
        ) = robust_baseline_and_scale(
            probabilities
        )

        support = (
            calculate_support_probabilities(
                probabilities,
                radius=(
                    self.support_radius_windows
                ),
            )
        )

        global_scale = max(
            float(
                global_scale
            ),
            0.02,
        )

        point_scores = (
            0.55
            * (
                probabilities
                - global_baseline
            )
            / global_scale
            +
            0.30
            * (
                support
                - global_baseline
            )
            / global_scale
            +
            0.15
            * (
                probabilities
                - local_baseline
            )
            / global_scale
        )

        best_index = int(
            np.argmax(
                point_scores
            )
        )

        fine_peak_sec = float(
            centers[
                best_index
            ]
        )

        fine_peak_probability = float(
            probabilities[
                best_index
            ]
        )

        peak_support_probability = float(
            support[
                best_index
            ]
        )

        peak_global_z = float(
            (
                fine_peak_probability
                - global_baseline
            )
            / global_scale
        )

        support_global_z = float(
            (
                peak_support_probability
                - global_baseline
            )
            / global_scale
        )

        local_prominence = float(
            fine_peak_probability
            - local_baseline
        )

        local_prominence_z = float(
            local_prominence
            / global_scale
        )

        # Compare the candidate peak against
        # the borders of its local neighborhood.
        edge_count = max(
            1,
            min(
                3,
                len(
                    probabilities
                )
                // 4,
            ),
        )

        edge_values = np.concatenate(
            [
                probabilities[
                    :edge_count
                ],
                probabilities[
                    -edge_count:
                ],
            ]
        )

        edge_probability = float(
            edge_values.mean()
        )

        edge_contrast = float(
            peak_support_probability
            - edge_probability
        )

        edge_contrast_z = float(
            edge_contrast
            / global_scale
        )

        positive_excess = np.maximum(
            probabilities
            - global_baseline,
            0.0,
        )

        area_density = float(
            positive_excess.mean()
        )

        area_density_z = float(
            area_density
            / global_scale
        )

        # Fixed heuristic.
        # No ground-truth timestamp is used here.
        fine_candidate_score = float(
            0.30
            * peak_global_z
            +
            0.25
            * support_global_z
            +
            0.20
            * local_prominence_z
            +
            0.15
            * edge_contrast_z
            +
            0.10
            * area_density_z
        )

        timeline = []

        for index, item in enumerate(
            fine_results
        ):
            timeline.append(
                {
                    "start_sec": float(
                        item[
                            "start_sec"
                        ]
                    ),
                    "end_sec": float(
                        item[
                            "end_sec"
                        ]
                    ),
                    "center_sec": float(
                        item[
                            "center_sec"
                        ]
                    ),
                    "probability": float(
                        probabilities[
                            index
                        ]
                    ),
                    "support_probability": float(
                        support[
                            index
                        ]
                    ),
                    "point_score": float(
                        point_scores[
                            index
                        ]
                    ),
                }
            )

        return {
            "original_rank": int(
                original_rank
            ),
            "coarse_timecode_sec": float(
                candidate_time
            ),
            "coarse_score": float(
                candidate.get(
                    "region_score",
                    candidate.get(
                        "score",
                        0.0,
                    ),
                )
            ),
            "fine_peak_sec": (
                fine_peak_sec
            ),
            "fine_peak_probability": (
                fine_peak_probability
            ),
            "fine_support_probability": (
                peak_support_probability
            ),
            "global_baseline": float(
                global_baseline
            ),
            "global_scale": float(
                global_scale
            ),
            "local_baseline": float(
                local_baseline
            ),
            "local_scale": float(
                local_scale
            ),
            "local_prominence": (
                local_prominence
            ),
            "edge_probability": (
                edge_probability
            ),
            "edge_contrast": (
                edge_contrast
            ),
            "area_density": (
                area_density
            ),
            "peak_global_z": (
                peak_global_z
            ),
            "support_global_z": (
                support_global_z
            ),
            "local_prominence_z": (
                local_prominence_z
            ),
            "edge_contrast_z": (
                edge_contrast_z
            ),
            "area_density_z": (
                area_density_z
            ),
            "fine_score": (
                fine_candidate_score
            ),
            "fine_windows_scanned": int(
                len(
                    fine_results
                )
            ),
            "timeline": (
                timeline
            ),
        }

    # =====================================================
    # REFINE TOP CANDIDATES
    # =====================================================

    def refine_candidates(
        self,
        *,
        video_path: str | Path,
        duration_sec: float,
        candidates: list[
            dict[str, Any]
        ],
        global_baseline: float,
        global_scale: float,
        top_k: int = 3,
        show_progress: bool = False,
    ) -> list[dict[str, Any]]:
        selected = (
            candidates[
                :top_k
            ]
        )

        refined = []

        for index, candidate in enumerate(
            selected,
            start=1,
        ):
            result = (
                self.refine_candidate(
                    video_path=(
                        video_path
                    ),
                    duration_sec=(
                        duration_sec
                    ),
                    candidate=(
                        candidate
                    ),
                    global_baseline=(
                        global_baseline
                    ),
                    global_scale=(
                        global_scale
                    ),
                    original_rank=(
                        index
                    ),
                    show_progress=(
                        show_progress
                    ),
                )
            )

            refined.append(
                result
            )

        refined.sort(
            key=lambda item: (
                item[
                    "fine_score"
                ]
            ),
            reverse=True,
        )

        for rank, item in enumerate(
            refined,
            start=1,
        ):
            item[
                "refined_rank"
            ] = int(
                rank
            )

        return refined