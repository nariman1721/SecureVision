from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "region_score",
    "max_candidate_score",
    "mean_candidate_score",
    "peak_prominence",
    "mean_prominence",
    "excess_area",
    "normalized_area",
    "consistency",
    "persistence_mean",
    "probability_std",
    "edge_drop",
    "windows_in_region",
    "duration_sec",
]


def build_baseline_scores(
    probabilities: np.ndarray,
    support_radius: int = 1,
) -> dict[str, np.ndarray | float]:
    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if probabilities.ndim != 1:
        raise ValueError(
            "probabilities must be 1D"
        )

    if len(probabilities) == 0:
        raise ValueError(
            "probabilities cannot be empty"
        )

    baseline = float(
        np.median(
            probabilities
        )
    )

    centered = (
        probabilities
        - baseline
    )

    mad = float(
        np.median(
            np.abs(
                centered
            )
        )
    )

    robust_scale = max(
        1.4826 * mad,
        0.02,
    )

    normalized = (
        centered
        / robust_scale
    )

    support = np.zeros_like(
        normalized
    )

    persistence = np.zeros_like(
        normalized
    )

    candidate_score = np.zeros_like(
        normalized
    )

    for index in range(
        len(normalized)
    ):
        left = max(
            0,
            index - support_radius,
        )

        right = min(
            len(normalized),
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

        support_value = float(
            neighborhood.mean()
        )

        persistence_value = float(
            np.mean(
                centered_neighborhood
                > 0
            )
        )

        score = (
            0.35
            * float(
                normalized[
                    index
                ]
            )
            + 0.65
            * support_value
        )

        if persistence_value < 0.5:
            score *= 0.60

        support[
            index
        ] = support_value

        persistence[
            index
        ] = persistence_value

        candidate_score[
            index
        ] = score

    return {
        "baseline": baseline,
        "robust_scale": robust_scale,
        "centered": centered,
        "normalized": normalized,
        "support": support,
        "persistence": persistence,
        "candidate_score": candidate_score,
    }


def build_candidate_table(
    timeline: pd.DataFrame,
    *,
    scan_stride_sec: float = 2.0,
    support_radius: int = 1,
) -> pd.DataFrame:
    timeline = (
        timeline
        .sort_values(
            "window_index"
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    probabilities = (
        timeline[
            "raw_probability"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    scoring = (
        build_baseline_scores(
            probabilities,
            support_radius=(
                support_radius
            ),
        )
    )

    baseline = float(
        scoring[
            "baseline"
        ]
    )

    timeline[
        "centered_probability"
    ] = scoring[
        "centered"
    ]

    timeline[
        "normalized_score"
    ] = scoring[
        "normalized"
    ]

    timeline[
        "support_score"
    ] = scoring[
        "support"
    ]

    timeline[
        "persistence"
    ] = scoring[
        "persistence"
    ]

    timeline[
        "candidate_score"
    ] = scoring[
        "candidate_score"
    ]

    stable_indices = [
        index
        for index, row
        in timeline.iterrows()
        if (
            float(
                row[
                    "candidate_score"
                ]
            )
            > 0
            and float(
                row[
                    "persistence"
                ]
            )
            >= 0.50
        )
    ]

    if not stable_indices:
        stable_indices = [
            int(
                np.argmax(
                    timeline[
                        "candidate_score"
                    ]
                    .to_numpy()
                )
            )
        ]

    groups: list[list[int]] = []

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

    rows: list[dict[str, Any]] = []

    for candidate_index, group in enumerate(
        groups,
        start=1,
    ):
        items = (
            timeline.iloc[
                group
            ]
            .copy()
        )

        candidate_scores = (
            items[
                "candidate_score"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        positive_scores = np.maximum(
            candidate_scores,
            0.0,
        )

        centers = (
            items[
                "center_sec"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        if positive_scores.sum() > 0:
            timecode_sec = float(
                np.average(
                    centers,
                    weights=(
                        positive_scores
                    ),
                )
            )

        else:
            timecode_sec = float(
                centers.mean()
            )

        raw = (
            items[
                "raw_probability"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        centered = (
            items[
                "centered_probability"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        normalized = (
            items[
                "normalized_score"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        persistence = (
            items[
                "persistence"
            ]
            .to_numpy(
                dtype=np.float64
            )
        )

        max_candidate_score = float(
            candidate_scores.max()
        )

        mean_candidate_score = float(
            candidate_scores.mean()
        )

        region_score = float(
            0.55
            * max_candidate_score
            + 0.45
            * mean_candidate_score
        )

        peak_probability = float(
            raw.max()
        )

        mean_probability = float(
            raw.mean()
        )

        peak_prominence = float(
            peak_probability
            - baseline
        )

        mean_prominence = float(
            mean_probability
            - baseline
        )

        excess_area = float(
            np.maximum(
                centered,
                0.0,
            ).sum()
            * scan_stride_sec
        )

        normalized_area = float(
            np.maximum(
                normalized,
                0.0,
            ).sum()
        )

        consistency = float(
            np.mean(
                raw
                > baseline
            )
        )

        persistence_mean = float(
            persistence.mean()
        )

        probability_std = float(
            raw.std()
        )

        if len(raw) >= 3:
            edge_mean = float(
                (
                    raw[0]
                    + raw[-1]
                )
                / 2.0
            )

            edge_drop = float(
                peak_probability
                - edge_mean
            )

        else:
            edge_drop = float(
                peak_prominence
            )

        start_sec = float(
            items[
                "start_sec"
            ].iloc[0]
        )

        end_sec = float(
            items[
                "end_sec"
            ].iloc[-1]
        )

        duration_sec = float(
            end_sec
            - start_sec
        )

        rows.append(
            {
                "candidate_index": int(
                    candidate_index
                ),
                "start_sec": (
                    start_sec
                ),
                "end_sec": (
                    end_sec
                ),
                "timecode_sec": float(
                    round(
                        timecode_sec,
                        3,
                    )
                ),
                "baseline_probability": (
                    baseline
                ),
                "region_score": (
                    region_score
                ),
                "max_candidate_score": (
                    max_candidate_score
                ),
                "mean_candidate_score": (
                    mean_candidate_score
                ),
                "peak_probability": (
                    peak_probability
                ),
                "mean_probability": (
                    mean_probability
                ),
                "peak_prominence": (
                    peak_prominence
                ),
                "mean_prominence": (
                    mean_prominence
                ),
                "excess_area": (
                    excess_area
                ),
                "normalized_area": (
                    normalized_area
                ),
                "consistency": (
                    consistency
                ),
                "persistence_mean": (
                    persistence_mean
                ),
                "probability_std": (
                    probability_std
                ),
                "edge_drop": (
                    edge_drop
                ),
                "windows_in_region": int(
                    len(
                        items
                    )
                ),
                "duration_sec": (
                    duration_sec
                ),
            }
        )

    candidates = pd.DataFrame(
        rows
    )

    candidates = (
        candidates
        .sort_values(
            "region_score",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    candidates[
        "original_rank"
    ] = np.arange(
        1,
        len(
            candidates
        )
        + 1,
    )

    return candidates


def assign_candidate_labels(
    candidates: pd.DataFrame,
    event_times: list[float],
    positive_radius_sec: float = 5.0,
) -> pd.DataFrame:
    candidates = (
        candidates.copy()
    )

    def distance_to_nearest_event(
        timecode: float,
    ) -> float:
        return float(
            min(
                abs(
                    float(timecode)
                    - float(event)
                )
                for event
                in event_times
            )
        )

    candidates[
        "nearest_event_distance_sec"
    ] = candidates[
        "timecode_sec"
    ].apply(
        distance_to_nearest_event
    )

    candidates[
        "reranker_label"
    ] = (
        candidates[
            "nearest_event_distance_sec"
        ]
        <= positive_radius_sec
    ).astype(
        np.int64
    )

    return candidates