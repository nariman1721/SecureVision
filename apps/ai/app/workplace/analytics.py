from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2

from app.workplace.detector import (
    PersonDetector,
)

from app.workplace.tracker import (
    PersonTracker,
)

from app.workplace.zone_monitor import (
    NormalizedWorkZone,
    ZoneMonitor,
)


class WorkplaceAnalytics:
    def __init__(
        self,
        *,
        sample_interval_sec: float = 1.0,
        confidence_threshold: float = 0.45,
        work_zone: NormalizedWorkZone | None = None,
        tracker_max_distance: float = 0.22,
        tracker_max_missed: int = 3,
        min_absence_sec: float = 2.0,
    ):
        if sample_interval_sec <= 0:
            raise ValueError(
                "sample_interval_sec must be positive."
            )

        self.sample_interval_sec = float(
            sample_interval_sec
        )

        self.min_absence_sec = float(
            min_absence_sec
        )

        self.tracker_max_distance = float(
            tracker_max_distance
        )

        self.tracker_max_missed = int(
            tracker_max_missed
        )

        self.detector = (
            PersonDetector(
                confidence_threshold=(
                    confidence_threshold
                )
            )
        )

        self.zone_monitor = (
            ZoneMonitor(
                work_zone
                or NormalizedWorkZone(
                    x1=0.18,
                    y1=0.42,
                    x2=0.72,
                    y2=0.72,
                )
            )
        )

    @staticmethod
    def _select_worker_track(
        track_stats: dict[
            int,
            dict[str, Any],
        ],
    ) -> int | None:
        candidates = [
            (
                track_id,
                stats,
            )
            for track_id, stats
            in track_stats.items()
            if (
                stats[
                    "zone_hits"
                ]
                > 0
            )
        ]

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item[1][
                    "zone_hits"
                ],
                item[1][
                    "hits"
                ],
                item[1][
                    "mean_confidence"
                ],
            ),
            reverse=True,
        )

        return int(
            candidates[0][0]
        )

    @staticmethod
    def _build_absence_intervals(
        timeline: list[
            dict[str, Any]
        ],
        *,
        minimum_duration_sec: float,
    ) -> tuple[
        list[dict[str, float]],
        list[dict[str, float]],
    ]:
        all_intervals = []

        current_start = None
        current_end = None

        for item in timeline:
            if (
                item[
                    "state"
                ]
                == "away"
            ):
                if current_start is None:
                    current_start = float(
                        item[
                            "timecode_sec"
                        ]
                    )

                current_end = float(
                    item[
                        "timecode_sec"
                    ]
                    + item[
                        "sample_duration_sec"
                    ]
                )

            elif current_start is not None:
                all_intervals.append(
                    {
                        "start_sec": float(
                            round(
                                current_start,
                                3,
                            )
                        ),

                        "end_sec": float(
                            round(
                                current_end,
                                3,
                            )
                        ),

                        "duration_sec": float(
                            round(
                                current_end
                                - current_start,
                                3,
                            )
                        ),
                    }
                )

                current_start = None
                current_end = None

        if current_start is not None:
            all_intervals.append(
                {
                    "start_sec": float(
                        round(
                            current_start,
                            3,
                        )
                    ),

                    "end_sec": float(
                        round(
                            current_end,
                            3,
                        )
                    ),

                    "duration_sec": float(
                        round(
                            current_end
                            - current_start,
                            3,
                        )
                    ),
                }
            )

        significant = [
            interval
            for interval
            in all_intervals
            if (
                interval[
                    "duration_sec"
                ]
                >= minimum_duration_sec
            )
        ]

        return (
            all_intervals,
            significant,
        )

    def analyze(
        self,
        video_path: str | Path,
        *,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        started = (
            time.perf_counter()
        )

        video_path = Path(
            video_path
        ).resolve()

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        # Fresh tracker for every video.
        tracker = (
            PersonTracker(
                max_distance=(
                    self.tracker_max_distance
                ),
                max_missed=(
                    self.tracker_max_missed
                ),
            )
        )

        capture = cv2.VideoCapture(
            str(
                video_path
            )
        )

        if not capture.isOpened():
            raise RuntimeError(
                f"Unable to open video: "
                f"{video_path}"
            )

        fps = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
        )

        frame_count = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        frame_width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        frame_height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        if fps <= 0:
            capture.release()

            raise RuntimeError(
                "Invalid video FPS."
            )

        duration_sec = (
            frame_count
            / fps
        )

        sample_times = []

        current_time = 0.0

        while (
            current_time
            < duration_sec
        ):
            sample_times.append(
                current_time
            )

            current_time += (
                self.sample_interval_sec
            )

        raw_timeline = []

        track_stats: dict[
            int,
            dict[str, Any],
        ] = {}

        try:
            total = len(
                sample_times
            )

            for index, timestamp_sec in enumerate(
                sample_times,
                start=1,
            ):
                if show_progress:
                    print(
                        f"\rcashier-zone scan: "
                        f"{index}/{total}",
                        end="",
                        flush=True,
                    )

                capture.set(
                    cv2.CAP_PROP_POS_MSEC,
                    timestamp_sec
                    * 1000.0,
                )

                success, frame = (
                    capture.read()
                )

                if (
                    not success
                    or frame is None
                ):
                    raw_timeline.append(
                        {
                            "timecode_sec": float(
                                round(
                                    timestamp_sec,
                                    3,
                                )
                            ),

                            "frame_valid": False,

                            "people": [],
                        }
                    )

                    continue

                detections = (
                    self.detector.detect(
                        frame
                    )
                )

                tracked_people = (
                    tracker.update(
                        detections,

                        timestamp_sec=(
                            timestamp_sec
                        ),

                        frame_width=(
                            frame_width
                        ),

                        frame_height=(
                            frame_height
                        ),
                    )
                )

                people = []

                for person in (
                    tracked_people
                ):
                    track_id = int(
                        person[
                            "track_id"
                        ]
                    )

                    in_zone = (
                        self.zone_monitor
                        .detection_inside_zone(
                            person,

                            frame_width=(
                                frame_width
                            ),

                            frame_height=(
                                frame_height
                            ),
                        )
                    )

                    confidence = float(
                        person[
                            "confidence"
                        ]
                    )

                    if (
                        track_id
                        not in track_stats
                    ):
                        track_stats[
                            track_id
                        ] = {
                            "hits": 0,
                            "zone_hits": 0,
                            "confidence_sum": 0.0,
                            "mean_confidence": 0.0,
                            "first_seen_sec": float(
                                timestamp_sec
                            ),
                            "last_seen_sec": float(
                                timestamp_sec
                            ),
                        }

                    stats = (
                        track_stats[
                            track_id
                        ]
                    )

                    stats[
                        "hits"
                    ] += 1

                    if in_zone:
                        stats[
                            "zone_hits"
                        ] += 1

                    stats[
                        "confidence_sum"
                    ] += confidence

                    stats[
                        "mean_confidence"
                    ] = (
                        stats[
                            "confidence_sum"
                        ]
                        / stats[
                            "hits"
                        ]
                    )

                    stats[
                        "last_seen_sec"
                    ] = float(
                        timestamp_sec
                    )

                    people.append(
                        {
                            "track_id": (
                                track_id
                            ),

                            "confidence": (
                                confidence
                            ),

                            "in_cashier_zone": bool(
                                in_zone
                            ),

                            "center": (
                                person[
                                    "center"
                                ]
                            ),

                            "bbox": (
                                person[
                                    "bbox"
                                ]
                            ),
                        }
                    )

                raw_timeline.append(
                    {
                        "timecode_sec": float(
                            round(
                                timestamp_sec,
                                3,
                            )
                        ),

                        "frame_valid": True,

                        "people": (
                            people
                        ),
                    }
                )

        finally:
            capture.release()

            if show_progress:
                print()

        selected_worker_track_id = (
            self._select_worker_track(
                track_stats
            )
        )

        timeline = []

        for index, item in enumerate(
            raw_timeline
        ):
            timestamp_sec = float(
                item[
                    "timecode_sec"
                ]
            )

            if (
                index + 1
                < len(
                    raw_timeline
                )
            ):
                next_time = float(
                    raw_timeline[
                        index + 1
                    ][
                        "timecode_sec"
                    ]
                )

            else:
                next_time = float(
                    duration_sec
                )

            sample_duration = max(
                0.0,
                next_time
                - timestamp_sec,
            )

            if not item[
                "frame_valid"
            ]:
                state = (
                    "unknown"
                )

                cashier_present = False

                cashier_candidates = []

            else:
                cashier_candidates = [
                    person
                    for person
                    in item[
                        "people"
                    ]
                    if (
                        person[
                            "in_cashier_zone"
                        ]
                    )
                ]

                cashier_present = bool(
                    cashier_candidates
                )

                state = (
                    "present"
                    if cashier_present
                    else "away"
                )

            if cashier_candidates:
                best_candidate = max(
                    cashier_candidates,

                    key=lambda person: (
                        person[
                            "confidence"
                        ]
                    ),
                )

                active_cashier_track_id = int(
                    best_candidate[
                        "track_id"
                    ]
                )

                cashier_confidence = float(
                    best_candidate[
                        "confidence"
                    ]
                )

            else:
                active_cashier_track_id = None
                cashier_confidence = None

            timeline.append(
                {
                    "timecode_sec": (
                        timestamp_sec
                    ),

                    "sample_duration_sec": float(
                        round(
                            sample_duration,
                            3,
                        )
                    ),

                    "state": (
                        state
                    ),

                    "cashier_present": bool(
                        cashier_present
                    ),

                    "active_cashier_track_id": (
                        active_cashier_track_id
                    ),

                    "cashier_candidate_track_ids": [
                        int(
                            person[
                                "track_id"
                            ]
                        )
                        for person
                        in cashier_candidates
                    ],

                    "cashier_confidence": (
                        cashier_confidence
                    ),

                    "person_count": int(
                        len(
                            item[
                                "people"
                            ]
                        )
                    ),

                    "visible_track_ids": [
                        int(
                            person[
                                "track_id"
                            ]
                        )
                        for person
                        in item[
                            "people"
                        ]
                    ],
                }
            )

        present_sec = sum(
            item[
                "sample_duration_sec"
            ]
            for item
            in timeline
            if (
                item[
                    "state"
                ]
                == "present"
            )
        )

        away_sec = sum(
            item[
                "sample_duration_sec"
            ]
            for item
            in timeline
            if (
                item[
                    "state"
                ]
                == "away"
            )
        )

        unknown_sec = sum(
            item[
                "sample_duration_sec"
            ]
            for item
            in timeline
            if (
                item[
                    "state"
                ]
                == "unknown"
            )
        )

        known_sec = (
            present_sec
            + away_sec
        )

        if known_sec > 0:
            presence_percent = (
                present_sec
                / known_sec
                * 100.0
            )

        else:
            presence_percent = 0.0

        (
            all_absences,
            significant_absences,
        ) = (
            self._build_absence_intervals(
                timeline,

                minimum_duration_sec=(
                    self.min_absence_sec
                ),
            )
        )

        longest_absence_sec = max(
            (
                interval[
                    "duration_sec"
                ]
                for interval
                in significant_absences
            ),
            default=0.0,
        )

        track_summaries = []

        for track_id, stats in sorted(
            track_stats.items()
        ):
            hits = int(
                stats[
                    "hits"
                ]
            )

            zone_hits = int(
                stats[
                    "zone_hits"
                ]
            )

            track_summaries.append(
                {
                    "track_id": int(
                        track_id
                    ),

                    "hits": (
                        hits
                    ),

                    "cashier_zone_hits": (
                        zone_hits
                    ),

                    "cashier_zone_ratio": float(
                        round(
                            (
                                zone_hits
                                / hits
                                if hits
                                else 0.0
                            ),
                            4,
                        )
                    ),

                    "mean_confidence": float(
                        round(
                            stats[
                                "mean_confidence"
                            ],
                            4,
                        )
                    ),

                    "first_seen_sec": float(
                        stats[
                            "first_seen_sec"
                        ]
                    ),

                    "last_seen_sec": float(
                        stats[
                            "last_seen_sec"
                        ]
                    ),

                    "selected_as_primary_worker_track": bool(
                        track_id
                        == selected_worker_track_id
                    ),
                }
            )

        elapsed = (
            time.perf_counter()
            - started
        )

        return {
            "status": (
                "done"
            ),

            "file_name": (
                video_path.name
            ),

            "video": {
                "duration_sec": float(
                    duration_sec
                ),

                "fps": float(
                    fps
                ),

                "frame_count": int(
                    frame_count
                ),

                "width": int(
                    frame_width
                ),

                "height": int(
                    frame_height
                ),
            },

            "configuration": {
                "sample_interval_sec": (
                    self.sample_interval_sec
                ),

                "confidence_threshold": (
                    self.detector
                    .confidence_threshold
                ),

                "tracker_max_distance": (
                    self.tracker_max_distance
                ),

                "tracker_max_missed": (
                    self.tracker_max_missed
                ),

                "min_absence_sec": (
                    self.min_absence_sec
                ),

                "cashier_zone": {
                    "x1": (
                        self.zone_monitor
                        .zone
                        .x1
                    ),

                    "y1": (
                        self.zone_monitor
                        .zone
                        .y1
                    ),

                    "x2": (
                        self.zone_monitor
                        .zone
                        .x2
                    ),

                    "y2": (
                        self.zone_monitor
                        .zone
                        .y2
                    ),
                },

                "device": str(
                    self.detector.device
                ),
            },

            "worker_selection": {
                "primary_track_id": (
                    selected_worker_track_id
                ),

                "presence_method": (
                    "cashier-zone occupancy"
                ),

                "tracking_used_for_presence": (
                    False
                ),

                "tracking_used_for_analytics": (
                    True
                ),

                "id_switch_resistant": (
                    True
                ),

                "anonymous_tracking": (
                    True
                ),

                "face_recognition": (
                    False
                ),
            },

            "summary": {
                "present_sec": float(
                    round(
                        present_sec,
                        3,
                    )
                ),

                "away_sec": float(
                    round(
                        away_sec,
                        3,
                    )
                ),

                "unknown_sec": float(
                    round(
                        unknown_sec,
                        3,
                    )
                ),

                "presence_percent": float(
                    round(
                        presence_percent,
                        2,
                    )
                ),

                "absence_count": int(
                    len(
                        significant_absences
                    )
                ),

                "longest_absence_sec": float(
                    round(
                        longest_absence_sec,
                        3,
                    )
                ),

                "tracks_created": int(
                    tracker
                    .total_tracks_created
                ),

                "primary_worker_track_id": (
                    selected_worker_track_id
                ),
            },

            "absence_intervals": (
                significant_absences
            ),

            "all_detected_absence_intervals": (
                all_absences
            ),

            "tracks": (
                track_summaries
            ),

            "timeline": (
                timeline
            ),

            "processing_time_sec": float(
                round(
                    elapsed,
                    3,
                )
            ),
        }