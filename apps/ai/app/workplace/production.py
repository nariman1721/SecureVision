from __future__ import annotations
import time
from pathlib import Path
from typing import Any
import cv2
from app.workplace.detector import (
    PersonDetector,
)
from app.workplace.posture import (
    PostureEstimator,
)
from app.workplace.tracker import (
    PersonTracker,
)
from app.workplace.zone_monitor import (
    NormalizedWorkZone,
    ZoneMonitor,
)


class ProductionWorkplaceAnalytics:
    """
    Unified production workplace analytics.

    Produces:
    - present / away
    - sitting / standing / unknown posture
    - absence intervals
    - anonymous track information
    - unified timeline
    - final production JSON

    Presence does NOT depend on persistent track identity.
    This makes the pipeline resistant to simple ID switches.
    """

    def __init__(
        self,
        *,
        sample_interval_sec: float = 0.5,
        confidence_threshold: float = 0.45,
        min_absence_sec: float = 2.0,
        tracker_max_distance: float = 0.22,
        tracker_max_missed: int = 3,
        cashier_zone: NormalizedWorkZone | None = None,
    ):
        if sample_interval_sec <= 0:
            raise ValueError(
                "sample_interval_sec must be positive."
            )

        if min_absence_sec < 0:
            raise ValueError(
                "min_absence_sec cannot be negative."
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

        self.cashier_zone = (
            ZoneMonitor(
                cashier_zone
                or NormalizedWorkZone(
                    x1=0.18,
                    y1=0.42,
                    x2=0.72,
                    y2=0.72,
                )
            )
        )

        self.posture = (
            PostureEstimator()
        )

    @staticmethod
    def _select_cashier_candidate(
        candidates: list[
            dict[str, Any]
        ],
    ) -> dict[str, Any] | None:
        if not candidates:
            return None

        return max(
            candidates,
            key=lambda person: (
                person[
                    "confidence"
                ]
            ),
        )

    @staticmethod
    def _select_primary_track(
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
                    "cashier_zone_hits"
                ]
                > 0
            )
        ]

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                item[1][
                    "cashier_zone_hits"
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
        min_absence_sec: float,
    ) -> tuple[
        list[dict[str, float]],
        list[dict[str, float]],
    ]:
        all_intervals = []

        start_sec = None
        end_sec = None

        for item in timeline:
            if (
                item[
                    "state"
                ]
                == "away"
            ):
                if start_sec is None:
                    start_sec = float(
                        item[
                            "timecode_sec"
                        ]
                    )

                end_sec = float(
                    item[
                        "timecode_sec"
                    ]
                    + item[
                        "sample_duration_sec"
                    ]
                )

            elif start_sec is not None:
                all_intervals.append(
                    {
                        "start_sec": float(
                            round(
                                start_sec,
                                3,
                            )
                        ),
                        "end_sec": float(
                            round(
                                end_sec,
                                3,
                            )
                        ),
                        "duration_sec": float(
                            round(
                                end_sec
                                - start_sec,
                                3,
                            )
                        ),
                    }
                )

                start_sec = None
                end_sec = None

        if start_sec is not None:
            all_intervals.append(
                {
                    "start_sec": float(
                        round(
                            start_sec,
                            3,
                        )
                    ),
                    "end_sec": float(
                        round(
                            end_sec,
                            3,
                        )
                    ),
                    "duration_sec": float(
                        round(
                            end_sec
                            - start_sec,
                            3,
                        )
                    ),
                }
            )

        significant_intervals = [
            interval
            for interval
            in all_intervals
            if (
                interval[
                    "duration_sec"
                ]
                >= min_absence_sec
            )
        ]

        return (
            all_intervals,
            significant_intervals,
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
                f"Video not found: "
                f"{video_path}"
            )

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
                        f"\rworkplace production scan: "
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

                    in_cashier_zone = (
                        self.cashier_zone
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
                            "cashier_zone_hits": 0,
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

                    if in_cashier_zone:
                        stats[
                            "cashier_zone_hits"
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
                                in_cashier_zone
                            ),
                            "bbox": (
                                person[
                                    "bbox"
                                ]
                            ),
                            "center": (
                                person[
                                    "center"
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

            sample_duration_sec = max(
                0.0,
                next_time
                - timestamp_sec,
            )

            if not item[
                "frame_valid"
            ]:
                timeline.append(
                    {
                        "timecode_sec": (
                            timestamp_sec
                        ),

                        "sample_duration_sec": float(
                            round(
                                sample_duration_sec,
                                3,
                            )
                        ),

                        "state": (
                            "unknown"
                        ),

                        "posture": (
                            "unknown"
                        ),

                        "cashier_present": (
                            False
                        ),

                        "active_cashier_track_id": (
                            None
                        ),

                        "cashier_candidate_track_ids": (
                            []
                        ),

                        "person_count": (
                            0
                        ),

                        "visible_track_ids": (
                            []
                        ),

                        "detector_confidence": (
                            None
                        ),

                        "posture_confidence": (
                            None
                        ),

                        "posture_features": (
                            None
                        ),
                    }
                )

                continue

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

            cashier = (
                self._select_cashier_candidate(
                    cashier_candidates
                )
            )

            if cashier is None:
                timeline.append(
                    {
                        "timecode_sec": (
                            timestamp_sec
                        ),

                        "sample_duration_sec": float(
                            round(
                                sample_duration_sec,
                                3,
                            )
                        ),

                        "state": (
                            "away"
                        ),

                        "posture": (
                            "away"
                        ),

                        "cashier_present": (
                            False
                        ),

                        "active_cashier_track_id": (
                            None
                        ),

                        "cashier_candidate_track_ids": (
                            []
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

                        "detector_confidence": (
                            None
                        ),

                        "posture_confidence": (
                            None
                        ),

                        "posture_features": (
                            None
                        ),
                    }
                )

                continue

            posture_result = (
                self.posture.classify(
                    cashier,
                    frame_width=(
                        frame_width
                    ),
                    frame_height=(
                        frame_height
                    ),
                )
            )

            timeline.append(
                {
                    "timecode_sec": (
                        timestamp_sec
                    ),

                    "sample_duration_sec": float(
                        round(
                            sample_duration_sec,
                            3,
                        )
                    ),

                    "state": (
                        "present"
                    ),

                    "posture": (
                        posture_result[
                            "posture"
                        ]
                    ),

                    "cashier_present": (
                        True
                    ),

                    "active_cashier_track_id": int(
                        cashier[
                            "track_id"
                        ]
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

                    "detector_confidence": float(
                        round(
                            cashier[
                                "confidence"
                            ],
                            4,
                        )
                    ),

                    "posture_confidence": (
                        posture_result[
                            "heuristic_confidence"
                        ]
                    ),

                    "posture_features": (
                        posture_result[
                            "features"
                        ]
                    ),
                }
            )

        # =================================================

        # =================================================
        # VISUALIZATION DATA
        # Preserve anonymous detections for frontend overlay.
        # =================================================

        if len(timeline) != len(raw_timeline):
            raise RuntimeError(
                "Timeline and detection samples do not match."
            )

        for output_item, raw_item in zip(
            timeline,
            raw_timeline,
        ):
            output_item["frame_valid"] = bool(
                raw_item["frame_valid"]
            )

            output_item["people"] = []

            for person in raw_item["people"]:
                raw_bbox = person["bbox"]

                if isinstance(raw_bbox, dict):
                    bbox = [
                        round(float(raw_bbox[key]), 2)
                        for key in ("x1", "y1", "x2", "y2")
                    ]
                else:
                    bbox = [
                        round(float(value), 2)
                        for value in raw_bbox
                    ]

                if len(bbox) != 4:
                    continue

                x1, y1, x2, y2 = bbox

                output_item["people"].append(
                    {
                        "track_id": int(
                            person["track_id"]
                        ),
                        "confidence": round(
                            float(person["confidence"]),
                            4,
                        ),
                        "in_cashier_zone": bool(
                            person["in_cashier_zone"]
                        ),
                        "bbox": bbox,
                        "center": [
                            round((x1 + x2) / 2, 2),
                            round((y1 + y2) / 2, 2),
                        ],
                    }
                )

        # TIME AGGREGATION
        # =================================================

        present_sec = 0.0
        away_sec = 0.0
        unknown_sec = 0.0

        sitting_sec = 0.0
        standing_sec = 0.0
        unknown_posture_sec = 0.0

        for item in timeline:
            duration = float(
                item[
                    "sample_duration_sec"
                ]
            )

            state = (
                item[
                    "state"
                ]
            )

            posture = (
                item[
                    "posture"
                ]
            )

            if state == "present":
                present_sec += duration

                if posture == "sitting":
                    sitting_sec += duration

                elif posture == "standing":
                    standing_sec += duration

                else:
                    unknown_posture_sec += (
                        duration
                    )

            elif state == "away":
                away_sec += duration

            else:
                unknown_sec += duration

        known_presence_sec = (
            present_sec
            + away_sec
        )

        if known_presence_sec > 0:
            presence_percent = (
                present_sec
                / known_presence_sec
                * 100.0
            )
        else:
            presence_percent = 0.0

        known_posture_sec = (
            sitting_sec
            + standing_sec
        )

        if known_posture_sec > 0:
            sitting_percent = (
                sitting_sec
                / known_posture_sec
                * 100.0
            )

            standing_percent = (
                standing_sec
                / known_posture_sec
                * 100.0
            )

        else:
            sitting_percent = 0.0
            standing_percent = 0.0

        (
            all_absence_intervals,
            absence_intervals,
        ) = (
            self._build_absence_intervals(
                timeline,
                min_absence_sec=(
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
                in absence_intervals
            ),
            default=0.0,
        )

        primary_track_id = (
            self._select_primary_track(
                track_stats
            )
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
                    "cashier_zone_hits"
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

                    "primary_track_candidate": bool(
                        track_id
                        == primary_track_id
                    ),
                }
            )

        processing_time_sec = (
            time.perf_counter()
            - started
        )

        return {
            "status": (
                "done"
            ),

            "analysis_type": (
                "workplace_activity_analytics"
            ),

            "file_name": (
                video_path.name
            ),

            "video": {
                "duration_sec": float(
                    round(
                        duration_sec,
                        3,
                    )
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

            "visualization": {
                "bbox_format": "xyxy",
                "bbox_coordinate_system": "pixel",
                "cashier_zone_coordinate_system": "normalized_0_1",
                "sample_interval_sec": self.sample_interval_sec,
                "anonymous_tracking": True,
                "face_recognition": False,
                "intermediate_frames_are_estimates": True,
            },

            "configuration": {
                "sample_interval_sec": (
                    self.sample_interval_sec
                ),

                "confidence_threshold": (
                    self.detector
                    .confidence_threshold
                ),

                "min_absence_sec": (
                    self.min_absence_sec
                ),

                "cashier_zone": {
                    "x1": (
                        self.cashier_zone
                        .zone
                        .x1
                    ),

                    "y1": (
                        self.cashier_zone
                        .zone
                        .y1
                    ),

                    "x2": (
                        self.cashier_zone
                        .zone
                        .x2
                    ),

                    "y2": (
                        self.cashier_zone
                        .zone
                        .y2
                    ),
                },

                "device": str(
                    self.detector.device
                ),
            },

            "methodology": {
                "presence_method": (
                    "cashier_zone_occupancy"
                ),

                "posture_method": (
                    "camera_calibrated_bbox_geometry"
                ),

                "tracking_method": (
                    "anonymous_centroid_tracking"
                ),

                "tracking_used_for_presence": (
                    False
                ),

                "id_switch_resistant_presence": (
                    True
                ),

                "face_recognition": (
                    False
                ),

                "biometric_identification": (
                    False
                ),
            },

            "summary": {
                "duration_sec": float(
                    round(
                        duration_sec,
                        3,
                    )
                ),

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

                "sitting_sec": float(
                    round(
                        sitting_sec,
                        3,
                    )
                ),

                "standing_sec": float(
                    round(
                        standing_sec,
                        3,
                    )
                ),

                "unknown_posture_sec": float(
                    round(
                        unknown_posture_sec,
                        3,
                    )
                ),

                "sitting_percent_of_known_posture": float(
                    round(
                        sitting_percent,
                        2,
                    )
                ),

                "standing_percent_of_known_posture": float(
                    round(
                        standing_percent,
                        2,
                    )
                ),

                "absence_count": int(
                    len(
                        absence_intervals
                    )
                ),

                "longest_absence_sec": float(
                    round(
                        longest_absence_sec,
                        3,
                    )
                ),

                "samples": int(
                    len(
                        timeline
                    )
                ),

                "tracks_created": int(
                    tracker
                    .total_tracks_created
                ),
            },

            "absence_intervals": (
                absence_intervals
            ),

            "all_detected_absence_intervals": (
                all_absence_intervals
            ),

            "tracking": {
                "primary_track_candidate": (
                    primary_track_id
                ),

                "tracks": (
                    track_summaries
                ),
            },

            "timeline": (
                timeline
            ),

            "processing_time_sec": float(
                round(
                    processing_time_sec,
                    3,
                )
            ),
        }