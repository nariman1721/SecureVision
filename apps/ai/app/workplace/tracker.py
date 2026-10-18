from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any


@dataclass
class _Track:
    track_id: int

    center_x: float
    center_y: float

    bbox: dict[str, float]

    confidence: float

    first_seen_sec: float
    last_seen_sec: float

    hits: int = 1
    missed: int = 0


class PersonTracker:
    """
    Lightweight anonymous tracker.

    No face recognition.
    No biometric identity.

    People are associated between sampled frames
    using normalized centroid distance.
    """

    def __init__(
        self,
        *,
        max_distance: float = 0.22,
        max_missed: int = 3,
    ):
        if max_distance <= 0:
            raise ValueError(
                "max_distance must be positive."
            )

        if max_missed < 0:
            raise ValueError(
                "max_missed cannot be negative."
            )

        self.max_distance = float(
            max_distance
        )

        self.max_missed = int(
            max_missed
        )

        self._tracks: dict[
            int,
            _Track,
        ] = {}

        self._next_track_id = 1

        self.total_tracks_created = 0

    @staticmethod
    def _normalized_center(
        detection: dict[str, Any],
        *,
        frame_width: int,
        frame_height: int,
    ) -> tuple[float, float]:
        center = detection[
            "center"
        ]

        x = float(
            center[
                "x"
            ]
        ) / float(
            frame_width
        )

        y = float(
            center[
                "y"
            ]
        ) / float(
            frame_height
        )

        return (
            x,
            y,
        )

    @staticmethod
    def _distance(
        track: _Track,
        center_x: float,
        center_y: float,
    ) -> float:
        return hypot(
            track.center_x
            - center_x,

            track.center_y
            - center_y,
        )

    def _create_track(
        self,
        *,
        detection: dict[str, Any],
        center_x: float,
        center_y: float,
        timestamp_sec: float,
    ) -> _Track:
        track = _Track(
            track_id=(
                self._next_track_id
            ),

            center_x=(
                center_x
            ),

            center_y=(
                center_y
            ),

            bbox=dict(
                detection[
                    "bbox"
                ]
            ),

            confidence=float(
                detection[
                    "confidence"
                ]
            ),

            first_seen_sec=float(
                timestamp_sec
            ),

            last_seen_sec=float(
                timestamp_sec
            ),
        )

        self._tracks[
            track.track_id
        ] = track

        self._next_track_id += 1

        self.total_tracks_created += 1

        return track

    def update(
        self,
        detections: list[
            dict[str, Any]
        ],
        *,
        timestamp_sec: float,
        frame_width: int,
        frame_height: int,
    ) -> list[dict[str, Any]]:
        normalized_detections = []

        for index, detection in enumerate(
            detections
        ):
            center_x, center_y = (
                self._normalized_center(
                    detection,
                    frame_width=(
                        frame_width
                    ),
                    frame_height=(
                        frame_height
                    ),
                )
            )

            normalized_detections.append(
                {
                    "index": (
                        index
                    ),

                    "detection": (
                        detection
                    ),

                    "center_x": (
                        center_x
                    ),

                    "center_y": (
                        center_y
                    ),
                }
            )

        possible_matches = []

        for track_id, track in (
            self._tracks.items()
        ):
            for item in (
                normalized_detections
            ):
                distance = (
                    self._distance(
                        track,
                        item[
                            "center_x"
                        ],
                        item[
                            "center_y"
                        ],
                    )
                )

                if (
                    distance
                    <= self.max_distance
                ):
                    possible_matches.append(
                        (
                            distance,
                            track_id,
                            item[
                                "index"
                            ],
                        )
                    )

        possible_matches.sort(
            key=lambda item: (
                item[0]
            )
        )

        matched_tracks = set()
        matched_detections = set()

        observations: dict[
            int,
            dict[str, Any],
        ] = {}

        detection_lookup = {
            item[
                "index"
            ]: item
            for item
            in normalized_detections
        }

        for (
            distance,
            track_id,
            detection_index,
        ) in possible_matches:
            if (
                track_id
                in matched_tracks
            ):
                continue

            if (
                detection_index
                in matched_detections
            ):
                continue

            track = (
                self._tracks[
                    track_id
                ]
            )

            item = detection_lookup[
                detection_index
            ]

            detection = item[
                "detection"
            ]

            track.center_x = float(
                item[
                    "center_x"
                ]
            )

            track.center_y = float(
                item[
                    "center_y"
                ]
            )

            track.bbox = dict(
                detection[
                    "bbox"
                ]
            )

            track.confidence = float(
                detection[
                    "confidence"
                ]
            )

            track.last_seen_sec = float(
                timestamp_sec
            )

            track.hits += 1
            track.missed = 0

            matched_tracks.add(
                track_id
            )

            matched_detections.add(
                detection_index
            )

            observations[
                detection_index
            ] = {
                **detection,

                "track_id": int(
                    track_id
                ),

                "tracking_distance": float(
                    distance
                ),
            }

        existing_track_ids = list(
            self._tracks.keys()
        )

        for track_id in (
            existing_track_ids
        ):
            if (
                track_id
                not in matched_tracks
            ):
                self._tracks[
                    track_id
                ].missed += 1

        for item in (
            normalized_detections
        ):
            detection_index = (
                item[
                    "index"
                ]
            )

            if (
                detection_index
                in matched_detections
            ):
                continue

            detection = (
                item[
                    "detection"
                ]
            )

            track = (
                self._create_track(
                    detection=(
                        detection
                    ),

                    center_x=(
                        item[
                            "center_x"
                        ]
                    ),

                    center_y=(
                        item[
                            "center_y"
                        ]
                    ),

                    timestamp_sec=(
                        timestamp_sec
                    ),
                )
            )

            observations[
                detection_index
            ] = {
                **detection,

                "track_id": int(
                    track.track_id
                ),

                "tracking_distance": None,
            }

        stale_track_ids = [
            track_id
            for track_id, track
            in self._tracks.items()
            if (
                track.missed
                > self.max_missed
            )
        ]

        for track_id in stale_track_ids:
            del self._tracks[
                track_id
            ]

        return [
            observations[
                index
            ]
            for index
            in sorted(
                observations.keys()
            )
        ]