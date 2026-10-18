from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizedWorkZone:
    x1: float = 0.10
    y1: float = 0.10
    x2: float = 0.90
    y2: float = 0.95

    def __post_init__(
        self,
    ):
        values = (
            self.x1,
            self.y1,
            self.x2,
            self.y2,
        )

        if not all(
            0.0
            <= value
            <= 1.0
            for value
            in values
        ):
            raise ValueError(
                "Work-zone coordinates "
                "must be between 0 and 1."
            )

        if (
            self.x1
            >= self.x2
            or self.y1
            >= self.y2
        ):
            raise ValueError(
                "Invalid work-zone rectangle."
            )


class ZoneMonitor:
    def __init__(
        self,
        zone: NormalizedWorkZone
        | None = None,
    ):
        self.zone = (
            zone
            or NormalizedWorkZone()
        )

    def pixel_bounds(
        self,
        *,
        frame_width: int,
        frame_height: int,
    ) -> dict[str, int]:
        return {
            "x1": int(
                self.zone.x1
                * frame_width
            ),

            "y1": int(
                self.zone.y1
                * frame_height
            ),

            "x2": int(
                self.zone.x2
                * frame_width
            ),

            "y2": int(
                self.zone.y2
                * frame_height
            ),
        }

    def contains_point(
        self,
        *,
        x: float,
        y: float,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        bounds = (
            self.pixel_bounds(
                frame_width=(
                    frame_width
                ),
                frame_height=(
                    frame_height
                ),
            )
        )

        return bool(
            bounds[
                "x1"
            ]
            <= x
            <= bounds[
                "x2"
            ]
            and bounds[
                "y1"
            ]
            <= y
            <= bounds[
                "y2"
            ]
        )

    def detection_inside_zone(
        self,
        detection: dict[
            str,
            Any,
        ],
        *,
        frame_width: int,
        frame_height: int,
    ) -> bool:
        center = (
            detection[
                "center"
            ]
        )

        return (
            self.contains_point(
                x=float(
                    center[
                        "x"
                    ]
                ),
                y=float(
                    center[
                        "y"
                    ]
                ),
                frame_width=(
                    frame_width
                ),
                frame_height=(
                    frame_height
                ),
            )
        )