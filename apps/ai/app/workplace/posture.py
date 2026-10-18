from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PostureThresholds:
    # Standing person usually occupies
    # a larger vertical part of the frame.
    standing_height_ratio: float = 0.30

    # Sitting person normally has
    # a smaller vertical bounding box.
    sitting_height_ratio: float = 0.27

    # Standing person's upper bbox point
    # normally appears higher in the frame.
    standing_top_y_ratio: float = 0.34

    # Sitting person's upper bbox point
    # normally appears lower.
    sitting_top_y_ratio: float = 0.37


class PostureEstimator:
    """
    Camera-calibrated posture estimation.

    Uses only anonymous person bounding-box geometry.

    No face recognition.
    No biometric identification.

    Output:
        sitting
        standing
        unknown
    """

    def __init__(
        self,
        thresholds: PostureThresholds | None = None,
    ):
        self.thresholds = (
            thresholds
            or PostureThresholds()
        )

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:
        return max(
            minimum,
            min(
                maximum,
                value,
            ),
        )

    def classify(
        self,
        detection: dict[str, Any],
        *,
        frame_width: int,
        frame_height: int,
    ) -> dict[str, Any]:
        bbox = detection[
            "bbox"
        ]

        x1 = float(
            bbox["x1"]
        )

        y1 = float(
            bbox["y1"]
        )

        x2 = float(
            bbox["x2"]
        )

        y2 = float(
            bbox["y2"]
        )

        width = max(
            1.0,
            x2 - x1,
        )

        height = max(
            1.0,
            y2 - y1,
        )

        height_ratio = (
            height
            / float(
                frame_height
            )
        )

        width_ratio = (
            width
            / float(
                frame_width
            )
        )

        top_y_ratio = (
            y1
            / float(
                frame_height
            )
        )

        bottom_y_ratio = (
            y2
            / float(
                frame_height
            )
        )

        aspect_ratio = (
            height
            / width
        )

        thresholds = (
            self.thresholds
        )

        # -------------------------------------------------
        # STRONG STANDING SIGNAL
        # -------------------------------------------------

        if (
            height_ratio
            >= thresholds
            .standing_height_ratio
        ):
            posture = (
                "standing"
            )

            height_score = (
                self._clamp(
                    (
                        height_ratio
                        - thresholds
                        .sitting_height_ratio
                    )
                    /
                    (
                        thresholds
                        .standing_height_ratio
                        - thresholds
                        .sitting_height_ratio
                    )
                )
            )

            top_score = (
                self._clamp(
                    (
                        thresholds
                        .sitting_top_y_ratio
                        - top_y_ratio
                    )
                    /
                    (
                        thresholds
                        .sitting_top_y_ratio
                        - thresholds
                        .standing_top_y_ratio
                    )
                )
            )

            confidence = (
                height_score
                + top_score
            ) / 2.0

        # -------------------------------------------------
        # STRONG SITTING SIGNAL
        # -------------------------------------------------

        elif (
            height_ratio
            <= thresholds
            .sitting_height_ratio
        ):
            posture = (
                "sitting"
            )

            height_score = (
                self._clamp(
                    (
                        thresholds
                        .standing_height_ratio
                        - height_ratio
                    )
                    /
                    (
                        thresholds
                        .standing_height_ratio
                        - thresholds
                        .sitting_height_ratio
                    )
                )
            )

            top_score = (
                self._clamp(
                    (
                        top_y_ratio
                        - thresholds
                        .standing_top_y_ratio
                    )
                    /
                    (
                        thresholds
                        .sitting_top_y_ratio
                        - thresholds
                        .standing_top_y_ratio
                    )
                )
            )

            confidence = (
                height_score
                + top_score
            ) / 2.0

        # -------------------------------------------------
        # AMBIGUOUS HEIGHT:
        # use upper bbox position as second signal
        # -------------------------------------------------

        else:
            if (
                top_y_ratio
                <= thresholds
                .standing_top_y_ratio
            ):
                posture = (
                    "standing"
                )

                confidence = 0.65

            elif (
                top_y_ratio
                >= thresholds
                .sitting_top_y_ratio
            ):
                posture = (
                    "sitting"
                )

                confidence = 0.65

            else:
                posture = (
                    "unknown"
                )

                confidence = 0.0

        return {
            "posture": (
                posture
            ),

            "heuristic_confidence": float(
                round(
                    confidence,
                    4,
                )
            ),

            "features": {
                "height_ratio": float(
                    round(
                        height_ratio,
                        4,
                    )
                ),

                "width_ratio": float(
                    round(
                        width_ratio,
                        4,
                    )
                ),

                "top_y_ratio": float(
                    round(
                        top_y_ratio,
                        4,
                    )
                ),

                "bottom_y_ratio": float(
                    round(
                        bottom_y_ratio,
                        4,
                    )
                ),

                "aspect_ratio": float(
                    round(
                        aspect_ratio,
                        4,
                    )
                ),
            },

            "method": (
                "camera_calibrated_bbox_geometry"
            ),
        }