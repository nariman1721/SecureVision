from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import torch

from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

from torchvision.transforms.functional import (
    to_tensor,
)


PERSON_CLASS_ID = 1


class PersonDetector:
    def __init__(
        self,
        *,
        confidence_threshold: float = 0.45,
    ):
        self.confidence_threshold = float(
            confidence_threshold
        )

        self.device = self._get_device()

        weights = (
            SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        )

        self.model = (
            ssdlite320_mobilenet_v3_large(
                weights=weights,
            )
        )

        self.model = self.model.to(
            self.device
        )

        self.model.eval()

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device(
                "cuda"
            )

        if torch.backends.mps.is_available():
            return torch.device(
                "mps"
            )

        return torch.device(
            "cpu"
        )

    @torch.inference_mode()
    def detect(
        self,
        frame,
    ) -> list[dict[str, Any]]:
        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        tensor = (
            to_tensor(
                rgb
            )
            .to(
                self.device
            )
        )

        prediction = (
            self.model(
                [
                    tensor
                ]
            )[0]
        )

        boxes = (
            prediction[
                "boxes"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        labels = (
            prediction[
                "labels"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        scores = (
            prediction[
                "scores"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        detections = []

        for box, label, score in zip(
            boxes,
            labels,
            scores,
        ):
            if (
                int(
                    label
                )
                != PERSON_CLASS_ID
            ):
                continue

            if (
                float(
                    score
                )
                < self.confidence_threshold
            ):
                continue

            x1, y1, x2, y2 = [
                float(
                    value
                )
                for value
                in box
            ]

            width = max(
                0.0,
                x2 - x1,
            )

            height = max(
                0.0,
                y2 - y1,
            )

            center_x = (
                x1
                + x2
            ) / 2.0

            center_y = (
                y1
                + y2
            ) / 2.0

            detections.append(
                {
                    "class": (
                        "person"
                    ),

                    "confidence": float(
                        score
                    ),

                    "bbox": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    },

                    "center": {
                        "x": center_x,
                        "y": center_y,
                    },

                    "width": width,
                    "height": height,
                    "area": (
                        width
                        * height
                    ),
                }
            )

        detections.sort(
            key=lambda item: (
                item[
                    "confidence"
                ]
            ),
            reverse=True,
        )

        return detections