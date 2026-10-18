from __future__ import annotations

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.model.motion_features import (
    extract_motion_features_from_frames,
    motion_feature_names,
)
from app.video.temporal_localizer import (
    get_positive_probability,
)


def build_frames(*, moving: bool) -> list[np.ndarray]:
    frames = []
    for index in range(16):
        frame = np.zeros((112, 64, 3), dtype=np.uint8)
        x = 8 + index if moving else 8
        cv2.rectangle(frame, (x, 45), (x + 12, 57), (255, 255, 255), -1)
        frames.append(frame)
    return frames


def main() -> None:
    static = extract_motion_features_from_frames(build_frames(moving=False))
    moving = extract_motion_features_from_frames(build_frames(moving=True))

    expected = len(motion_feature_names())
    assert static.shape == (expected,)
    assert moving.shape == (expected,)
    assert np.isfinite(static).all()
    assert np.isfinite(moving).all()
    assert moving[0] > static[0]
    assert moving[5] > static[5]

    X = np.stack([static, moving])
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42)),
        ]
    )
    model.fit(X, np.asarray([0, 1]))
    probability = get_positive_probability(model, moving.reshape(1, -1))
    assert 0.0 <= probability <= 1.0

    print(f"Motion feature dimension: {expected}")
    print("Dense-motion synthetic test passed.")


if __name__ == "__main__":
    main()
