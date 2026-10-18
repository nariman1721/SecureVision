from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


DEFAULT_NUM_FRAMES = 16
DEFAULT_WIDTH = 64
DEFAULT_HEIGHT = 112
DEFAULT_GRID_ROWS = 4
DEFAULT_GRID_COLUMNS = 2


def _validate_interval(start_sec: float, end_sec: float) -> None:
    if start_sec < 0:
        raise ValueError("start_sec cannot be negative")
    if end_sec <= start_sec:
        raise ValueError("end_sec must be greater than start_sec")


def read_uniform_clip_frames(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    *,
    num_frames: int = DEFAULT_NUM_FRAMES,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> list[np.ndarray]:
    path = Path(video_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    _validate_interval(float(start_sec), float(end_sec))
    if num_frames < 2:
        raise ValueError("num_frames must be at least 2")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    times = np.linspace(float(start_sec), float(end_sec), num_frames)
    frames: list[np.ndarray] = []
    try:
        for time_sec in times:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(time_sec) * 1000.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                if frames:
                    frames.append(frames[-1].copy())
                    continue
                raise RuntimeError(
                    f"Could not read {path.name} at {time_sec:.3f}s"
                )
            frames.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()

    return frames


def _grid_slices(
    height: int,
    width: int,
    rows: int,
    columns: int,
) -> list[tuple[slice, slice]]:
    if rows <= 0 or columns <= 0:
        raise ValueError("grid dimensions must be positive")
    y_edges = np.linspace(0, height, rows + 1, dtype=int)
    x_edges = np.linspace(0, width, columns + 1, dtype=int)
    return [
        (slice(y_edges[row], y_edges[row + 1]), slice(x_edges[col], x_edges[col + 1]))
        for row in range(rows)
        for col in range(columns)
    ]


def motion_feature_names(
    *,
    grid_rows: int = DEFAULT_GRID_ROWS,
    grid_columns: int = DEFAULT_GRID_COLUMNS,
) -> list[str]:
    names = [
        "global_motion_mean",
        "global_motion_std",
        "global_motion_max",
        "global_motion_p90",
        "global_motion_trend",
        "global_flow_mean",
        "global_flow_p90",
        "global_flow_active_fraction",
    ]
    for cell in range(grid_rows * grid_columns):
        names.extend(
            [
                f"cell_{cell}_diff_mean",
                f"cell_{cell}_diff_std",
                f"cell_{cell}_diff_p90",
                f"cell_{cell}_flow_mean",
                f"cell_{cell}_flow_p90",
                f"cell_{cell}_flow_active_fraction",
                f"cell_{cell}_first_last_diff",
                f"cell_{cell}_edge_density",
            ]
        )
    return names


def extract_motion_features_from_frames(
    frames: list[np.ndarray],
    *,
    grid_rows: int = DEFAULT_GRID_ROWS,
    grid_columns: int = DEFAULT_GRID_COLUMNS,
) -> np.ndarray:
    if len(frames) < 2:
        raise ValueError("At least two frames are required")

    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise ValueError("All frames must have the same shape")

    gray = [
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        for frame in frames
    ]
    differences = [np.abs(current - previous) for previous, current in zip(gray, gray[1:])]

    flow_magnitudes: list[np.ndarray] = []
    for previous, current in zip(gray, gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            (previous * 255.0).astype(np.uint8),
            (current * 255.0).astype(np.uint8),
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_magnitudes.append(np.clip(magnitude, 0.0, 20.0) / 20.0)

    per_step_motion = np.asarray([float(item.mean()) for item in differences])
    per_step_flow = np.asarray([float(item.mean()) for item in flow_magnitudes])
    trend_axis = np.arange(len(per_step_motion), dtype=np.float32)
    trend = float(np.polyfit(trend_axis, per_step_motion, 1)[0])

    features = [
        float(per_step_motion.mean()),
        float(per_step_motion.std()),
        float(per_step_motion.max()),
        float(np.quantile(per_step_motion, 0.90)),
        trend,
        float(per_step_flow.mean()),
        float(np.quantile(per_step_flow, 0.90)),
        float(np.mean(np.concatenate(flow_magnitudes) > 0.02)),
    ]

    height, width = gray[0].shape
    for y_slice, x_slice in _grid_slices(
        height, width, grid_rows, grid_columns
    ):
        cell_diff = np.stack([item[y_slice, x_slice] for item in differences])
        cell_flow = np.stack([item[y_slice, x_slice] for item in flow_magnitudes])
        first_last = np.abs(gray[-1][y_slice, x_slice] - gray[0][y_slice, x_slice])
        edges = cv2.Canny(
            (gray[len(gray) // 2][y_slice, x_slice] * 255.0).astype(np.uint8),
            50,
            150,
        )
        features.extend(
            [
                float(cell_diff.mean()),
                float(cell_diff.std()),
                float(np.quantile(cell_diff, 0.90)),
                float(cell_flow.mean()),
                float(np.quantile(cell_flow, 0.90)),
                float(np.mean(cell_flow > 0.02)),
                float(first_last.mean()),
                float(np.mean(edges > 0)),
            ]
        )

    result = np.asarray(features, dtype=np.float32)
    expected_size = len(
        motion_feature_names(grid_rows=grid_rows, grid_columns=grid_columns)
    )
    if result.shape != (expected_size,):
        raise RuntimeError(
            f"Unexpected motion feature shape {result.shape}; expected {(expected_size,)}"
        )
    if not np.isfinite(result).all():
        raise RuntimeError("Motion features contain non-finite values")
    return result


def extract_motion_features(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    *,
    num_frames: int = DEFAULT_NUM_FRAMES,
) -> np.ndarray:
    frames = read_uniform_clip_frames(
        video_path,
        start_sec,
        end_sec,
        num_frames=num_frames,
    )
    return extract_motion_features_from_frames(frames)
