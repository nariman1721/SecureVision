import base64
import cv2


def extract_frame(video_path: str, timestamp_sec: float):
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError("Unable to open video")

    capture.set(
        cv2.CAP_PROP_POS_MSEC,
        timestamp_sec * 1000,
    )

    success, frame = capture.read()
    capture.release()

    if not success:
        raise ValueError(
            f"Unable to extract frame at {timestamp_sec}s"
        )

    return frame


def frame_to_base64(frame):
    success, encoded = cv2.imencode(".jpg", frame)

    if not success:
        raise ValueError("Unable to encode frame")

    return base64.b64encode(
        encoded.tobytes()
    ).decode("utf-8")