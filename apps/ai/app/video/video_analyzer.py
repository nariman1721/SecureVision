import cv2


def analyze_video_metadata(video_path: str):
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ValueError("Cannot open video")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    capture.release()

    if fps <= 0:
        raise ValueError("Invalid video FPS")

    duration = frame_count / fps

    return {
        "duration_sec": round(duration, 3),
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": width,
        "height": height,
    }