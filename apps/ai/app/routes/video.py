import os
import tempfile
from pathlib import Path

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
)

from app.video.video_analyzer import analyze_video_metadata
from app.video.frame_extractor import (
    extract_frame,
    frame_to_base64,
)

router = APIRouter(prefix="/video", tags=["Video"])

MAX_VIDEO_SIZE = 150 * 1024 * 1024


@router.post("/inspect")
async def inspect_video(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name

    if Path(filename).suffix.lower() != ".mov":
        raise HTTPException(
            status_code=400,
            detail="Only .mov files are supported",
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mov",
            delete=False,
        ) as temp:
            temp_path = temp.name
            total_size = 0

            while chunk := await file.read(1024 * 1024):
                total_size += len(chunk)

                if total_size > MAX_VIDEO_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="Video is too large",
                    )

                temp.write(chunk)

        metadata = analyze_video_metadata(temp_path)

        midpoint = metadata["duration_sec"] / 2

        frame = extract_frame(temp_path, midpoint)

        return {
            "status": "success",
            "filename": filename,
            "metadata": metadata,
            "sample_frame": {
                "timestamp_sec": midpoint,
                "image_base64": frame_to_base64(frame),
            },
            "analysis_status": "metadata_only",
            "message": "ML violation detection is not enabled yet",
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(error),
        )

    finally:
        await file.close()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)