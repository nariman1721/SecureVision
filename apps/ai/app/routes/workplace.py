from __future__ import annotations

import hmac
import os
import shutil
import threading
import uuid
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Header,
    HTTPException,
    UploadFile,
    status,
)

from fastapi.responses import JSONResponse

from app.services.workplace_job_store import (
    WORKPLACE_JOB_STORE,
)

from app.workplace.production import (
    ProductionWorkplaceAnalytics,
)


router = APIRouter(
    prefix="/api/v1/workplace",
    tags=["workplace"],
)


AI_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

RUNTIME_DIR = (
    AI_ROOT
    / "runtime"
    / "workplace_jobs"
)

MAX_UPLOAD_BYTES = (
    150
    * 1024
    * 1024
)

UPLOAD_CHUNK_BYTES = (
    1024
    * 1024
)


_ANALYZER = None

_ANALYZER_CREATION_LOCK = (
    threading.Lock()
)

_ANALYSIS_LOCK = (
    threading.Lock()
)


def require_api_key(
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
    ),
) -> str:
    expected = os.getenv(
        "SECUREVISION_API_KEY"
    )

    if not expected:
        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Server API key is not configured."
            ),
        )

    if (
        x_api_key is None
        or not hmac.compare_digest(
            x_api_key,
            expected,
        )
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail="Invalid API key",
        )

    return x_api_key


def get_analyzer(
) -> ProductionWorkplaceAnalytics:
    global _ANALYZER

    if _ANALYZER is None:
        with _ANALYZER_CREATION_LOCK:
            if _ANALYZER is None:
                _ANALYZER = (
                    ProductionWorkplaceAnalytics(
                        sample_interval_sec=0.5,
                        confidence_threshold=0.45,
                        min_absence_sec=2.0,
                    )
                )

    return _ANALYZER


def run_workplace_job(
    *,
    job_id: str,
    video_path: Path,
) -> None:
    try:
        analyzer = get_analyzer()

        with _ANALYSIS_LOCK:
            result = analyzer.analyze(
                video_path,
                show_progress=False,
            )

        WORKPLACE_JOB_STORE.mark_done(
            job_id=job_id,
            result=result,
        )

    except Exception as exc:
        WORKPLACE_JOB_STORE.mark_failed(
            job_id=job_id,
            error=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
        )

    finally:
        shutil.rmtree(
            video_path.parent,
            ignore_errors=True,
        )


@router.post(
    "/analyze",
    status_code=(
        status.HTTP_202_ACCEPTED
    ),
)
async def analyze_workplace(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    _: str = Depends(
        require_api_key
    ),
):
    original_name = (
        file.filename
        or "input.mov"
    )

    extension = (
        Path(original_name)
        .suffix
        .lower()
    )

    if extension != ".mov":
        await file.close()

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Only .mov files are supported."
            ),
        )

    job_id = str(
        uuid.uuid4()
    )

    job_dir = (
        RUNTIME_DIR
        / job_id
    )

    job_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    video_path = (
        job_dir
        / "input.mov"
    )

    total_bytes = 0

    try:
        with video_path.open(
            "wb"
        ) as output:
            while True:
                chunk = await file.read(
                    UPLOAD_CHUNK_BYTES
                )

                if not chunk:
                    break

                total_bytes += len(
                    chunk
                )

                if (
                    total_bytes
                    > MAX_UPLOAD_BYTES
                ):
                    raise HTTPException(
                        status_code=(
                            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
                        ),
                        detail=(
                            "File exceeds 150 MB limit."
                        ),
                    )

                output.write(
                    chunk
                )

    except Exception:
        shutil.rmtree(
            job_dir,
            ignore_errors=True,
        )

        raise

    finally:
        await file.close()

    if total_bytes == 0:
        shutil.rmtree(
            job_dir,
            ignore_errors=True,
        )

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Uploaded file is empty."
            ),
        )

    WORKPLACE_JOB_STORE.create_job(
        job_id=job_id,
        file_name=original_name,
    )

    background_tasks.add_task(
        run_workplace_job,
        job_id=job_id,
        video_path=video_path,
    )

    return JSONResponse(
        status_code=(
            status.HTTP_202_ACCEPTED
        ),
        content={
            "job_id": job_id,
            "analysis_type": (
                "workplace_activity_analytics"
            ),
            "status": "processing",
            "file_name": original_name,
        },
    )


@router.get(
    "/jobs/{job_id}",
)
def get_workplace_job(
    job_id: str,
    _: str = Depends(
        require_api_key
    ),
):
    job = (
        WORKPLACE_JOB_STORE
        .get_job(
            job_id
        )
    )

    if job is None:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail="Job not found",
        )

    return job