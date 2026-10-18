from __future__ import annotations

import hmac
import os
import shutil
import threading
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Header,
    HTTPException,
    UploadFile,
)

from app.services import (
    job_store,
)

from app.video.production_inference import (
    ProductionVideoInference,
)


AI_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

RUNTIME_DIR = (
    AI_ROOT
    / "runtime"
    / "jobs"
)

MAX_VIDEO_SIZE = (
    150
    * 1024
    * 1024
)

UPLOAD_CHUNK_SIZE = (
    1024
    * 1024
)

router = APIRouter(
    prefix="/api/v1",
    tags=[
        "Analysis",
    ],
)


# =========================================================
# API KEY
# =========================================================

def get_expected_api_key() -> str:
    api_key = os.getenv(
        "SECUREVISION_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "SECUREVISION_API_KEY "
            "environment variable "
            "is not configured."
        )

    return api_key


def require_api_key(
    x_api_key: Annotated[
        str | None,
        Header(
            alias="X-API-Key"
        ),
    ] = None,
) -> None:
    try:
        expected = (
            get_expected_api_key()
        )

    except RuntimeError:
        raise HTTPException(
            status_code=503,
            detail=(
                "API key is not configured "
                "on the server."
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
            status_code=401,
            detail="Invalid API key.",
        )


# =========================================================
# INFERENCE SINGLETON
# =========================================================

_inference_instance: (
    ProductionVideoInference
    | None
) = None

_inference_init_lock = (
    threading.Lock()
)

_inference_run_lock = (
    threading.Lock()
)


def get_inference() -> (
    ProductionVideoInference
):
    global _inference_instance

    if (
        _inference_instance
        is not None
    ):
        return (
            _inference_instance
        )

    with _inference_init_lock:
        if (
            _inference_instance
            is None
        ):
            _inference_instance = (
                ProductionVideoInference()
            )

    return (
        _inference_instance
    )


# =========================================================
# RESULT ADAPTER
# =========================================================

def build_api_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    violations = []

    for violation in (
        result.get(
            "violations",
            [],
        )
    ):
        evidence_frames = [
            frame[
                "base64"
            ]
            for frame
            in violation.get(
                "evidence_frames",
                [],
            )
            if frame.get(
                "base64"
            )
        ]

        violations.append(
            {
                "type": (
                    violation[
                        "type"
                    ]
                ),

                "timecode_sec": float(
                    violation[
                        "timecode_sec"
                    ]
                ),

                "confidence": float(
                    violation[
                        "confidence"
                    ]
                ),

                "description": (
                    violation[
                        "description"
                    ]
                ),

                "evidence_frames": (
                    evidence_frames[
                        :3
                    ]
                ),

                "legal_status": (
                    violation.get(
                        "legal_status"
                    )
                ),

                "automated_legal_conclusion": (
                    violation.get(
                        "automated_legal_conclusion"
                    )
                ),

                "normative_source": (
                    violation.get(
                        "normative_source"
                    )
                ),

                "potentially_relevant_rules": (
                    violation.get(
                        "potentially_relevant_rules",
                        [],
                    )
                ),

                "temporal": (
                    violation.get(
                        "temporal"
                    )
                ),
            }
        )

    return {
        "verdict": (
            result[
                "verdict"
            ]
        ),

        "detected": bool(
            result[
                "detected"
            ]
        ),

        "violation_count": int(
            result[
                "violation_count"
            ]
        ),

        "violations": (
            violations
        ),

        "legal_context": (
            result.get(
                "legal_context"
            )
        ),

        "analysis": (
            result.get(
                "analysis"
            )
        ),
    }


# =========================================================
# BACKGROUND ANALYSIS
# =========================================================

def process_job(
    *,
    job_id: str,
    video_path: Path,
) -> None:
    try:
        inference = (
            get_inference()
        )

        # MPS / classifier access is serialized.
        # This also prevents multiple large
        # video jobs from exhausting memory.
        with _inference_run_lock:
            result = (
                inference.analyze(
                    video_path,
                    show_progress=False,
                    include_evidence=True,
                )
            )

        api_result = (
            build_api_result(
                result
            )
        )

        job_store.set_done(
            job_id=job_id,
            result=api_result,
        )

    except Exception as error:
        job_store.set_failed(
            job_id=job_id,
            error=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )

    finally:
        try:
            if video_path.exists():
                video_path.unlink()

            job_dir = (
                video_path.parent
            )

            if job_dir.exists():
                shutil.rmtree(
                    job_dir,
                    ignore_errors=True,
                )

        except Exception:
            pass


# =========================================================
# POST /api/v1/analyze
# =========================================================

@router.post(
    "/analyze",
    status_code=202,
)
async def analyze_video(
    background_tasks: BackgroundTasks,

    file: UploadFile = File(...),

    x_api_key: Annotated[
        str | None,
        Header(
            alias="X-API-Key"
        ),
    ] = None,
):
    require_api_key(
        x_api_key
    )

    filename = Path(
        file.filename
        or ""
    ).name

    if not filename:
        raise HTTPException(
            status_code=400,
            detail=(
                "Filename is missing."
            ),
        )

    if (
        Path(
            filename
        )
        .suffix
        .lower()
        != ".mov"
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Only .mov files "
                "are supported."
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

    total_size = 0

    try:
        with video_path.open(
            "wb"
        ) as output:
            while True:
                chunk = await file.read(
                    UPLOAD_CHUNK_SIZE
                )

                if not chunk:
                    break

                total_size += len(
                    chunk
                )

                if (
                    total_size
                    > MAX_VIDEO_SIZE
                ):
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            "Video is too large. "
                            "Maximum size is "
                            "150 MB."
                        ),
                    )

                output.write(
                    chunk
                )

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Uploaded file is empty."
                ),
            )

        job_store.create(
            job_id=job_id,
            filename=filename,
        )

        background_tasks.add_task(
            process_job,
            job_id=job_id,
            video_path=video_path,
        )

        return {
            "job_id": job_id,
            "status": "processing",
        }

    except HTTPException:
        shutil.rmtree(
            job_dir,
            ignore_errors=True,
        )

        raise

    except Exception as error:
        shutil.rmtree(
            job_dir,
            ignore_errors=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Unable to store "
                f"uploaded video: {error}"
            ),
        )

    finally:
        await file.close()


# =========================================================
# GET /api/v1/jobs/{job_id}
# =========================================================

@router.get(
    "/jobs/{job_id}"
)
def get_job(
    job_id: str,

    x_api_key: Annotated[
        str | None,
        Header(
            alias="X-API-Key"
        ),
    ] = None,
):
    require_api_key(
        x_api_key
    )

    job = (
        job_store.get(
            job_id
        )
    )

    if job is None:
        raise HTTPException(
            status_code=404,
            detail="Job not found.",
        )

    response = {
        "job_id": (
            job[
                "job_id"
            ]
        ),

        "status": (
            job[
                "status"
            ]
        ),

        "filename": (
            job[
                "filename"
            ]
        ),

        "created_at": (
            job[
                "created_at"
            ]
        ),

        "updated_at": (
            job[
                "updated_at"
            ]
        ),
    }

    if (
        job[
            "status"
        ]
        == "done"
    ):
        response[
            "result"
        ] = job[
            "result"
        ]

    elif (
        job[
            "status"
        ]
        == "failed"
    ):
        response[
            "error"
        ] = job[
            "error"
        ]

    return response