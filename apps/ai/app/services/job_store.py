from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from typing import Any


def utc_now() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


class JobStore:
    def __init__(self):
        self._jobs: dict[
            str,
            dict[str, Any],
        ] = {}

        self._lock = RLock()

    def create(
        self,
        *,
        job_id: str,
        filename: str,
    ) -> dict[str, Any]:
        now = utc_now()

        job = {
            "job_id": job_id,
            "status": "processing",
            "filename": filename,
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
        }

        with self._lock:
            self._jobs[
                job_id
            ] = job

        return deepcopy(
            job
        )

    def get(
        self,
        job_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(
                job_id
            )

            if job is None:
                return None

            return deepcopy(
                job
            )

    def set_done(
        self,
        *,
        job_id: str,
        result: dict[str, Any],
    ) -> None:
        with self._lock:
            job = self._jobs[
                job_id
            ]

            job[
                "status"
            ] = "done"

            job[
                "result"
            ] = result

            job[
                "error"
            ] = None

            job[
                "updated_at"
            ] = utc_now()

    def set_failed(
        self,
        *,
        job_id: str,
        error: str,
    ) -> None:
        with self._lock:
            job = self._jobs[
                job_id
            ]

            job[
                "status"
            ] = "failed"

            job[
                "result"
            ] = None

            job[
                "error"
            ] = error

            job[
                "updated_at"
            ] = utc_now()


job_store = JobStore()