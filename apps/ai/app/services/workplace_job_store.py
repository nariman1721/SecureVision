from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import RLock
from typing import Any


def _utc_now() -> str:
    return (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )


class WorkplaceJobStore:
    def __init__(self):
        self._jobs: dict[
            str,
            dict[str, Any],
        ] = {}

        self._lock = (
            RLock()
        )

    def create_job(
        self,
        *,
        job_id: str,
        file_name: str,
    ) -> dict[str, Any]:
        now = (
            _utc_now()
        )

        job = {
            "job_id": (
                job_id
            ),

            "analysis_type": (
                "workplace_activity_analytics"
            ),

            "status": (
                "processing"
            ),

            "file_name": (
                file_name
            ),

            "created_at": (
                now
            ),

            "updated_at": (
                now
            ),

            "result": (
                None
            ),

            "error": (
                None
            ),
        }

        with self._lock:
            self._jobs[
                job_id
            ] = job

        return deepcopy(
            job
        )

    def get_job(
        self,
        job_id: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            job = (
                self._jobs.get(
                    job_id
                )
            )

            if job is None:
                return None

            return deepcopy(
                job
            )

    def mark_done(
        self,
        *,
        job_id: str,
        result: dict[str, Any],
    ) -> None:
        with self._lock:
            job = (
                self._jobs.get(
                    job_id
                )
            )

            if job is None:
                return

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
            ] = _utc_now()

    def mark_failed(
        self,
        *,
        job_id: str,
        error: str,
    ) -> None:
        with self._lock:
            job = (
                self._jobs.get(
                    job_id
                )
            )

            if job is None:
                return

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
            ] = _utc_now()


WORKPLACE_JOB_STORE = (
    WorkplaceJobStore()
)