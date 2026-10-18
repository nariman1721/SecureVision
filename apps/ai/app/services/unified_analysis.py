from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.video.production_inference import (
    ProductionVideoInference,
)

from app.workplace.production import (
    ProductionWorkplaceAnalytics,
)


class UnifiedSecureVisionAnalysis:
    def __init__(self):
        self.cash_analyzer = (
            ProductionVideoInference()
        )

        self.workplace_analyzer = (
            ProductionWorkplaceAnalytics(
                sample_interval_sec=0.5,
                confidence_threshold=0.45,
                min_absence_sec=2.0,
            )
        )

    def analyze(
        self,
        video_path: str | Path,
        *,
        show_progress: bool = False,
    ) -> dict[str, Any]:
        started = (
            time.perf_counter()
        )

        video_path = Path(
            video_path
        ).resolve()

        if not video_path.exists():
            raise FileNotFoundError(
                f"Video not found: "
                f"{video_path}"
            )

        # -----------------------------------------
        # 1. CASH DISCIPLINE
        # -----------------------------------------

        cash_started = (
            time.perf_counter()
        )

        cash_result = (
            self.cash_analyzer.analyze(
                video_path,
                show_progress=(
                    show_progress
                ),
                include_evidence=True,
            )
        )

        cash_processing_sec = (
            time.perf_counter()
            - cash_started
        )

        # -----------------------------------------
        # 2. WORKPLACE ACTIVITY
        # -----------------------------------------

        workplace_started = (
            time.perf_counter()
        )

        workplace_result = (
            self.workplace_analyzer
            .analyze(
                video_path,
                show_progress=(
                    show_progress
                ),
            )
        )

        workplace_processing_sec = (
            time.perf_counter()
            - workplace_started
        )

        # -----------------------------------------
        # 3. UNIFIED SUMMARY
        # -----------------------------------------

        violations = (
            cash_result.get(
                "violations",
                [],
            )
            or []
        )

        workplace_summary = (
            workplace_result.get(
                "summary",
                {},
            )
        )

        violation_detected = bool(
            cash_result.get(
                "violation_detected",
                len(violations) > 0,
            )
        )

        total_processing_sec = (
            time.perf_counter()
            - started
        )

        return {
            "status": "done",

            "analysis_type": (
                "securevision_full_analysis"
            ),

            "file_name": (
                video_path.name
            ),

            "summary": {
                "cash_discipline": {
                    "verdict": (
                        cash_result.get(
                            "verdict"
                        )
                    ),

                    "violation_detected": (
                        violation_detected
                    ),

                    "violation_count": (
                        len(
                            violations
                        )
                    ),
                },

                "workplace": {
                    "presence_percent": (
                        workplace_summary.get(
                            "presence_percent"
                        )
                    ),

                    "present_sec": (
                        workplace_summary.get(
                            "present_sec"
                        )
                    ),

                    "away_sec": (
                        workplace_summary.get(
                            "away_sec"
                        )
                    ),

                    "sitting_sec": (
                        workplace_summary.get(
                            "sitting_sec"
                        )
                    ),

                    "standing_sec": (
                        workplace_summary.get(
                            "standing_sec"
                        )
                    ),

                    "absence_count": (
                        workplace_summary.get(
                            "absence_count"
                        )
                    ),
                },
            },

            "cash_discipline": (
                cash_result
            ),

            "workplace": (
                workplace_result
            ),

            "processing": {
                "cash_discipline_sec": float(
                    round(
                        cash_processing_sec,
                        3,
                    )
                ),

                "workplace_sec": float(
                    round(
                        workplace_processing_sec,
                        3,
                    )
                ),

                "total_sec": float(
                    round(
                        total_processing_sec,
                        3,
                    )
                ),
            },
        }