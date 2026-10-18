from __future__ import annotations
from app.rules import (
    KazakhstanCashRuleEngine,
)
import time
from pathlib import Path
from typing import Any

from app.video.evidence_extractor import (
    EvidenceExtractor,
)

from app.video.temporal_localizer import (
    TemporalLocalizer,
)


AI_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

DEFAULT_CLASSIFIER_PATH = (
    AI_ROOT
    / "models"
    / "securevision_final_classifier.joblib"
)


class ProductionVideoInference:
    def __init__(
        self,
        *,
        classifier_path: str | Path
        = DEFAULT_CLASSIFIER_PATH,
        max_violations: int = 3,
        minimum_candidate_distance_sec: float = 4.0,
    ):
        self.classifier_path = Path(
            classifier_path
        ).resolve()

        if not self.classifier_path.exists():
            raise FileNotFoundError(
                f"Classifier not found: "
                f"{self.classifier_path}"
            )

        self.max_violations = int(
            max_violations
        )

        self.minimum_candidate_distance_sec = float(
            minimum_candidate_distance_sec
        )

        self.localizer = (
            TemporalLocalizer(
                classifier_path=(
                    self.classifier_path
                ),
                scan_window_sec=6.0,
                scan_stride_sec=2.0,
                smoothing_radius=1,
                support_radius=1,
                top_candidates=5,
            )
        )

        self.evidence_extractor = (
            EvidenceExtractor()
        )
        self.rule_engine = (
            KazakhstanCashRuleEngine()
        )

    def _candidate_is_far_enough(
        self,
        candidate_time: float,
        selected: list[
            dict[str, Any]
        ],
    ) -> bool:
        for existing in selected:
            existing_time = float(
                existing[
                    "timecode_sec"
                ]
            )

            if (
                abs(
                    candidate_time
                    - existing_time
                )
                <
                self.minimum_candidate_distance_sec
            ):
                return False

        return True

    def _select_candidates(
        self,
        localization: dict[
            str,
            Any,
        ],
    ) -> list[dict[str, Any]]:
        threshold = float(
            self.localizer.threshold
        )

        candidates = (
            localization.get(
                "top_candidates",
                [],
            )
        )

        accepted = []

        for candidate in candidates:
            probability = float(
                candidate.get(
                    "peak_probability",
                    0.0,
                )
            )

            if probability < threshold:
                continue

            timecode = float(
                candidate[
                    "timecode_sec"
                ]
            )

            if not (
                self._candidate_is_far_enough(
                    timecode,
                    accepted,
                )
            ):
                continue

            accepted.append(
                candidate
            )

            if (
                len(
                    accepted
                )
                >= self.max_violations
            ):
                break

        return accepted

    def analyze(
        self,
        video_path: str | Path,
        *,
        show_progress: bool = False,
        include_evidence: bool = True,
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

        if (
            video_path.suffix.lower()
            != ".mov"
        ):
            raise ValueError(
                "Production inference "
                "expects a .mov file."
            )

        localization = (
            self.localizer.localize(
                video_path=(
                    video_path
                ),
                show_progress=(
                    show_progress
                ),
            )
        )

        selected_candidates = (
            self._select_candidates(
                localization
            )
        )

        legal_context = (
            self.rule_engine
            .build_generic_violation_context()
        )

        violations = []

        for index, candidate in enumerate(
            selected_candidates,
            start=1,
        ):
            timecode_sec = float(
                candidate[
                    "timecode_sec"
                ]
            )

            probability = float(
                candidate[
                    "peak_probability"
                ]
            )

            if include_evidence:
                evidence = (
                    self.evidence_extractor.extract(
                        video_path=(
                            video_path
                        ),
                        timecode_sec=(
                            timecode_sec
                        ),
                    )
                )
            else:
                evidence = []

            violations.append(
                {
                    "id": (
                        f"violation-{index}"
                    ),

                    # Dataset currently contains
                    # binary labels only.
                    "type": (
                        legal_context[
                            "violation_type"
                        ]
                    ),

                    "timecode_sec": float(
                        round(
                            timecode_sec,
                            3,
                        )
                    ),

                    "confidence": float(
                        round(
                            probability,
                            6,
                        )
                    ),

                    "description": (
                        legal_context[
                            "explanation"
                        ]
                    ),

                    "evidence_frames": (
                        evidence
                    ),

                    "temporal": {
                        "start_sec": float(
                            round(
                                candidate.get(
                                    "start_sec",
                                    timecode_sec,
                                ),
                                3,
                            )
                        ),

                        "end_sec": float(
                            round(
                                candidate.get(
                                    "end_sec",
                                    timecode_sec,
                                ),
                                3,
                            )
                        ),

                        "region_score": float(
                            candidate.get(
                                "region_score",
                                candidate.get(
                                    "score",
                                    0.0,
                                ),
                            )
                        ),

                        "peak_probability": (
                            probability
                        ),

                        "mean_probability": float(
                            candidate.get(
                                "mean_probability",
                                probability,
                            )
                        ),

                        "persistence": float(
                            candidate.get(
                                "persistence",
                                0.0,
                            )
                        ),

                        "windows_in_region": int(
                            candidate.get(
                                "windows_in_region",
                                1,
                            )
                        ),
                    },

                    # Phase 9B will fill this
                    # only after Kazakhstan
                    # normative sources are verified.
                    "legal_status": (
                        legal_context[
                            "legal_status"
                        ]
                    ),

                    "automated_legal_conclusion": (
                        legal_context[
                            "automated_legal_conclusion"
                        ]
                    ),

                    "normative_source": (
                        legal_context[
                            "normative_source"
                        ]
                    ),

                    "potentially_relevant_rules": (
                        legal_context[
                            "potentially_relevant_rules"
                        ]
                    ),
                }
            )

        elapsed_sec = (
            time.perf_counter()
            - started
        )

        detected = bool(
            len(
                violations
            )
            > 0
        )

        if detected:
            overall_legal_context = (
                legal_context
            )
        else:
            overall_legal_context = (
                self.rule_engine
                .build_no_violation_context()
            )

        return {
            "status": (
                "done"
            ),

            "file_name": (
                video_path.name
            ),

            "verdict": (
                "violation"
                if detected
                else "no_violation"
            ),

            "detected": (
                detected
            ),

            "violation_count": int(
                len(
                    violations
                )
            ),

            "violations": (
                violations
            ),

            "legal_context": (
                overall_legal_context
            ),

            "analysis": {
                "duration_sec": float(
                    localization[
                        "duration_sec"
                    ]
                ),

                "windows_scanned": int(
                    localization[
                        "windows_scanned"
                    ]
                ),

                "classifier_threshold": float(
                    self.localizer.threshold
                ),

                "baseline_probability": float(
                    localization[
                        "baseline_probability"
                    ]
                ),

                "candidate_regions_found": int(
                    len(
                        localization.get(
                            "top_candidates",
                            [],
                        )
                    )
                ),

                "accepted_candidates": int(
                    len(
                        selected_candidates
                    )
                ),

                "processing_time_sec": float(
                    round(
                        elapsed_sec,
                        3,
                    )
                ),

                "classifier": (
                    self.classifier_path.name
                ),

                "feature_strategy": (
                    self.localizer.feature_strategy
                ),

                "device": str(
                    self.localizer.device
                ),
            },
        }
