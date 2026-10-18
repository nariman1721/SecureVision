from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.video.production_inference import (
    ProductionVideoInference,
)

from training.diagnose_temporal_signal import (
    load_train_manifest,
)


AI_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

REPORTS_DIR = (
    AI_ROOT
    / "reports"
)


def resolve_video(
    video_id: str,
) -> Path:
    manifest = (
        load_train_manifest()
    )

    rows = (
        manifest[
            manifest[
                "canonical_video_id"
            ]
            .astype(str)
            .eq(
                str(
                    video_id
                )
            )
        ]
    )

    if rows.empty:
        raise RuntimeError(
            f"Video ID not found "
            f"in train manifest: "
            f"{video_id}"
        )

    path = Path(
        str(
            rows.iloc[0][
                "video_path"
            ]
        )
    )

    if not path.is_absolute():
        path = (
            AI_ROOT
            / path
        )

    path = (
        path.resolve()
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Video not found: "
            f"{path}"
        )

    return path


def build_preview(
    result: dict,
) -> dict:
    preview = {
        "status": (
            result[
                "status"
            ]
        ),

        "file_name": (
            result[
                "file_name"
            ]
        ),

        "verdict": (
            result[
                "verdict"
            ]
        ),

        "detected": (
            result[
                "detected"
            ]
        ),

        "violation_count": (
            result[
                "violation_count"
            ]
        ),

        "analysis": (
            result[
                "analysis"
            ]
        ),

        "violations": [],
    }

    for violation in (
        result[
            "violations"
        ]
    ):
        preview[
            "violations"
        ].append(
            {
                "id": (
                    violation[
                        "id"
                    ]
                ),

                "type": (
                    violation[
                        "type"
                    ]
                ),

                "timecode_sec": (
                    violation[
                        "timecode_sec"
                    ]
                ),

                "confidence": (
                    violation[
                        "confidence"
                    ]
                ),

                "description": (
                    violation[
                        "description"
                    ]
                ),

                "evidence_frames": int(
                    len(
                        violation[
                            "evidence_frames"
                        ]
                    )
                ),

                "evidence_sizes_base64": [
                    len(
                        frame[
                            "base64"
                        ]
                    )
                    for frame
                    in violation[
                        "evidence_frames"
                    ]
                ],

                "temporal": (
                    violation[
                        "temporal"
                    ]
                ),

                "normative_source": (
                    violation[
                        "normative_source"
                    ]
                ),
            }
        )

    return preview


def main():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--video-id",
        default="X-18",
    )

    parser.add_argument(
        "--no-evidence",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    video_path = (
        resolve_video(
            args.video_id
        )
    )

    print("=" * 72)
    print(
        "SecureVision — Phase 9A "
        "Production Inference Test"
    )
    print("=" * 72)

    print()
    print(
        "Video ID:",
        args.video_id,
    )

    print(
        "Video:",
        video_path,
    )

    print(
        "Evidence enabled:",
        not args.no_evidence,
    )

    print()
    print("LOADING PIPELINE")
    print("-" * 72)

    inference = (
        ProductionVideoInference()
    )

    print(
        "Classifier:",
        inference.classifier_path.name,
    )

    print(
        "Device:",
        inference.localizer.device,
    )

    print(
        "Threshold:",
        inference.localizer.threshold,
    )

    print()
    print("ANALYZING VIDEO")
    print("-" * 72)

    result = (
        inference.analyze(
            video_path,
            show_progress=True,
            include_evidence=(
                not args.no_evidence
            ),
        )
    )

    output_file = (
        REPORTS_DIR
        / (
            "phase9_production_"
            f"{args.video_id}.json"
        )
    )

    output_file.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    preview = (
        build_preview(
            result
        )
    )

    print()
    print("=" * 72)
    print("RESULT")
    print("=" * 72)

    print(
        json.dumps(
            preview,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print("=" * 72)
    print("OUTPUT")
    print("=" * 72)

    print(
        output_file
    )

    print()
    print(
        "✅ Production inference complete."
    )

    if not args.no_evidence:
        print(
            "✅ Evidence JPEG frames "
            "encoded as base64."
        )


if __name__ == "__main__":
    main()