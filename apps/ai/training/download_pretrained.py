from __future__ import annotations

import hashlib
import json
from pathlib import Path

from torch.hub import load_state_dict_from_url


AI_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = AI_ROOT / "models"

WEIGHTS_FILE = (
    MODELS_DIR
    / "x3d_xs_kinetics400.pyth"
)

MANIFEST_FILE = (
    MODELS_DIR
    / "x3d_xs_kinetics400.json"
)

CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/"
    "pytorchvideo/model_zoo/kinetics/"
    "X3D_XS.pyth"
)


def calculate_sha256(
    path: Path,
) -> str:
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def main():
    print("=" * 72)
    print(
        "SecureVision — Download "
        "Pretrained X3D-XS"
    )
    print("=" * 72)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(
        "Checkpoint URL:"
    )

    print(
        CHECKPOINT_URL
    )

    print()
    print(
        "Local path:"
    )

    print(
        WEIGHTS_FILE
    )

    print()
    print(
        "Downloading/loading official "
        "PyTorchVideo checkpoint..."
    )

    checkpoint = (
        load_state_dict_from_url(
            CHECKPOINT_URL,
            model_dir=str(
                MODELS_DIR
            ),
            map_location="cpu",
            progress=True,
            file_name=(
                WEIGHTS_FILE.name
            ),
        )
    )

    if not WEIGHTS_FILE.exists():
        raise RuntimeError(
            "Checkpoint download "
            "completed but local file "
            "was not found."
        )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Unexpected checkpoint format."
        )

    if (
        "model_state"
        not in checkpoint
    ):
        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state'."
        )

    state_dict = (
        checkpoint[
            "model_state"
        ]
    )

    sha256 = calculate_sha256(
        WEIGHTS_FILE
    )

    manifest = {
        "model": (
            "x3d_xs"
        ),

        "pretraining_dataset": (
            "Kinetics-400"
        ),

        "source": (
            CHECKPOINT_URL
        ),

        "weights_file": (
            WEIGHTS_FILE.name
        ),

        "sha256": (
            sha256
        ),

        "state_dict_parameters": (
            len(state_dict)
        ),
    }

    MANIFEST_FILE.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("CHECKPOINT READY")
    print("=" * 72)

    print(
        "File size MB:",
        round(
            WEIGHTS_FILE
            .stat()
            .st_size
            / (1024 * 1024),
            3,
        ),
    )

    print(
        "SHA256:",
        sha256,
    )

    print(
        "State dict entries:",
        len(state_dict),
    )

    print()
    print(
        "✅ Official X3D-XS "
        "checkpoint is stored locally."
    )


if __name__ == "__main__":
    main()