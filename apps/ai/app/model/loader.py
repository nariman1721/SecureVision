from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import torch


AI_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

WEIGHTS_FILE = (
    AI_ROOT
    / "models"
    / "x3d_xs_kinetics400.pyth"
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    if (
        hasattr(
            torch.backends,
            "mps",
        )
        and torch.backends.mps.is_available()
    ):
        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


def get_weights_path() -> Path:
    return WEIGHTS_FILE


def calculate_weights_sha256() -> str:
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"Weights not found: "
            f"{WEIGHTS_FILE}"
        )

    hasher = hashlib.sha256()

    with WEIGHTS_FILE.open(
        "rb"
    ) as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            hasher.update(
                chunk
            )

    return hasher.hexdigest()


def load_local_weights(
    model: torch.nn.Module,
) -> None:
    if not WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            "\nPretrained X3D weights "
            "are not available locally.\n"
            f"Expected:\n"
            f"{WEIGHTS_FILE}\n\n"
            "Run:\n"
            "python -m "
            "training.download_pretrained"
        )

    checkpoint = torch.load(
        WEIGHTS_FILE,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise RuntimeError(
            "Invalid checkpoint format."
        )

    if (
        "model_state"
        not in checkpoint
    ):
        raise RuntimeError(
            "Checkpoint does not contain "
            "'model_state'."
        )

    model.load_state_dict(
        checkpoint[
            "model_state"
        ],
        strict=True,
    )


@lru_cache(maxsize=2)
def load_model(
    pretrained: bool = True,
):
    """
    Load X3D-XS.

    pretrained=True:
        Uses project-local official
        Kinetics-400 checkpoint.

    pretrained=False:
        Architecture only, random weights.
    """

    from pytorchvideo.models.hub import (
        x3d_xs,
    )

    device = get_device()

    # Never let PyTorchVideo download
    # anything implicitly here.
    model = x3d_xs(
        pretrained=False
    )

    if pretrained:
        load_local_weights(
            model
        )

    model = model.to(
        device
    )

    model.eval()

    return (
        model,
        device,
    )