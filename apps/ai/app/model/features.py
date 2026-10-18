from __future__ import annotations

import torch


X3D_FEATURE_DIM = 2048


def validate_batch(
    batch: torch.Tensor,
) -> None:
    if not isinstance(
        batch,
        torch.Tensor,
    ):
        raise TypeError(
            "batch must be a torch.Tensor"
        )

    if batch.ndim != 5:
        raise ValueError(
            "Expected X3D batch with "
            "shape [N, C, T, H, W]. "
            f"Received: {tuple(batch.shape)}"
        )

    if batch.shape[1] != 3:
        raise ValueError(
            "Expected 3 RGB channels."
        )


def extract_x3d_features(
    model: torch.nn.Module,
    batch: torch.Tensor,
) -> torch.Tensor:
    """
    Extract the representation immediately
    before the original Kinetics-400
    classification projection.

    Output:
        [N, 2048]
    """

    validate_batch(
        batch
    )

    if not hasattr(
        model,
        "blocks",
    ):
        raise ValueError(
            "Model does not expose "
            "X3D blocks."
        )

    blocks = list(
        model.blocks
    )

    if len(blocks) < 2:
        raise ValueError(
            "Unexpected X3D architecture."
        )

    model.eval()

    with torch.inference_mode():
        x = batch

        # Backbone:
        #
        # stem
        # ↓
        # residual stages
        #
        # but skip original
        # classification head.
        for block in blocks[:-1]:
            x = block(
                x
            )

        head = blocks[-1]

        if not hasattr(
            head,
            "pool",
        ):
            raise ValueError(
                "X3D head does not "
                "contain expected pool."
            )

        if head.pool is not None:
            x = head.pool(
                x
            )

        # IMPORTANT:
        # We intentionally do NOT apply:
        #
        # head.dropout
        # head.proj (2048 -> 400)
        # head.activation
        #
        # because we want the
        # representation before the
        # Kinetics classifier.

        if (
            hasattr(
                head,
                "output_pool",
            )
            and
            head.output_pool
            is not None
        ):
            x = (
                head.output_pool(
                    x
                )
            )

        else:
            x = x.mean(
                dim=(
                    2,
                    3,
                    4,
                ),
                keepdim=True,
            )

        features = (
            x.flatten(
                start_dim=1
            )
        )

    if (
        features.ndim != 2
    ):
        raise RuntimeError(
            "Expected 2D feature matrix."
        )

    if (
        features.shape[1]
        != X3D_FEATURE_DIM
    ):
        raise RuntimeError(
            "Unexpected X3D feature "
            "dimension. "
            f"Expected "
            f"{X3D_FEATURE_DIM}, "
            f"received "
            f"{features.shape[1]}"
        )

    if not torch.isfinite(
        features
    ).all():
        raise RuntimeError(
            "Features contain NaN or Inf."
        )

    return features