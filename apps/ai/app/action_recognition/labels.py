from __future__ import annotations


NO_VIOLATION = 0
VIOLATION = 1


LABEL_TO_NAME = {
    NO_VIOLATION: "no_violation",
    VIOLATION: "violation",
}


NAME_TO_LABEL = {
    value: key
    for key, value in LABEL_TO_NAME.items()
}


def label_to_name(label: int) -> str:
    label = int(label)

    if label not in LABEL_TO_NAME:
        raise ValueError(
            f"Unknown label: {label}"
        )

    return LABEL_TO_NAME[label]


def name_to_label(name: str) -> int:
    normalized = (
        str(name)
        .strip()
        .lower()
    )

    if normalized not in NAME_TO_LABEL:
        raise ValueError(
            f"Unknown label name: {name}"
        )

    return NAME_TO_LABEL[normalized]