from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


AI_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = (
    AI_ROOT / "data" / "external" / "huggingface_datasets.json"
)

REQUIRED_DATASET_FIELDS = {
    "id",
    "url",
    "license",
    "access",
    "commercial_status",
    "modality",
    "relevance",
    "recommended_use",
}

APPROVED_STATUSES = {
    "approved_metadata_only",
    "approved_auxiliary_only",
    "approved_training",
}


def load_registry(path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry_path = Path(path).resolve()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))

    if not isinstance(payload.get("datasets"), list):
        raise ValueError("Registry must contain a datasets list.")

    allowed = set(payload.get("policy", {}).get("allowed_licenses", []))
    if not allowed:
        raise ValueError("Registry must define allowed licenses.")

    seen_ids: set[str] = set()
    for index, dataset in enumerate(payload["datasets"]):
        missing = REQUIRED_DATASET_FIELDS - set(dataset)
        if missing:
            raise ValueError(
                f"Dataset row {index} is missing fields: {sorted(missing)}"
            )

        dataset_id = str(dataset["id"])
        if dataset_id in seen_ids:
            raise ValueError(f"Duplicate dataset id: {dataset_id}")
        seen_ids.add(dataset_id)

        status = str(dataset["commercial_status"])
        license_id = str(dataset["license"])
        if status in APPROVED_STATUSES and license_id not in allowed:
            raise ValueError(
                f"Approved dataset {dataset_id} uses disallowed license "
                f"{license_id}."
            )

    return payload


def build_audit_rows(payload: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for dataset in payload["datasets"]:
        status = str(dataset["commercial_status"])
        rows.append(
            {
                "dataset": str(dataset["id"]),
                "license": str(dataset["license"]),
                "access": str(dataset["access"]),
                "decision": "ALLOW" if status in APPROVED_STATUSES else "BLOCK",
                "scope": status,
            }
        )
    return rows


def print_table(rows: list[dict[str, str]]) -> None:
    headers = ("dataset", "license", "access", "decision", "scope")
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }

    print("  ".join(header.ljust(widths[header]) for header in headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print("  ".join(row[header].ljust(widths[header]) for header in headers))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the license-aware Hugging Face dataset registry."
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument(
        "--allowed-only",
        action="store_true",
        help="Print only candidates allowed for a narrowly defined use.",
    )
    args = parser.parse_args()

    payload = load_registry(args.registry)
    rows = build_audit_rows(payload)
    if args.allowed_only:
        rows = [row for row in rows if row["decision"] == "ALLOW"]

    print(f"Hugging Face registry snapshot: {payload['snapshot_date']}")
    print_table(rows)


if __name__ == "__main__":
    main()
