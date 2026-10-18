from __future__ import annotations

import json
import sys
import time
import uuid

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "apps/ai/data/raw"

API = "http://127.0.0.1:4000/api/analysis/full"

# Two contrasting test cases.
# Pass "all" on the command line to include a third.
CASES = ["Х-18.mov", "Х-10.mov"]

if "all" in sys.argv:
    CASES.append("Х-26.mov")


def request_json(url, *, data=None, headers=None, method=None):
    req = Request(
        url,
        data=data,
        headers=headers or {},
        method=method,
    )

    try:
        with urlopen(req, timeout=45) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)

    except HTTPError as exc:
        print(f"HTTP ERROR {exc.code}: {exc.read()[:500]!r}")
        raise

    except URLError as exc:
        print(f"NETWORK ERROR: {exc}")
        raise


def upload_video(path):
    boundary = "----SecureVision" + uuid.uuid4().hex

    payload = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="input.mov"\r\n'
        "Content-Type: video/quicktime\r\n"
        "\r\n"
    ).encode("utf-8")

    payload += path.read_bytes()
    payload += f"\r\n--{boundary}--\r\n".encode("utf-8")

    return request_json(
        API,
        data=payload,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )


def wait_for_job(job_id, timeout=240):
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        response = request_json(f"{API}/{job_id}")
        status = response.get("status")

        if status == "done":
            return response

        if status == "failed":
            raise RuntimeError(
                f"AI job failed: {response.get('error')}"
            )

        if status not in ("processing", "pending", "queued"):
            raise RuntimeError(
                f"Unexpected job status: {status!r}"
            )

        time.sleep(2)

    raise TimeoutError(
        f"Job exceeded {timeout} seconds: {job_id}"
    )


def run_case(name):
    path = RAW / name

    if not path.is_file():
        raise FileNotFoundError(
            f"Video not found: {path}"
        )

    print(f"\nVIDEO: {name}")
    print(f"Size: {path.stat().st_size / 1024 / 1024:.2f} MB")

    started = time.monotonic()

    upload = upload_video(path)
    job_id = upload.get("job_id")

    if not job_id:
        raise RuntimeError(
            f"Upload response missing job_id: {upload}"
        )

    print(f"✅ Upload accepted — job {job_id[:8]}...")

    job = wait_for_job(job_id)
    elapsed = round(time.monotonic() - started, 2)

    result = job.get("result")

    if not isinstance(result, dict):
        raise RuntimeError(
            "Finished job has no result object"
        )

    cash = result.get("cash_discipline")
    workplace = result.get("workplace")
    processing = result.get("processing")

    if not isinstance(cash, dict):
        raise RuntimeError("Cash Discipline result missing")

    if not isinstance(workplace, dict):
        raise RuntimeError("Workplace result missing")

    if not isinstance(processing, dict):
        raise RuntimeError("Processing metrics missing")

    violations = cash.get("violations", [])
    timeline = workplace.get("timeline", [])

    if not isinstance(violations, list):
        raise RuntimeError("Violations must be a list")

    if not isinstance(timeline, list) or not timeline:
        raise RuntimeError("Workplace timeline missing")

    annotated_samples = sum(
        isinstance(item.get("people"), list)
        and bool(item["people"])
        for item in timeline
    )

    summary = workplace.get("summary", {})

    print("✅ Job status: done")
    print("✅ Cash Discipline JSON: valid")
    print("✅ Workplace JSON: valid")
    print(f"✅ Workplace samples: {len(timeline)}")
    print(f"✅ Annotated samples: {annotated_samples}")
    print(f"✅ Suspected events: {len(violations)}")
    print(f"✅ Presence: {summary.get('presence_percent', 'N/A')}%")
    print(f"✅ End-to-end wall time: {elapsed}s")

    if elapsed > 300:
        raise RuntimeError("Exceeded five-minute limit")

    return {
        "video": name,
        "status": "passed",
        "wall_time_sec": elapsed,
        "suspected_events": len(violations),
        "timeline_samples": len(timeline),
        "annotated_samples": annotated_samples,
    }


def main():
    print("=" * 60)
    print("SECUREVISION FINAL END-TO-END SMOKE TEST")
    print("=" * 60)

    outcomes = []

    for name in CASES:
        try:
            outcomes.append(run_case(name))
        except Exception as exc:
            print(f"❌ {name}: {exc}")
            raise SystemExit(1) from exc

    output = ROOT / "release_checks"
    output.mkdir(exist_ok=True)

    report = output / "e2e_summary.json"
    report.write_text(
        json.dumps(outcomes, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("✅ ALL END-TO-END TESTS PASSED")
    print(f"Report: {report}")
    print("=" * 60)


if __name__ == "__main__":
    main()
