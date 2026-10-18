# SecureVision — Demo Setup

## Prerequisites

- Node.js and npm
- Python 3.11 environment in apps/ai/.venv
- Project dependencies already installed
- PostgreSQL is not required by the current
  in-memory AI job pipeline
- Local model files provisioned under apps/ai/models
- Three free ports: 3000, 4000, 8000

## Start

From repository root:

    ./scripts/start-demo.sh

Wait for all services to become ready.

In a second terminal:

    ./scripts/check-services.sh

Open:

    http://localhost:3000

## End-to-end smoke test

    python3 scripts/e2e_smoke.py

Optional additional test:

    python3 scripts/e2e_smoke.py all

## Demo flow

1. Select a permitted MOV video.
2. Start analysis.
3. Wait for the job to finish.
4. Show detected events and evidence.
5. Show workplace activity statistics.
6. Show annotated video and frame gallery.
7. Export JSON report.

## Troubleshooting

Frontend not available:
- Check port 3000
- Inspect Next.js terminal output

NestJS unavailable:
- Check port 4000
- Check NestJS environment and logs

AI unavailable:
- Check port 8000
- Check apps/ai/.venv
- Check local model files and TORCH_HOME

Upload fails:
- Ensure MOV format and file size below 150 MB
- Check API logs and network requests

Model tries to download:
- Verify local TorchVision checkpoint
- Check TORCH_HOME is forwarded into the AI process

## Data safety

Do not publish private videos, label sheets,
API keys or temporary job directories.
