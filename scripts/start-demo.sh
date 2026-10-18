#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export TORCH_HOME="$ROOT/apps/ai/models/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "SecureVision Demo Mode"
echo "Frontend: http://localhost:3000"
echo "NestJS:   http://localhost:4000"
echo "AI:       http://localhost:8000"
echo "Local Torch weights: $TORCH_HOME"

npm run dev
