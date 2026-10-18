#!/usr/bin/env bash

echo "=========================================="
echo "SecureVision — Service Health Check"
echo "=========================================="

check() {
  local name="$1"
  local url="$2"

  code=$(curl --max-time 5 -s -o /dev/null -w "%{http_code}" "$url")

  if [[ "$code" =~ ^2[0-9][0-9]$ ]]; then
    echo "✅ $name — HTTP $code"
  else
    echo "❌ $name — HTTP $code"
  fi
}

check "Next.js frontend" "http://127.0.0.1:3000/"
check "NestJS backend" "http://127.0.0.1:4000/"
check "FastAPI AI" "http://127.0.0.1:8000/openapi.json"

echo
echo "API endpoints:"
curl --max-time 5 -fsS http://127.0.0.1:8000/openapi.json \
  | python3 -c '
import json,sys
data=json.load(sys.stdin)
for path in data.get("paths",{}):
    if "/api/v1/" in path:
        print(" ",path)
' || echo "⚠️ AI endpoint listing failed"
