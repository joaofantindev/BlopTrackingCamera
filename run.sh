#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "==> criando ambiente virtual"
  python3 -m venv .venv
fi

source .venv/bin/activate

if ! python -c "import cv2" >/dev/null 2>&1; then
  echo "==> instalando dependencias"
  pip install --upgrade pip >/dev/null
  pip install -r requirements.txt
fi

exec python blob_tracker.py "$@"
