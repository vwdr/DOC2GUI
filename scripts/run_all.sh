#!/usr/bin/env bash
set -euo pipefail

if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN=python3.12
  VENV_DIR=.venv312
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
  VENV_DIR=.venv
else
  echo "python3 not found. Install Python 3.12+ and retry." >&2
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

"$VENV_PIP" install -r requirements.txt
"$VENV_DIR/bin/playwright" install

"$VENV_PY" scripts/generate_pdfs.py
"$VENV_PY" -m cnn.train_layout_cnn --output models/layout_cnn.pt

PYTHONPATH=. "$VENV_PY" -m webapp.server > /tmp/doc2gui_server.log 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "$SERVER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 20); do
  if curl -s http://127.0.0.1:8000/ >/dev/null; then
    READY=1
    break
  fi
  sleep 0.5
done

if [ "$READY" -ne 1 ]; then
  echo "Server did not start; see /tmp/doc2gui_server.log" >&2
  exit 1
fi

"$VENV_PY" -m eval.run_eval
