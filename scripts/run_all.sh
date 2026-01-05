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

REPORT_PATH="runs/eval_report.json"
RESULT_HTML="/tmp/doc2gui_result.html"

if [ -f "$REPORT_PATH" ]; then
  "$VENV_PY" - <<'PY'
import json
from pathlib import Path

report_path = Path("runs/eval_report.json")
reports = json.loads(report_path.read_text(encoding="utf-8"))
all_passed = all(r.get("accuracy") == 1.0 for r in reports)
status = "PASS" if all_passed else "FAIL"

print(f"RESULT: {status}")
for r in reports:
    print(
        f"- {r['case']}: accuracy={r['accuracy']} retries={r['retries']} run_dir={r['run_dir']}"
    )

html_rows = []
for r in reports:
    html_rows.append(
        f"<tr><td>{r['case']}</td><td>{r['accuracy']}</td><td>{r['retries']}</td><td>{r['run_dir']}</td></tr>"
    )

rows = "\n".join(html_rows)
html = f"""<!doctype html>
<html>
  <head>
    <meta charset='utf-8'/>
    <title>Doc2GUI Run Results</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 24px; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
      th {{ background: #f0f0f0; }}
      .status {{ font-weight: bold; }}
    </style>
  </head>
  <body>
    <h1>Doc2GUI Run Results</h1>
    <p class='status'>Status: {status}</p>
    <table>
      <thead>
        <tr><th>Case</th><th>Accuracy</th><th>Retries</th><th>Run Dir</th></tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    <p>Submissions are in <code>webapp/submissions/</code></p>
    <p>Artifacts are in <code>runs/&lt;timestamp&gt;/</code></p>
  </body>
</html>
"""

Path("/tmp/doc2gui_result.html").write_text(html, encoding="utf-8")
PY

  echo "Result page saved to $RESULT_HTML"
  echo "Open it with: open $RESULT_HTML (macOS) or xdg-open $RESULT_HTML (Linux)"
else
  echo "Missing $REPORT_PATH"
fi
