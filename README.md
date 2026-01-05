# Doc2GUI Operator (Document -> Form-Filling UI Agent)

Ingests PDF instructions, builds a RAG index, uses a PyTorch CNN for layout tagging, and drives a browser with Playwright to complete local web forms.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install
```

Generate PDFs and assets:

```bash
python scripts/generate_pdfs.py
```

Train the layout CNN:

```bash
python cnn/train_layout_cnn.py --output models/layout_cnn.pt
```

Run the local web server:

```bash
python webapp/server.py
```

Run a single agent scenario:

```bash
python -m agent.agent_loop \
  --pdf data/pdfs/insurance_form.pdf \
  --user-data data/user_data/insurance.json \
  --form-url http://127.0.0.1:8000/form/insurance \
  --docs data/docs/policy.txt
```

Run evaluation suite (3 cases):

```bash
python eval/run_eval.py
```

## One-Shot Run

```bash
./scripts/run_all.sh
```

## LLM/VLM Choices

Tier A (fast CPU):
- Transformers: `google/flan-t5-base` (default)
- Ollama alternative: `ollama run phi3` then `--llm-provider ollama --llm-model phi3`

Tier B (better quality, more compute):
- Transformers: `google/flan-t5-large` or `mistralai/Mistral-7B-Instruct-v0.2` (GPU recommended)
- Ollama alternative: `llama3.1:8b`

## Output Artifacts

Each run writes to `runs/<timestamp>/`:
- `screenshots/step_XX.png`
- `actions.json`
- `grounding.json`
- `audit_log.txt`

## Notes

- This demo is safe/local-only. Forms run on `127.0.0.1:8000`.
- CNN layout tagging is applied to PDF text lines to label sections/fields and improve chunk structuring.
