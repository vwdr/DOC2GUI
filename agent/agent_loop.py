import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from playwright.sync_api import sync_playwright

from agent.executor import execute_actions
from agent.llm import LLM, LLMConfig
from agent.logger import RunLogger
from agent.planner import FieldInfo, plan_actions
from agent.utils import extract_fields
from rag.build_index import build_rag_index
from rag.index import RagIndex
from rag.retriever import retrieve


def _load_user_data(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_index(pdf: Path, docs: List[Path], model_path: Path, embed_model: str, index_dir: Path, rebuild: bool) -> RagIndex:
    if (index_dir / "embeddings.npy").exists() and not rebuild:
        return RagIndex.load(index_dir)
    return build_rag_index(pdf, docs, model_path, embed_model, index_dir)


def _build_query(fields: List[FieldInfo], user_data: Dict[str, Any]) -> str:
    field_text = ", ".join(f"{f.field_id}({f.label})" for f in fields)
    return f"Form fields: {field_text}. User data keys: {', '.join(user_data.keys())}. Provide instructions and constraints."  # noqa: E501


def run_agent(args: argparse.Namespace) -> Path:
    pdf_path = Path(args.pdf)
    user_data = _load_user_data(Path(args.user_data))
    docs = [Path(doc) for doc in args.docs]
    index_dir = Path(args.index_dir)
    model_path = Path(args.model_path)

    llm = LLM(LLMConfig(provider=args.llm_provider, model=args.llm_model, temperature=0.0))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=args.headless)
        page = browser.new_page()
        page.goto(args.form_url, wait_until="networkidle")
        fields = extract_fields(page)

        index = _ensure_index(pdf_path, docs, model_path, args.embed_model, index_dir, args.rebuild_index)
        query = _build_query(fields, user_data)
        retrieved = retrieve(index, query, k=args.top_k)
        retrieved_chunks = [{"chunk_id": chunk.chunk_id, "text": chunk.text} for chunk, _ in retrieved]

        actions = plan_actions(llm, fields, user_data, retrieved_chunks)

        logger = RunLogger(Path(args.run_dir))
        execute_actions(page, actions, logger)
        logger.save()

        chunk_text_map = {chunk.chunk_id: chunk.text for chunk in index.chunks}
        logger.save_audit_log(chunk_text_map)

        browser.close()

    return logger.run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--user-data", required=True)
    parser.add_argument("--form-url", required=True)
    parser.add_argument("--docs", nargs="*", default=[])
    parser.add_argument("--index-dir", default="data/index")
    parser.add_argument("--model-path", default="models/layout_cnn.pt")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--llm-provider", default="transformers")
    parser.add_argument("--llm-model", default="google/flan-t5-base")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--run-dir", default="runs")
    args = parser.parse_args()

    run_dir = run_agent(args)
    print(f"Run saved to {run_dir}")
