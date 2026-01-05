import argparse
from pathlib import Path
from typing import List

import pdfplumber

from cnn.layout_infer import classify_pdf_lines
from rag.chunker import RagChunk, build_chunks
from rag.index import RagIndex
from rag.retriever import build_index


def _chunk_text(text: str, source: str, max_tokens: int = 180) -> List[RagChunk]:
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk_words = words[i : i + max_tokens]
        chunk_id = f"{Path(source).stem}_c{i // max_tokens}"
        chunks.append(
            RagChunk(
                chunk_id=chunk_id,
                text=" ".join(chunk_words),
                meta={"source": source, "page": "-", "section": "Reference"},
            )
        )
    return chunks


def _extract_pdf_text(pdf_path: Path) -> str:
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def build_rag_index(pdf_path: Path, docs: List[Path], model_path: Path, embedding_model: str, out_dir: Path) -> RagIndex:
    layout_lines = []
    if model_path.exists():
        layout_lines = classify_pdf_lines(pdf_path, model_path)
    pdf_chunks = build_chunks(pdf_path, layout_lines)

    doc_chunks: List[RagChunk] = []
    for doc in docs:
        if doc.suffix.lower() == ".pdf":
            text = _extract_pdf_text(doc)
            doc_chunks.extend(_chunk_text(text, str(doc)))
        else:
            text = doc.read_text(encoding="utf-8")
            doc_chunks.extend(_chunk_text(text, str(doc)))

    all_chunks = pdf_chunks + doc_chunks
    index = build_index(all_chunks, embedding_model)
    index.save(out_dir)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--docs", nargs="*", default=[])
    parser.add_argument("--model-path", default="models/layout_cnn.pt")
    parser.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--out", default="data/index")
    args = parser.parse_args()

    build_rag_index(
        pdf_path=Path(args.pdf),
        docs=[Path(doc) for doc in args.docs],
        model_path=Path(args.model_path),
        embedding_model=args.embed_model,
        out_dir=Path(args.out),
    )
