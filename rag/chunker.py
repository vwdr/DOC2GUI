from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber

from cnn.layout_infer import LayoutLine


@dataclass
class RagChunk:
    chunk_id: str
    text: str
    meta: Dict[str, str]


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _chunk_words(words: List[str], max_tokens: int) -> List[str]:
    chunks = []
    buffer: List[str] = []
    for word in words:
        buffer.append(word)
        if len(buffer) >= max_tokens:
            chunks.append(" ".join(buffer))
            buffer = []
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks


def build_chunks(pdf_path: Path, layout_lines: Optional[List[LayoutLine]] = None, max_tokens: int = 180) -> List[RagChunk]:
    layout_map: Dict[str, str] = {}
    if layout_lines:
        for line in layout_lines:
            layout_map[_normalize(line.text)] = line.tag

    chunks: List[RagChunk] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            current_section = "General"
            page_tokens: List[str] = []
            for line in lines:
                tag = layout_map.get(_normalize(line))
                if tag == "header":
                    current_section = line
                page_tokens.append(line)
            if not page_tokens:
                continue
            chunks_text = _chunk_words(" ".join(page_tokens).split(), max_tokens)
            for idx, chunk_text in enumerate(chunks_text):
                chunk_id = f"{pdf_path.stem}_p{page_num}_c{idx}"
                header = f"SECTION: {current_section}\n"
                chunk_full = header + chunk_text
                chunks.append(
                    RagChunk(
                        chunk_id=chunk_id,
                        text=chunk_full,
                        meta={
                            "source": str(pdf_path),
                            "page": str(page_num),
                            "section": current_section,
                        },
                    )
                )
    return chunks
