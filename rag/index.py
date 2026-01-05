import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np

from rag.chunker import RagChunk


@dataclass
class RagIndex:
    embeddings: np.ndarray
    chunks: List[RagChunk]
    model_name: str

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "embeddings.npy", self.embeddings)
        payload = {
            "model_name": self.model_name,
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }
        (out_dir / "chunks.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def load(out_dir: Path) -> "RagIndex":
        embeddings = np.load(out_dir / "embeddings.npy")
        payload = json.loads((out_dir / "chunks.json").read_text(encoding="utf-8"))
        chunks = [RagChunk(**chunk) for chunk in payload["chunks"]]
        return RagIndex(embeddings=embeddings, chunks=chunks, model_name=payload["model_name"])
