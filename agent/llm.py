from dataclasses import dataclass
import os

import requests
import torch


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class LLMConfig:
    provider: str  # "ollama" or "transformers"
    model: str
    temperature: float = 0.0
    max_new_tokens: int = 512


class LLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        self._pipeline = None

    def _infer_device(self):
        if torch.cuda.is_available():
            return 0
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return -1

    def _ensure_pipeline(self) -> None:
        if self.config.provider != "transformers" or self._pipeline:
            return
        from transformers import pipeline

        self._pipeline = pipeline(
            "text2text-generation",
            model=self.config.model,
            device=self._infer_device(),
        )

    def generate(self, prompt: str) -> str:
        if self.config.provider == "ollama":
            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_new_tokens,
                },
            }
            response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
            response.raise_for_status()
            return response.json().get("response", "")

        self._ensure_pipeline()
        if not self._pipeline:
            raise RuntimeError("Transformers pipeline not initialized")
        gen_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.temperature > 0,
        }
        if self.config.temperature > 0:
            gen_kwargs["temperature"] = self.config.temperature
        outputs = self._pipeline(prompt, **gen_kwargs)
        return outputs[0]["generated_text"]
