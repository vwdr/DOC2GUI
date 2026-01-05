from dataclasses import dataclass

import requests


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

    def _ensure_pipeline(self) -> None:
        if self.config.provider != "transformers" or self._pipeline:
            return
        from transformers import pipeline

        self._pipeline = pipeline(
            "text2text-generation",
            model=self.config.model,
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
        outputs = self._pipeline(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.temperature > 0,
            temperature=self.config.temperature,
        )
        return outputs[0]["generated_text"]
