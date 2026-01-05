from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pdfplumber
import torch
from torchvision import transforms

from cnn.model import LayoutCNN, ModelConfig


LABELS = ["header", "label", "other"]


@dataclass
class LayoutLine:
    text: str
    tag: str
    page_num: int
    bbox: Tuple[float, float, float, float]


def _group_words_to_lines(words: List[dict], y_tolerance: float = 3.0) -> List[dict]:
    lines = []
    for word in sorted(words, key=lambda w: (w["top"], w["x0"])):
        placed = False
        for line in lines:
            if abs(word["top"] - line["top"]) <= y_tolerance:
                line["words"].append(word)
                line["top"] = min(line["top"], word["top"])
                line["bottom"] = max(line["bottom"], word["bottom"])
                placed = True
                break
        if not placed:
            lines.append({"top": word["top"], "bottom": word["bottom"], "words": [word]})

    results = []
    for line in lines:
        words_sorted = sorted(line["words"], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in words_sorted).strip()
        x0 = min(w["x0"] for w in words_sorted)
        x1 = max(w["x1"] for w in words_sorted)
        results.append({"text": text, "x0": x0, "x1": x1, "top": line["top"], "bottom": line["bottom"]})
    return results


def _load_model(model_path: Path) -> Tuple[LayoutCNN, ModelConfig, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    config = ModelConfig(**checkpoint["config"])
    model = LayoutCNN(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, config, device


def classify_pdf_lines(pdf_path: Path, model_path: Path) -> List[LayoutLine]:
    model, config, device = _load_model(model_path)
    transform = transforms.Compose([
        transforms.Resize((config.input_size[0], config.input_size[1])),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    results: List[LayoutLine] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words()
            if not words:
                continue
            lines = _group_words_to_lines(words)
            page_image = page.to_image(resolution=150).original.convert("L")
            scale_x = page_image.width / page.width
            scale_y = page_image.height / page.height

            for line in lines:
                if not line["text"]:
                    continue
                x0 = max(0, int(line["x0"] * scale_x) - 2)
                x1 = min(page_image.width, int(line["x1"] * scale_x) + 2)
                top = max(0, int(line["top"] * scale_y) - 2)
                bottom = min(page_image.height, int(line["bottom"] * scale_y) + 2)
                crop = page_image.crop((x0, top, x1, bottom))
                tensor = transform(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(tensor)
                    pred = torch.argmax(logits, dim=1).item()
                tag = LABELS[pred]
                results.append(LayoutLine(text=line["text"], tag=tag, page_num=page_num, bbox=(line["x0"], line["top"], line["x1"], line["bottom"])))

    return results


def extract_sections(layout_lines: List[LayoutLine]) -> List[str]:
    sections = []
    for line in layout_lines:
        if line.tag == "header":
            sections.append(line.text)
    return sections
