import argparse
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from cnn.model import LayoutCNN, ModelConfig


LABELS = ["header", "label", "other"]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _render_text(text: str, size: int, image_size: Tuple[int, int]) -> Image.Image:
    img = Image.new("L", image_size, color=255)
    draw = ImageDraw.Draw(img)
    font = _load_font(size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = max(2, (image_size[1] - text_w) // 2)
    y = max(2, (image_size[0] - text_h) // 2)
    draw.text((x, y), text, fill=0, font=font)
    return img


def _render_other(image_size: Tuple[int, int]) -> Image.Image:
    img = Image.new("L", image_size, color=255)
    draw = ImageDraw.Draw(img)
    for _ in range(3):
        x0 = random.randint(0, image_size[1] - 10)
        y0 = random.randint(0, image_size[0] - 10)
        x1 = random.randint(x0 + 5, min(image_size[1], x0 + 80))
        y1 = random.randint(y0 + 5, min(image_size[0], y0 + 30))
        draw.rectangle([x0, y0, x1, y1], outline=0, width=1)
    return img


class SyntheticLayoutDataset(Dataset):
    def __init__(self, num_samples: int, image_size: Tuple[int, int]):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        label = random.randint(0, len(LABELS) - 1)
        if label == 0:
            text = random.choice([
                "CLAIM DETAILS",
                "APPLICANT INFO",
                "MEDICAL SUMMARY",
                "EMPLOYMENT HISTORY",
            ])
            img = _render_text(text, size=26, image_size=self.image_size)
        elif label == 1:
            text = random.choice([
                "Full Name:",
                "Policy ID:",
                "Incident Date:",
                "Phone:",
                "Symptoms:",
            ])
            img = _render_text(text, size=18, image_size=self.image_size)
        else:
            img = _render_other(self.image_size)
        tensor = self.transform(img)
        return tensor, label


def train(model_path: Path, epochs: int = 3, batch_size: int = 32) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    config = ModelConfig()
    model = LayoutCNN(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = SyntheticLayoutDataset(num_samples=1200, image_size=config.input_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        print(f"epoch={epoch+1} loss={avg_loss:.4f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": config.__dict__}, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/layout_cnn.pt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train(Path(args.output), epochs=args.epochs, batch_size=args.batch_size)
