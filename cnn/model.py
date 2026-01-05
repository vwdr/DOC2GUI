from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class ModelConfig:
    input_size: Tuple[int, int] = (64, 256)
    num_classes: int = 3


class LayoutCNN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        h, w = config.input_size
        reduced_h = h // 8
        reduced_w = w // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * reduced_h * reduced_w, 128),
            nn.ReLU(),
            nn.Linear(128, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
