from __future__ import annotations

from dataclasses import dataclass

import open_clip
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


@dataclass
class BuiltModel:
    model: nn.Module
    norm: str


class ClipLinearProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = open_clip.create_model("ViT-L-14", pretrained="openai")
        self.visual = model.visual
        for p in self.visual.parameters():
            p.requires_grad = False
        embed_dim = getattr(model, "text_projection").shape[1]
        self.head = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.visual(x)
        if feat.ndim == 1:
            feat = feat.unsqueeze(0)
        return self.head(feat).squeeze(-1)


class ResNetBinary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, 1)
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)


def build_model(arch: str) -> BuiltModel:
    arch = arch.lower()
    if arch == "clip_linear":
        return BuiltModel(model=ClipLinearProbe(), norm="clip")
    if arch == "resnet50":
        return BuiltModel(model=ResNetBinary(), norm="imagenet")
    raise ValueError(f"unsupported arch: {arch}")
