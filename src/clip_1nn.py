from __future__ import annotations

from dataclasses import dataclass

import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

CLIP_1NN_ARCH = "clip_1nn"
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "openai"
CLIP_BACKBONE = f"{CLIP_MODEL_NAME}/{CLIP_PRETRAINED}"
CLIP_NORM = "clip"
FIXED_THRESHOLD = 0.5
SCORE_MODE = "fake_real_margin"
DEFAULT_BANK_CHUNK_SIZE = 8192


@dataclass
class FeatureBatch:
    features: torch.Tensor
    labels: torch.Tensor
    paths: list[str]


class Clip1NNFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = open_clip.create_model(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
        self.visual = model.visual
        for p in self.visual.parameters():
            p.requires_grad = False
        self.visual.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.visual(x)
        if feat.ndim == 1:
            feat = feat.unsqueeze(0)
        return F.normalize(feat.float(), dim=-1)


@torch.no_grad()
def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    desc: str,
) -> FeatureBatch:
    model.eval()
    feature_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    paths: list[str] = []

    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = torch.as_tensor(batch["label"], dtype=torch.int64)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            feats = model(images)
        feature_chunks.append(feats.detach().cpu())
        label_chunks.append(labels.cpu())
        paths.extend(list(batch["path"]))

    if not feature_chunks:
        raise ValueError(f"no features extracted for '{desc}'")

    return FeatureBatch(
        features=torch.cat(feature_chunks, dim=0),
        labels=torch.cat(label_chunks, dim=0),
        paths=paths,
    )


def split_feature_bank(features: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    real_mask = labels == 0
    fake_mask = labels == 1
    if int(real_mask.sum().item()) == 0:
        raise ValueError("feature bank has no real samples")
    if int(fake_mask.sum().item()) == 0:
        raise ValueError("feature bank has no fake samples")
    return features[real_mask].contiguous(), features[fake_mask].contiguous()


@torch.no_grad()
def _max_similarity(
    query: torch.Tensor,
    bank: torch.Tensor,
    bank_chunk_size: int,
) -> torch.Tensor:
    if bank.ndim != 2 or bank.size(0) == 0:
        raise ValueError("bank must be a non-empty 2D tensor")

    bank_chunk_size = max(int(bank_chunk_size), 1)
    best: torch.Tensor | None = None
    for start in range(0, bank.size(0), bank_chunk_size):
        chunk = bank[start:start + bank_chunk_size].to(device=query.device, dtype=query.dtype)
        sims = query @ chunk.T
        chunk_best = sims.max(dim=1).values
        best = chunk_best if best is None else torch.maximum(best, chunk_best)

    if best is None:
        raise ValueError("failed to compute similarities")
    return best


@torch.no_grad()
def score_1nn_features(
    query_features: torch.Tensor,
    real_bank: torch.Tensor,
    fake_bank: torch.Tensor,
    device: torch.device,
    query_batch_size: int,
    bank_chunk_size: int = DEFAULT_BANK_CHUNK_SIZE,
    desc: str = "score",
) -> torch.Tensor:
    if query_features.ndim != 2:
        raise ValueError("query_features must be a 2D tensor")
    if query_features.size(0) == 0:
        return torch.empty(0, dtype=torch.float32)

    query_batch_size = max(int(query_batch_size), 1)
    scores: list[torch.Tensor] = []

    for start in tqdm(range(0, query_features.size(0), query_batch_size), desc=desc, leave=False):
        query = query_features[start:start + query_batch_size].to(device=device, dtype=torch.float32)
        max_real = _max_similarity(query, real_bank, bank_chunk_size=bank_chunk_size)
        max_fake = _max_similarity(query, fake_bank, bank_chunk_size=bank_chunk_size)
        score = ((max_fake - max_real) + 2.0) / 4.0
        scores.append(score.clamp_(0.0, 1.0).cpu())

    return torch.cat(scores, dim=0)
