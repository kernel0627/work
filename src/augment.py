from __future__ import annotations

import io
import random
from dataclasses import dataclass

from PIL import Image, ImageFilter
from torchvision import transforms


class RandomJPEG:
    def __init__(self, prob: float = 0.5, quality_min: int = 30, quality_max: int = 100) -> None:
        self.prob = prob
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        quality = random.randint(self.quality_min, self.quality_max)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomGaussianBlur:
    def __init__(self, prob: float = 0.5, sigma_min: float = 0.0, sigma_max: float = 3.0) -> None:
        self.prob = prob
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.prob:
            return img
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


@dataclass
class AugmentConfig:
    image_size: int = 256
    crop_size: int = 224
    hflip: bool = True
    jpeg_prob: float = 0.5
    jpeg_quality_min: int = 30
    jpeg_quality_max: int = 100
    blur_prob: float = 0.5
    blur_sigma_min: float = 0.0
    blur_sigma_max: float = 3.0
    norm: str = "clip"


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _norm_layer(norm: str):
    if norm == "clip":
        return transforms.Normalize(CLIP_MEAN, CLIP_STD)
    if norm == "imagenet":
        return transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    raise ValueError(f"unknown norm: {norm}")


def build_train_transform(cfg: AugmentConfig):
    ops = [
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomCrop(cfg.crop_size),
    ]
    if cfg.hflip:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        RandomGaussianBlur(prob=cfg.blur_prob, sigma_min=cfg.blur_sigma_min, sigma_max=cfg.blur_sigma_max),
        RandomJPEG(prob=cfg.jpeg_prob, quality_min=cfg.jpeg_quality_min, quality_max=cfg.jpeg_quality_max),
        transforms.ToTensor(),
        _norm_layer(cfg.norm),
    ]
    return transforms.Compose(ops)


def build_eval_transform(norm: str = "clip", image_size: int = 256, crop_size: int = 224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        _norm_layer(norm),
    ])
