from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    path: str
    label: int
    source: str


def _is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VALID_EXTS


def _collect_images_under(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*") if _is_img(p)])


class RealFakeFolderDataset(Dataset):
    def __init__(
        self,
        real_root: str | Path,
        fake_root: str | Path,
        source: str,
        transform: Optional[Callable] = None,
        categories: Optional[Sequence[str]] = None,
        limit_real: Optional[int] = None,
        limit_fake: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.real_root = Path(real_root)
        self.fake_root = Path(fake_root)
        self.source = source
        self.transform = transform
        self.categories = list(categories) if categories else None
        self.limit_real = limit_real
        self.limit_fake = limit_fake
        self.seed = seed
        self.samples = self._build_index()
        if not self.samples:
            raise ValueError(f"no images found for source={source}")

    def _sub_roots(self, root: Path) -> List[Path]:
        if self.categories:
            return [root / x for x in self.categories if (root / x).exists()]
        if any((root / x).is_dir() for x in ["0_real", "1_fake", "real", "fake"]):
            return [root]
        if root.exists() and root.is_dir() and any(_is_img(p) for p in root.iterdir()):
            return [root]
        subs = [p for p in root.iterdir()] if root.exists() else []
        subs = [p for p in subs if p.is_dir()]
        return sorted(subs)

    def _collect_class(self, top_root: Path, class_names: set[str], label: int) -> List[Sample]:
        sub_roots = self._sub_roots(top_root)
        all_paths: List[Path] = []
        for sub in sub_roots:
            class_dirs = [p for p in sub.rglob("*") if p.is_dir() and p.name.lower() in class_names]
            if class_dirs:
                for class_dir in class_dirs:
                    all_paths.extend(_collect_images_under(class_dir))
            else:
                if sub == top_root or self.categories:
                    all_paths.extend(_collect_images_under(sub))
        # de-duplicate in case nested matches overlap
        seen = set()
        uniq: List[Sample] = []
        for p in all_paths:
            s = str(p)
            if s not in seen:
                seen.add(s)
                uniq.append(Sample(path=s, label=label, source=self.source))
        return uniq

    def _limit(self, items: List[Sample], limit: Optional[int]) -> List[Sample]:
        if limit is None or len(items) <= limit:
            return items
        rng = random.Random(self.seed)
        items = items.copy()
        rng.shuffle(items)
        return items[:limit]

    def _build_index(self) -> List[Sample]:
        real = self._collect_class(self.real_root, {"0_real", "real"}, 0)
        fake = self._collect_class(self.fake_root, {"1_fake", "fake"}, 1)
        real = self._limit(real, self.limit_real)
        fake = self._limit(fake, self.limit_fake)
        items = real + fake
        rng = random.Random(self.seed)
        rng.shuffle(items)
        return items

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": sample.label,
            "source": sample.source,
            "path": sample.path,
        }
