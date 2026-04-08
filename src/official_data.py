from __future__ import annotations

from pathlib import Path

DEFAULT_REPRO_SOURCES = ["stylegan", "biggan", "ldm_200", "dalle"]


def _has_nested_class_dirs(base: Path) -> bool:
    if not base.is_dir():
        return False
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if (child / "0_real").is_dir() and (child / "1_fake").is_dir():
            return True
        if (child / "real").is_dir() and (child / "fake").is_dir():
            return True
    return False


def _resolve_cnn_source(base: Path, source: str) -> dict:
    flat_real = base / "0_real"
    flat_fake = base / "1_fake"
    if flat_real.is_dir() and flat_fake.is_dir():
        return {"source": source, "real_root": flat_real, "fake_root": flat_fake, "layout": "flat"}

    named_real = base / "real"
    named_fake = base / "fake"
    if named_real.is_dir() and named_fake.is_dir():
        return {"source": source, "real_root": named_real, "fake_root": named_fake, "layout": "flat_named"}

    if _has_nested_class_dirs(base):
        return {"source": source, "real_root": base, "fake_root": base, "layout": "nested"}

    if not base.exists():
        raise FileNotFoundError(f"source '{source}' expected dataset directory '{base}', but it does not exist")

    raise FileNotFoundError(
        f"source '{source}' has unsupported layout under '{base}'. "
        "Expected either direct 0_real/1_fake folders or category subdirectories containing them."
    )


def resolve_official_eval_pair(data_root: str | Path, source: str) -> dict:
    root = Path(data_root)
    source = source.lower()

    cnn_sources = {
        "progan": root / "test" / "progan",
        "cyclegan": root / "test" / "cyclegan",
        "biggan": root / "test" / "biggan",
        "stylegan": root / "test" / "stylegan",
        "gaugan": root / "test" / "gaugan",
        "stargan": root / "test" / "stargan",
        "deepfake": root / "test" / "deepfake",
        "seeingdark": root / "test" / "seeingdark",
        "san": root / "test" / "san",
        "crn": root / "test" / "crn",
        "imle": root / "test" / "imle",
    }
    if source in cnn_sources:
        base = cnn_sources[source]
        return _resolve_cnn_source(base, source)

    diffusion_root = root / "diffusion_datasets"
    diffusion_pairs = {
        "guided": (diffusion_root / "imagenet", diffusion_root / "guided"),
        "ldm_200": (diffusion_root / "laion", diffusion_root / "ldm_200"),
        "ldm_200_cfg": (diffusion_root / "laion", diffusion_root / "ldm_200_cfg"),
        "ldm_100": (diffusion_root / "laion", diffusion_root / "ldm_100"),
        "glide_100_27": (diffusion_root / "laion", diffusion_root / "glide_100_27"),
        "glide_50_27": (diffusion_root / "laion", diffusion_root / "glide_50_27"),
        "glide_100_10": (diffusion_root / "laion", diffusion_root / "glide_100_10"),
        "dalle": (diffusion_root / "laion", diffusion_root / "dalle"),
    }
    if source in diffusion_pairs:
        real_root, fake_root = diffusion_pairs[source]
        return {"source": source, "real_root": real_root, "fake_root": fake_root}

    raise ValueError(f"unknown source: {source}")
