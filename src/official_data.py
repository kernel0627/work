from __future__ import annotations

from pathlib import Path

DEFAULT_REPRO_SOURCES = ["stylegan", "biggan", "ldm_200", "dalle"]


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
        return {"source": source, "real_root": base / "0_real", "fake_root": base / "1_fake"}

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
