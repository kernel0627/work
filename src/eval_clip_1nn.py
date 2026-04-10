from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_BANK_CHUNK_SIZE = 8192
DEFAULT_SOURCES = ["stylegan", "biggan", "ldm_200", "dalle"]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./datasets")
    p.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--output-dir", type=str, default="./eval_results/clip_1nn_subset")
    p.add_argument("--bank-chunk-size", type=int, default=DEFAULT_BANK_CHUNK_SIZE)
    p.add_argument("--no-tensorboard", action="store_true")
    return p.parse_args()


def build_loader(dataset, batch_size: int, num_workers: int, device):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )


def _require_dataset_root(path: Path, source: str, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"resolved {kind}_root for source '{source}' does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"resolved {kind}_root for source '{source}' is not a directory: {path}")


def _load_artifact(path: str | Path) -> dict:
    import torch

    from src.clip_1nn import CLIP_1NN_ARCH

    artifact = torch.load(path, map_location="cpu")
    if artifact.get("arch") != CLIP_1NN_ARCH:
        raise ValueError(f"checkpoint arch mismatch: expected '{CLIP_1NN_ARCH}', got '{artifact.get('arch')}'")
    if not isinstance(artifact.get("real_features"), torch.Tensor):
        raise ValueError("checkpoint missing tensor 'real_features'")
    if not isinstance(artifact.get("fake_features"), torch.Tensor):
        raise ValueError("checkpoint missing tensor 'fake_features'")
    return artifact


def main() -> None:
    args = parse_args()

    import numpy as np
    import torch

    from src.augment import build_eval_transform
    from src.clip_1nn import (
        CLIP_1NN_ARCH,
        CLIP_BACKBONE,
        CLIP_NORM,
        FIXED_THRESHOLD,
        SCORE_MODE,
        Clip1NNFeatureExtractor,
        extract_features,
        score_1nn_features,
    )
    from src.datasets import RealFakeFolderDataset
    from src.official_data import resolve_official_eval_pair
    from src.utils import (
        append_csv,
        binary_metrics,
        ensure_dir,
        find_best_threshold,
        plot_bar,
        plot_pr_curve,
        plot_roc_curve,
        save_json,
        setup_logger,
        summarize_rows,
    )

    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        SummaryWriter = None

    out_dir = ensure_dir(args.output_dir)
    plots_dir = ensure_dir(out_dir / "plots")
    tb_dir = ensure_dir(out_dir / "tensorboard")
    logger = setup_logger(out_dir / "eval.log", name=f"eval_{Path(args.output_dir).name}")

    writer = None
    if not args.no_tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(tb_dir))
    elif not args.no_tensorboard and SummaryWriter is None:
        logger.warning("tensorboard writer unavailable; install tensorboard to enable it")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)
    logger.info("tensorboard_dir=%s", tb_dir)

    artifact = _load_artifact(args.checkpoint)
    real_bank = artifact["real_features"].detach().cpu().float().contiguous()
    fake_bank = artifact["fake_features"].detach().cpu().float().contiguous()
    fixed_threshold = float(artifact.get("fixed_threshold", FIXED_THRESHOLD))
    score_mode = str(artifact.get("score_mode", SCORE_MODE))
    logger.info(
        "checkpoint=%s backbone=%s train_real=%s train_fake=%s score_mode=%s",
        args.checkpoint,
        artifact.get("backbone", CLIP_BACKBONE),
        real_bank.size(0),
        fake_bank.size(0),
        score_mode,
    )

    model = Clip1NNFeatureExtractor().to(device)
    transform = build_eval_transform(norm=artifact.get("norm", CLIP_NORM))

    csv_path = out_dir / "per_source_results.csv"
    all_rows: list[dict] = []

    for source in args.sources:
        pair = resolve_official_eval_pair(args.data_root, source)
        real_root = Path(pair["real_root"])
        fake_root = Path(pair["fake_root"])
        logger.info(
            "source=%s layout=%s real_root=%s fake_root=%s",
            source,
            pair.get("layout", "default"),
            real_root,
            fake_root,
        )
        _require_dataset_root(real_root, source, "real")
        _require_dataset_root(fake_root, source, "fake")

        ds = RealFakeFolderDataset(
            real_root=real_root,
            fake_root=fake_root,
            source=source,
            transform=transform,
        )
        loader = build_loader(ds, args.batch_size, args.num_workers, device)
        feature_batch = extract_features(model, loader, device, args.amp, desc=f"extract_{source}")
        scores = score_1nn_features(
            feature_batch.features,
            real_bank,
            fake_bank,
            device=device,
            query_batch_size=args.batch_size,
            bank_chunk_size=args.bank_chunk_size,
            desc=f"score_{source}",
        )

        y_true = feature_batch.labels.numpy().astype(np.int64)
        y_prob = scores.numpy().astype(np.float32)
        best_t = find_best_threshold(y_true, y_prob)
        fixed = binary_metrics(y_true, y_prob, threshold=fixed_threshold)
        best = binary_metrics(y_true, y_prob, threshold=best_t)
        row = {
            "source": source,
            "n_samples": len(ds),
            **fixed,
            "fixed_threshold": fixed_threshold,
            "best_threshold": best_t,
            "best_acc": best["acc"],
            "best_real_acc": best["real_acc"],
            "best_fake_acc": best["fake_acc"],
        }
        append_csv(csv_path, row)
        all_rows.append(row)

        plot_pr_curve(y_true, y_prob, plots_dir / f"pr_{source}.png", title=f"PR curve - {source}")
        plot_roc_curve(y_true, y_prob, plots_dir / f"roc_{source}.png", title=f"ROC curve - {source}")
        if writer is not None:
            writer.add_scalar("per_source/ap", float(row["ap"]), len(all_rows))
            writer.add_scalar("per_source/roc_auc", float(row["roc_auc"]), len(all_rows))
            writer.add_scalar("per_source/acc", float(row["acc"]), len(all_rows))
            writer.add_scalar("per_source/best_acc", float(row["best_acc"]), len(all_rows))
            writer.add_text(f"per_source/{source}", str(row))
        logger.info(
            "source=%s n=%s ap=%.6f roc_auc=%.6f acc=%.6f best_acc=%.6f best_t=%.3f",
            source,
            len(ds),
            row["ap"],
            row["roc_auc"],
            row["acc"],
            row["best_acc"],
            row["best_threshold"],
        )

    summary = summarize_rows(all_rows, ["ap", "roc_auc", "acc", "best_acc", "real_acc", "fake_acc"])
    summary.update(
        {
            "checkpoint": args.checkpoint,
            "arch": artifact.get("arch", CLIP_1NN_ARCH),
            "backbone": artifact.get("backbone", CLIP_BACKBONE),
            "sources": args.sources,
            "fixed_threshold": fixed_threshold,
            "score_mode": score_mode,
        }
    )
    append_csv(out_dir / "summary.csv", summary)
    save_json(out_dir / "summary.json", summary)

    plot_bar(
        [r["source"] for r in all_rows],
        [float(r["ap"]) for r in all_rows],
        plots_dir / "ap_by_source.png",
        "AP by source",
        "AP",
    )
    plot_bar(
        [r["source"] for r in all_rows],
        [float(r["acc"]) for r in all_rows],
        plots_dir / "acc_by_source.png",
        "Acc@0.5 by source",
        "Accuracy",
    )
    plot_bar(
        [r["source"] for r in all_rows],
        [float(r["best_acc"]) for r in all_rows],
        plots_dir / "best_acc_by_source.png",
        "Best-threshold Acc by source",
        "Accuracy",
    )

    if writer is not None:
        for key in ["ap", "roc_auc", "acc", "best_acc", "real_acc", "fake_acc"]:
            summary_key = f"mean_{key}"
            if summary_key in summary:
                writer.add_scalar(f"summary/{key}", float(summary[summary_key]), 0)
        writer.add_text("summary/json", str(summary))
        writer.close()

    logger.info("summary=%s", summary)


if __name__ == "__main__":
    main()
