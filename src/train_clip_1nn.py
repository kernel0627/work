from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_BANK_CHUNK_SIZE = 8192


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-root", type=str, required=True)
    p.add_argument("--val-root", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--output-dir", type=str, default="./runs/clip_vitl14_1nn_progan")
    p.add_argument("--seed", type=int, default=42)
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
        split_feature_bank,
    )
    from src.datasets import RealFakeFolderDataset
    from src.utils import (
        append_csv,
        binary_metrics,
        ensure_dir,
        env_info,
        find_best_threshold,
        plot_pr_curve,
        plot_roc_curve,
        save_json,
        setup_logger,
    )

    try:
        from torch.utils.tensorboard import SummaryWriter
    except Exception:
        SummaryWriter = None

    out_dir = ensure_dir(args.output_dir)
    logs_dir = ensure_dir(out_dir / "logs")
    ckpts_dir = ensure_dir(out_dir / "ckpts")
    tb_dir = ensure_dir(out_dir / "tensorboard")
    plots_dir = ensure_dir(out_dir / "plots")
    reports_dir = ensure_dir(out_dir / "reports")
    logger = setup_logger(logs_dir / "console.log", name=f"train_{Path(args.output_dir).name}")

    save_json(reports_dir / "args.json", vars(args))
    save_json(reports_dir / "env.json", env_info())

    writer = None
    if not args.no_tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(tb_dir))
    elif not args.no_tensorboard and SummaryWriter is None:
        logger.warning("tensorboard writer unavailable; install tensorboard to enable it")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)
    logger.info("tensorboard_dir=%s", tb_dir)

    transform = build_eval_transform(norm=CLIP_NORM)
    train_ds = RealFakeFolderDataset(
        real_root=args.train_root,
        fake_root=args.train_root,
        source="progan_train",
        transform=transform,
        seed=args.seed,
    )
    val_ds = RealFakeFolderDataset(
        real_root=args.val_root,
        fake_root=args.val_root,
        source="progan_val",
        transform=transform,
        seed=args.seed,
    )

    train_loader = build_loader(train_ds, args.batch_size, args.num_workers, device)
    val_loader = build_loader(val_ds, args.batch_size, args.num_workers, device)

    model = Clip1NNFeatureExtractor().to(device)

    logger.info("extracting train bank from %s", args.train_root)
    train_batch = extract_features(model, train_loader, device, args.amp, desc="extract_train")
    real_features, fake_features = split_feature_bank(train_batch.features, train_batch.labels)

    logger.info("extracting val features from %s", args.val_root)
    val_batch = extract_features(model, val_loader, device, args.amp, desc="extract_val")

    logger.info(
        "scoring val set with 1-NN: train_real=%s train_fake=%s val=%s",
        real_features.size(0),
        fake_features.size(0),
        val_batch.features.size(0),
    )
    val_scores = score_1nn_features(
        val_batch.features,
        real_features,
        fake_features,
        device=device,
        query_batch_size=args.batch_size,
        bank_chunk_size=args.bank_chunk_size,
        desc="score_val",
    )

    y_true = val_batch.labels.numpy().astype(np.int64)
    y_prob = val_scores.numpy().astype(np.float32)
    best_t = find_best_threshold(y_true, y_prob)
    fixed = binary_metrics(y_true, y_prob, threshold=FIXED_THRESHOLD)
    best = binary_metrics(y_true, y_prob, threshold=best_t)
    val_metrics = {
        **fixed,
        "best_threshold": best_t,
        "best_acc": best["acc"],
        "best_real_acc": best["real_acc"],
        "best_fake_acc": best["fake_acc"],
    }

    val_row = {"split": "val", "n_samples": len(val_ds), **val_metrics}
    append_csv(logs_dir / "val_metrics.csv", val_row)
    plot_pr_curve(y_true, y_prob, plots_dir / "pr_val.png", title="PR curve - val")
    plot_roc_curve(y_true, y_prob, plots_dir / "roc_val.png", title="ROC curve - val")

    meta = {
        "arch": CLIP_1NN_ARCH,
        "backbone": CLIP_BACKBONE,
        "norm": CLIP_NORM,
        "score_mode": SCORE_MODE,
        "fixed_threshold": FIXED_THRESHOLD,
        "feature_dim": int(real_features.shape[1]),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "train_real": int(real_features.size(0)),
        "train_fake": int(fake_features.size(0)),
    }
    save_json(reports_dir / "meta.json", meta)
    save_json(reports_dir / "best_metrics.json", {"split": "val", **val_metrics})

    if writer is not None:
        writer.add_text("run/output_dir", str(out_dir))
        writer.add_text("run/arch", CLIP_1NN_ARCH)
        writer.add_text("run/backbone", CLIP_BACKBONE)
        writer.add_text("run/train_root", str(args.train_root))
        writer.add_text("run/val_root", str(args.val_root))
        writer.add_scalar("bank/train_real", int(real_features.size(0)), 0)
        writer.add_scalar("bank/train_fake", int(fake_features.size(0)), 0)
        writer.add_scalar("val/ap", float(val_metrics["ap"]), 0)
        writer.add_scalar("val/roc_auc", float(val_metrics["roc_auc"]), 0)
        writer.add_scalar("val/acc", float(val_metrics["acc"]), 0)
        writer.add_scalar("val/best_acc", float(val_metrics["best_acc"]), 0)
        writer.add_scalar("val/real_acc", float(val_metrics["real_acc"]), 0)
        writer.add_scalar("val/fake_acc", float(val_metrics["fake_acc"]), 0)
        writer.add_scalar("val/best_threshold", float(val_metrics["best_threshold"]), 0)
        writer.add_text("val/metrics", str(val_metrics))

    artifact = {
        "arch": CLIP_1NN_ARCH,
        "backbone": CLIP_BACKBONE,
        "norm": CLIP_NORM,
        "fixed_threshold": FIXED_THRESHOLD,
        "score_mode": SCORE_MODE,
        "train_root": args.train_root,
        "val_root": args.val_root,
        "feature_dim": int(real_features.shape[1]),
        "real_features": real_features.contiguous().cpu(),
        "fake_features": fake_features.contiguous().cpu(),
        "val_metrics": val_metrics,
        "meta": meta,
        "args": vars(args),
    }
    checkpoint_path = ckpts_dir / "best.pt"
    torch.save(artifact, checkpoint_path)

    final_report = {
        "best_checkpoint": str(checkpoint_path),
        "val_metrics": val_metrics,
        "train_real": int(real_features.size(0)),
        "train_fake": int(fake_features.size(0)),
        "score_mode": SCORE_MODE,
        "fixed_threshold": FIXED_THRESHOLD,
    }
    save_json(reports_dir / "final_report.json", final_report)

    logger.info(
        "saved 1-NN artifact: checkpoint=%s ap=%.6f roc_auc=%.6f acc=%.6f best_acc=%.6f best_t=%.3f",
        checkpoint_path,
        val_metrics["ap"],
        val_metrics["roc_auc"],
        val_metrics["acc"],
        val_metrics["best_acc"],
        val_metrics["best_threshold"],
    )

    if writer is not None:
        writer.add_text("final_report/json", str(final_report))
        writer.close()


if __name__ == "__main__":
    main()
