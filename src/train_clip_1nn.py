from __future__ import annotations

import argparse
import time
from pathlib import Path

DEFAULT_BANK_CHUNK_SIZE = 8192


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-root", type=str, required=True)
    p.add_argument("--val-root", type=str, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--output-dir", type=str, default="./runs/clip_vitl14_1nn_progan")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--patience", type=int, default=5)
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

    from src.augment import AugmentConfig, build_eval_transform, build_train_transform
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
        plot_history,
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

    train_transform = build_train_transform(AugmentConfig(norm=CLIP_NORM))
    eval_transform = build_eval_transform(norm=CLIP_NORM)
    train_ds = RealFakeFolderDataset(
        real_root=args.train_root,
        fake_root=args.train_root,
        source="progan_train",
        transform=train_transform,
        seed=args.seed,
    )
    val_ds = RealFakeFolderDataset(
        real_root=args.val_root,
        fake_root=args.val_root,
        source="progan_val",
        transform=eval_transform,
        seed=args.seed,
    )

    train_loader = build_loader(train_ds, args.batch_size, args.num_workers, device)
    val_loader = build_loader(val_ds, args.batch_size, args.num_workers, device)

    model = Clip1NNFeatureExtractor().to(device)

    meta = {
        "arch": CLIP_1NN_ARCH,
        "backbone": CLIP_BACKBONE,
        "norm": CLIP_NORM,
        "score_mode": SCORE_MODE,
        "fixed_threshold": FIXED_THRESHOLD,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "train_root": args.train_root,
        "val_root": args.val_root,
        "epochs": args.epochs,
        "patience": args.patience,
    }
    save_json(reports_dir / "meta.json", meta)

    if writer is not None:
        writer.add_text("run/output_dir", str(out_dir))
        writer.add_text("run/arch", CLIP_1NN_ARCH)
        writer.add_text("run/backbone", CLIP_BACKBONE)
        writer.add_text("run/train_root", str(args.train_root))
        writer.add_text("run/val_root", str(args.val_root))

    logger.info("extracting fixed val features from %s", args.val_root)
    val_batch = extract_features(model, val_loader, device, args.amp, desc="extract_val")
    y_true = val_batch.labels.numpy().astype(np.int64)

    metrics_csv = logs_dir / "train_history.csv"
    best_ap = -1.0
    best_epoch = -1
    epochs_no_improve = 0
    completed_epochs = 0
    final_report: dict[str, object] = {}

    def build_artifact(
        epoch: int,
        real_features: torch.Tensor,
        fake_features: torch.Tensor,
        val_metrics: dict,
        best_metric: float,
        feature_dim: int,
        train_real: int,
        train_fake: int,
        extra: dict,
    ) -> dict:
        return {
            "arch": CLIP_1NN_ARCH,
            "backbone": CLIP_BACKBONE,
            "norm": CLIP_NORM,
            "fixed_threshold": FIXED_THRESHOLD,
            "score_mode": SCORE_MODE,
            "train_root": args.train_root,
            "val_root": args.val_root,
            "feature_dim": feature_dim,
            "real_features": real_features.contiguous().cpu(),
            "fake_features": fake_features.contiguous().cpu(),
            "val_metrics": val_metrics,
            "meta": {
                **meta,
                "feature_dim": feature_dim,
                "train_real": train_real,
                "train_fake": train_fake,
            },
            "args": vars(args),
            "epoch": epoch,
            "best_metric": best_metric,
            "extra": extra,
        }

    for epoch in range(args.epochs):
        epoch_start = time.time()
        completed_epochs = epoch + 1

        logger.info("epoch=%s extracting train bank from %s", epoch, args.train_root)
        bank_start = time.time()
        train_batch = extract_features(model, train_loader, device, args.amp, desc=f"extract_train_e{epoch}")
        real_features, fake_features = split_feature_bank(train_batch.features, train_batch.labels)
        bank_seconds = time.time() - bank_start

        eval_start = time.time()
        val_scores = score_1nn_features(
            val_batch.features,
            real_features,
            fake_features,
            device=device,
            query_batch_size=args.batch_size,
            bank_chunk_size=args.bank_chunk_size,
            desc=f"score_val_e{epoch}",
        )
        eval_seconds = time.time() - eval_start

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

        total_seconds = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": float("nan"),
            "ap": val_metrics["ap"],
            "roc_auc": val_metrics["roc_auc"],
            "acc": val_metrics["acc"],
            "real_acc": val_metrics["real_acc"],
            "fake_acc": val_metrics["fake_acc"],
            "best_threshold": val_metrics["best_threshold"],
            "best_acc": val_metrics["best_acc"],
            "best_real_acc": val_metrics["best_real_acc"],
            "best_fake_acc": val_metrics["best_fake_acc"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "bank_real": int(real_features.size(0)),
            "bank_fake": int(fake_features.size(0)),
            "bank_seconds": bank_seconds,
            "eval_seconds": eval_seconds,
            "epoch_seconds": total_seconds,
        }
        append_csv(metrics_csv, row)
        plot_history(metrics_csv, plots_dir)

        if writer is not None:
            writer.add_scalar("bank/train_real", int(real_features.size(0)), epoch)
            writer.add_scalar("bank/train_fake", int(fake_features.size(0)), epoch)
            writer.add_scalar("time/bank_seconds", bank_seconds, epoch)
            writer.add_scalar("time/eval_seconds", eval_seconds, epoch)
            writer.add_scalar("time/epoch_seconds", total_seconds, epoch)
            writer.add_scalar("val/ap", float(val_metrics["ap"]), epoch)
            writer.add_scalar("val/roc_auc", float(val_metrics["roc_auc"]), epoch)
            writer.add_scalar("val/acc", float(val_metrics["acc"]), epoch)
            writer.add_scalar("val/best_acc", float(val_metrics["best_acc"]), epoch)
            writer.add_scalar("val/real_acc", float(val_metrics["real_acc"]), epoch)
            writer.add_scalar("val/fake_acc", float(val_metrics["fake_acc"]), epoch)
            writer.add_scalar("val/best_threshold", float(val_metrics["best_threshold"]), epoch)
            writer.add_text(f"val/metrics_epoch_{epoch}", str(val_metrics))

        feature_dim = int(real_features.shape[1])
        improved = val_metrics["ap"] > best_ap
        if improved:
            best_ap = float(val_metrics["ap"])
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        extra = {
            "epochs_no_improve": epochs_no_improve,
            "val_metrics": val_metrics,
            "arch": CLIP_1NN_ARCH,
            "best_epoch": best_epoch,
        }
        artifact = build_artifact(
            epoch=epoch,
            real_features=real_features,
            fake_features=fake_features,
            val_metrics=val_metrics,
            best_metric=best_ap,
            feature_dim=feature_dim,
            train_real=int(real_features.size(0)),
            train_fake=int(fake_features.size(0)),
            extra=extra,
        )
        torch.save(artifact, ckpts_dir / "last.pt")

        if (epoch + 1) % args.save_every == 0:
            torch.save(artifact, ckpts_dir / f"epoch_{epoch}.pt")

        if improved:
            torch.save(artifact, ckpts_dir / "best.pt")
            save_json(reports_dir / "best_metrics.json", {"epoch": epoch, **val_metrics})
            plot_pr_curve(y_true, y_prob, plots_dir / "pr_val.png", title=f"PR curve - val (best epoch {epoch})")
            plot_roc_curve(y_true, y_prob, plots_dir / "roc_val.png", title=f"ROC curve - val (best epoch {epoch})")
            logger.info(
                "new best: epoch=%s ap=%.6f acc=%.6f best_acc=%.6f best_t=%.3f",
                epoch,
                best_ap,
                val_metrics["acc"],
                val_metrics["best_acc"],
                val_metrics["best_threshold"],
            )

        logger.info(
            "epoch=%s ap=%.6f roc_auc=%.6f acc=%.6f best_acc=%.6f best_t=%.3f bank_real=%s bank_fake=%s epoch_s=%.1f",
            epoch,
            val_metrics["ap"],
            val_metrics["roc_auc"],
            val_metrics["acc"],
            val_metrics["best_acc"],
            val_metrics["best_threshold"],
            int(real_features.size(0)),
            int(fake_features.size(0)),
            total_seconds,
        )

        if not improved and epochs_no_improve >= args.patience:
            logger.info("early stop at epoch=%s due to patience=%s", epoch, args.patience)
            break

    best_checkpoint = ckpts_dir / "best.pt"
    final_report = {
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(ckpts_dir / "last.pt"),
        "best_epoch": best_epoch,
        "best_ap": best_ap,
        "num_epochs": args.epochs,
        "num_epochs_recorded": completed_epochs,
        "score_mode": SCORE_MODE,
        "fixed_threshold": FIXED_THRESHOLD,
    }
    save_json(reports_dir / "final_report.json", final_report)

    if writer is not None:
        writer.add_text("final_report/json", str(final_report))
        writer.close()


if __name__ == "__main__":
    main()
