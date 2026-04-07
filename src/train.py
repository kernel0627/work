from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from .augment import AugmentConfig, build_eval_transform, build_train_transform
from .datasets import RealFakeFolderDataset
from .models import build_model
from .utils import (
    ModelEMA,
    append_csv,
    binary_metrics,
    count_trainable_params,
    ensure_dir,
    env_info,
    find_best_threshold,
    gpu_mem_mb,
    load_checkpoint,
    plot_history,
    reset_gpu_peak_memory,
    save_checkpoint,
    save_json,
    set_seed,
    setup_logger,
    summarize_rows,
)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, amp: bool) -> Dict[str, float]:
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []

    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = torch.as_tensor(batch["label"], device=device, dtype=torch.float32)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            logits = model(images)

        probs = torch.sigmoid(logits)
        y_true.extend(labels.detach().cpu().numpy().astype(np.int64).tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())

    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_prob_np = np.asarray(y_prob, dtype=np.float32)

    best_t = find_best_threshold(y_true_np, y_prob_np)
    fixed = binary_metrics(y_true_np, y_prob_np, threshold=0.5)
    best = binary_metrics(y_true_np, y_prob_np, threshold=best_t)

    return {
        **fixed,
        "best_threshold": best_t,
        "best_acc": best["acc"],
        "best_real_acc": best["real_acc"],
        "best_fake_acc": best["fake_acc"],
    }


def parse_csv_arg(x: str | None) -> List[str] | None:
    if not x:
        return None
    items = [z.strip() for z in x.split(",") if z.strip()]
    return items or None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train-root", type=str, required=True)
    p.add_argument("--val-root", type=str, required=True)
    p.add_argument("--train-categories", type=str, default=None)
    p.add_argument("--val-categories", type=str, default=None)
    p.add_argument("--limit-real", type=int, default=None)
    p.add_argument("--limit-fake", type=int, default=None)

    p.add_argument("--arch", type=str, choices=["clip_linear", "resnet50"], required=True)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--patience", type=int, default=5)

    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--tb-log-steps", type=int, default=50)

    p.add_argument("--ema", action="store_true")
    p.add_argument("--ema-decay", type=float, default=0.9998)
    p.add_argument("--ema-cpu", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

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
    logger.info(f"device={device}")

    built = build_model(args.arch)
    model = built.model.to(device)

    aug_cfg = AugmentConfig(norm=built.norm)
    train_tf = build_train_transform(aug_cfg)
    eval_tf = build_eval_transform(norm=built.norm)

    train_ds = RealFakeFolderDataset(
        real_root=args.train_root,
        fake_root=args.train_root,
        source="progan_train",
        transform=train_tf,
        categories=parse_csv_arg(args.train_categories),
        limit_real=args.limit_real,
        limit_fake=args.limit_fake,
        seed=args.seed,
    )
    val_ds = RealFakeFolderDataset(
        real_root=args.val_root,
        fake_root=args.val_root,
        source="progan_val",
        transform=eval_tf,
        categories=parse_csv_arg(args.val_categories),
        limit_real=args.limit_real,
        limit_fake=args.limit_fake,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    criterion = nn.BCEWithLogitsLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    ema = None
    if args.ema:
        ema_device = "cpu" if args.ema_cpu else None
        ema = ModelEMA(model, decay=args.ema_decay, device=ema_device)

    start_epoch = 0
    best_ap = -1.0
    epochs_no_improve = 0

    if args.resume:
        ckpt = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            map_location=device,
        )
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_ap = float(ckpt.get("best_metric", -1.0))
        extra = ckpt.get("extra", {})
        epochs_no_improve = int(extra.get("epochs_no_improve", 0))
        logger.info(
            f"resumed from {args.resume}, start_epoch={start_epoch}, best_ap={best_ap:.6f}"
        )

    meta = {
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "trainable_params": count_trainable_params(model),
        "all_params": int(sum(p.numel() for p in model.parameters())),
        "train_categories": parse_csv_arg(args.train_categories),
        "val_categories": parse_csv_arg(args.val_categories),
        "limit_real": args.limit_real,
        "limit_fake": args.limit_fake,
    }
    save_json(reports_dir / "meta.json", meta)
    logger.info(f"meta={meta}")
    logger.info(f"tensorboard_dir={tb_dir}")

    if writer is not None:
        writer.add_text("run/output_dir", str(out_dir))
        writer.add_text("run/arch", args.arch)
        writer.add_text("run/train_root", str(args.train_root))
        writer.add_text("run/val_root", str(args.val_root))

    metrics_csv = logs_dir / "train_history.csv"
    global_step = start_epoch * max(len(train_loader), 1)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        reset_gpu_peak_memory()

        running_loss = 0.0
        n_seen = 0
        epoch_start = time.time()
        train_compute_start = time.time()

        prog = tqdm(train_loader, desc=f"train epoch {epoch}")
        for batch in prog:
            images = batch["image"].to(device, non_blocking=True)
            labels = torch.as_tensor(batch["label"], device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model)

            bs = images.size(0)
            running_loss += float(loss.item()) * bs
            n_seen += bs
            global_step += 1

            prog.set_postfix(loss=f"{running_loss / max(n_seen, 1):.4f}")

            if writer is not None and args.tb_log_steps > 0 and global_step % args.tb_log_steps == 0:
                writer.add_scalar("train/batch_loss", float(loss.item()), global_step)
                writer.add_scalar("train/running_loss", running_loss / max(n_seen, 1), global_step)
                writer.add_scalar("train/lr_step", float(optimizer.param_groups[0]["lr"]), global_step)

        train_seconds = time.time() - train_compute_start
        train_loss = running_loss / max(n_seen, 1)

        eval_start = time.time()
        eval_model = ema.ema_model if ema is not None else model
        val_metrics = evaluate(eval_model, val_loader, device, args.amp)
        eval_seconds = time.time() - eval_start

        total_seconds = time.time() - epoch_start
        lr = float(optimizer.param_groups[0]["lr"])
        gpu_mb = gpu_mem_mb()

        row = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_loss,
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
            "gpu_mem_mb": gpu_mb,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "epoch_seconds": total_seconds,
        }
        append_csv(metrics_csv, row)
        plot_history(metrics_csv, plots_dir)

        if writer is not None:
            writer.add_scalar("epoch/train_loss", train_loss, epoch)
            writer.add_scalar("epoch/lr", lr, epoch)
            writer.add_scalar("epoch/gpu_mem_mb", gpu_mb, epoch)
            writer.add_scalar("val/ap", row["ap"], epoch)
            writer.add_scalar("val/roc_auc", row["roc_auc"], epoch)
            writer.add_scalar("val/acc", row["acc"], epoch)
            writer.add_scalar("val/best_acc", row["best_acc"], epoch)
            writer.add_scalar("val/real_acc", row["real_acc"], epoch)
            writer.add_scalar("val/fake_acc", row["fake_acc"], epoch)
            writer.add_scalar("time/train_seconds", train_seconds, epoch)
            writer.add_scalar("time/eval_seconds", eval_seconds, epoch)
            writer.add_scalar("time/epoch_seconds", total_seconds, epoch)

        logger.info(
            "epoch=%s train_loss=%.6f ap=%.6f roc_auc=%.6f acc=%.6f best_acc=%.6f best_t=%.3f gpu_mem_mb=%.1f epoch_s=%.1f",
            epoch,
            train_loss,
            row["ap"],
            row["roc_auc"],
            row["acc"],
            row["best_acc"],
            row["best_threshold"],
            gpu_mb,
            total_seconds,
        )

        extra = {
            "val_metrics": val_metrics,
            "arch": args.arch,
            "epochs_no_improve": epochs_no_improve,
        }

        save_checkpoint(
            ckpts_dir / "last.pt",
            model,
            optimizer,
            scaler,
            epoch,
            best_ap,
            extra=extra,
            ema_state=ema.state_dict() if ema is not None else None,
        )

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                ckpts_dir / f"epoch_{epoch}.pt",
                model,
                optimizer,
                scaler,
                epoch,
                best_ap,
                extra=extra,
                ema_state=ema.state_dict() if ema is not None else None,
            )

        if val_metrics["ap"] > best_ap:
            best_ap = float(val_metrics["ap"])
            epochs_no_improve = 0
            extra["epochs_no_improve"] = epochs_no_improve

            save_checkpoint(
                ckpts_dir / "best.pt",
                model,
                optimizer,
                scaler,
                epoch,
                best_ap,
                extra=extra,
                ema_state=ema.state_dict() if ema is not None else None,
            )

            save_json(reports_dir / "best_metrics.json", {"epoch": epoch, **val_metrics})
            logger.info("new best: epoch=%s ap=%.6f", epoch, best_ap)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info("early stop at epoch=%s due to patience=%s", epoch, args.patience)
                break

    with open(metrics_csv, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    summary = summarize_rows(rows, ["train_loss", "ap", "roc_auc", "acc", "best_acc"])
    summary.update(
        {
            "best_ap": best_ap,
            "best_checkpoint": str(ckpts_dir / "best.pt"),
            "last_checkpoint": str(ckpts_dir / "last.pt"),
            "num_epochs_recorded": len(rows),
        }
    )

    save_json(reports_dir / "final_report.json", summary)
    logger.info(f"final_report={summary}")

    if writer is not None:
        writer.add_text("final_report/json", str(summary))
        writer.close()


if __name__ == "__main__":
    main()