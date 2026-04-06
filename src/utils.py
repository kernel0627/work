from __future__ import annotations

import csv
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_csv(path: str | Path, row: Dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_csv_rows(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def setup_logger(log_path: str | Path, name: str = 'ufd') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)
    acc = float(accuracy_score(y_true, y_pred))
    ap = float(average_precision_score(y_true, y_prob))
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float('nan')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    real_acc = float(tn / max(tn + fp, 1))
    fake_acc = float(tp / max(tp + fn, 1))
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    return {
        'acc': acc,
        'ap': ap,
        'roc_auc': roc_auc,
        'real_acc': real_acc,
        'fake_acc': fake_acc,
        'precision': precision,
        'recall': recall,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_t = 0.5
    best_acc = -1.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(np.int64)
        acc = accuracy_score(y_true, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    best_metric: float,
    extra: Optional[dict] = None,
) -> None:
    ckpt = {
        'model': model.state_dict(),
        'epoch': epoch,
        'best_metric': best_metric,
    }
    if optimizer is not None:
        ckpt['optimizer'] = optimizer.state_dict()
    if scaler is not None:
        ckpt['scaler'] = scaler.state_dict()
    if extra is not None:
        ckpt['extra'] = extra
    torch.save(ckpt, path)


def load_checkpoint(path: str | Path, model=None, optimizer=None, scaler=None, map_location='cpu') -> dict:
    ckpt = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(ckpt['model'], strict=True)
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scaler is not None and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    return ckpt


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def env_info() -> dict:
    info = {
        'python_version': sys.version,
        'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cudnn_enabled': torch.backends.cudnn.enabled,
    }
    if torch.cuda.is_available():
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['gpu_count'] = torch.cuda.device_count()
    return info


def gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.max_memory_allocated() / (1024 ** 2))


def reset_gpu_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def plot_history(csv_path: str | Path, out_dir: str | Path) -> None:
    rows = read_csv_rows(csv_path)
    if not rows:
        return
    out_dir = ensure_dir(out_dir)

    def arr(key: str) -> list[float]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[key]))
            except Exception:
                vals.append(float('nan'))
        return vals

    epochs = arr('epoch')
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, arr('train_loss'), label='train_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Train loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'train_loss.png', dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, arr('ap'), label='val_ap')
    plt.plot(epochs, arr('acc'), label='val_acc@0.5')
    plt.plot(epochs, arr('best_acc'), label='val_best_acc')
    if 'roc_auc' in rows[0]:
        plt.plot(epochs, arr('roc_auc'), label='val_roc_auc')
    plt.xlabel('epoch')
    plt.ylabel('metric')
    plt.title('Validation metrics')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(out_dir) / 'val_metrics.png', dpi=150)
    plt.close()


def plot_bar(names: list[str], values: list[float], out_path: str | Path, title: str, ylabel: str) -> None:
    plt.figure(figsize=(max(8, len(names) * 1.2), 5))
    plt.bar(names, values)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha='right')
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: str | Path, title: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: str | Path, title: str) -> None:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except ValueError:
        return
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def summarize_rows(rows: Iterable[dict], keys: list[str]) -> dict:
    rows = list(rows)
    out: dict[str, float | int] = {'num_sources': len(rows)}
    for key in keys:
        vals = []
        for row in rows:
            v = row.get(key)
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if not math.isnan(fv):
                vals.append(fv)
        out[f'mean_{key}'] = float(np.mean(vals)) if vals else float('nan')
    return out
