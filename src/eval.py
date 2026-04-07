from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from .augment import build_eval_transform
from .datasets import RealFakeFolderDataset
from .models import build_model
from .official_data import DEFAULT_REPRO_SOURCES, resolve_official_eval_pair
from .utils import (
    append_csv,
    ensure_dir,
    find_best_threshold,
    load_checkpoint,
    plot_bar,
    plot_pr_curve,
    plot_roc_curve,
    save_json,
    setup_logger,
    summarize_rows,
)
from .utils import binary_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=str, default='./datasets')
    p.add_argument('--sources', nargs='+', default=DEFAULT_REPRO_SOURCES)
    p.add_argument('--arch', type=str, choices=['clip_linear', 'resnet50'], required=True)
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--output-dir', type=str, default='./eval_results')
    p.add_argument('--no-tensorboard', action='store_true')
    return p.parse_args()


@torch.no_grad()
def evaluate_one(model, loader, device: torch.device, amp: bool) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    y_true: List[int] = []
    y_prob: List[float] = []
    for batch in tqdm(loader, desc='eval', leave=False):
        images = batch['image'].to(device, non_blocking=True)
        labels = torch.as_tensor(batch['label'], device=device, dtype=torch.float32)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == 'cuda'):
            logits = model(images)
        probs = torch.sigmoid(logits)
        y_true.extend(labels.detach().cpu().numpy().astype(np.int64).tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_prob_np = np.asarray(y_prob, dtype=np.float32)
    best_t = find_best_threshold(y_true_np, y_prob_np)
    fixed = binary_metrics(y_true_np, y_prob_np, threshold=0.5)
    best = binary_metrics(y_true_np, y_prob_np, threshold=best_t)
    metrics = {
        **fixed,
        'best_threshold': best_t,
        'best_acc': best['acc'],
        'best_real_acc': best['real_acc'],
        'best_fake_acc': best['fake_acc'],
    }
    return metrics, y_true_np, y_prob_np


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    plots_dir = ensure_dir(out_dir / 'plots')
    tb_dir = ensure_dir(out_dir / 'tensorboard')
    logger = setup_logger(out_dir / 'eval.log', name=f'eval_{Path(args.output_dir).name}')
    writer = None
    if not args.no_tensorboard and SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(tb_dir))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    built = build_model(args.arch)
    model = built.model.to(device)
    ckpt = load_checkpoint(args.checkpoint, model=model, optimizer=None, scaler=None, map_location=device)
    tf = build_eval_transform(norm=built.norm)

    csv_path = out_dir / 'per_source_results.csv'
    all_rows = []
    for source in args.sources:
        pair = resolve_official_eval_pair(args.data_root, source)
        ds = RealFakeFolderDataset(
            real_root=pair['real_root'],
            fake_root=pair['fake_root'],
            source=source,
            transform=tf,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(args.num_workers > 0),
        )
        metrics, y_true, y_prob = evaluate_one(model, loader, device, args.amp)
        row = {'source': source, 'n_samples': len(ds), **metrics}
        append_csv(csv_path, row)
        all_rows.append(row)
        plot_pr_curve(y_true, y_prob, plots_dir / f'pr_{source}.png', title=f'PR curve - {source}')
        plot_roc_curve(y_true, y_prob, plots_dir / f'roc_{source}.png', title=f'ROC curve - {source}')
        if writer is not None:
            writer.add_scalar('per_source/ap', row['ap'], len(all_rows))
            writer.add_scalar('per_source/roc_auc', row['roc_auc'], len(all_rows))
            writer.add_scalar('per_source/acc', row['acc'], len(all_rows))
            writer.add_scalar('per_source/best_acc', row['best_acc'], len(all_rows))
            writer.add_text(f'per_source/{source}', str(row))
        logger.info(
            'source=%s n=%s ap=%.6f roc_auc=%.6f acc=%.6f best_acc=%.6f best_t=%.3f',
            source,
            len(ds),
            row['ap'],
            row['roc_auc'],
            row['acc'],
            row['best_acc'],
            row['best_threshold'],
        )

    summary = summarize_rows(all_rows, ['ap', 'roc_auc', 'acc', 'best_acc', 'real_acc', 'fake_acc'])
    summary.update({
        'checkpoint': args.checkpoint,
        'arch': args.arch,
        'sources': args.sources,
        'ckpt_epoch': ckpt.get('epoch'),
    })
    append_csv(out_dir / 'summary.csv', summary)
    save_json(out_dir / 'summary.json', summary)

    plot_bar([r['source'] for r in all_rows], [float(r['ap']) for r in all_rows], plots_dir / 'ap_by_source.png', 'AP by source', 'AP')
    plot_bar([r['source'] for r in all_rows], [float(r['acc']) for r in all_rows], plots_dir / 'acc_by_source.png', 'Acc@0.5 by source', 'Accuracy')
    plot_bar([r['source'] for r in all_rows], [float(r['best_acc']) for r in all_rows], plots_dir / 'best_acc_by_source.png', 'Best-threshold Acc by source', 'Accuracy')

    if writer is not None:
        for k in ['ap', 'roc_auc', 'acc', 'best_acc', 'real_acc', 'fake_acc']:
            if k in summary:
                writer.add_scalar(f'summary/{k}', float(summary[k]), 0)
        writer.add_text('summary/json', str(summary))
        writer.close()

    logger.info(f'summary={summary}')


if __name__ == '__main__':
    main()
