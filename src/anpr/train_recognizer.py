from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .alphabet import CTCAlphabet
from .dataset import PlateOCRDataset, ocr_collate_fn
from .metrics import summarize_ocr
from .models import build_crnn, count_parameters
from .transforms import PlateTransform


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_loader(cfg: dict, split: str, alphabet: CTCAlphabet, train: bool) -> DataLoader:
    transform = PlateTransform(
        height=int(cfg.get("image_height", 32)),
        width=int(cfg.get("image_width", 128)),
        train=train and bool(cfg.get("augment", True)),
        grayscale=bool(cfg.get("grayscale", True)),
    )
    dataset = PlateOCRDataset(
        root=cfg["data_root"],
        split=split,
        alphabet=alphabet,
        transform=transform,
        dataset_format=cfg.get("dataset_format", "autoria_json"),
        csv_file=cfg.get(f"{split}_csv"),
        max_samples=cfg.get(f"max_{split}_samples"),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=train,
        num_workers=int(cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=ocr_collate_fn,
        drop_last=False,
    )


def decode_batch(log_probs: torch.Tensor, alphabet: CTCAlphabet) -> list[str]:
    # log_probs: [T,B,C]
    best = log_probs.permute(1, 0, 2).argmax(dim=2)
    return [alphabet.decode_greedy(seq) for seq in best]


def run_epoch(model, loader, optimizer, criterion, device, alphabet, train: bool) -> dict:
    model.train(train)
    total_loss = 0.0
    predictions: list[str] = []
    targets_text: list[str] = []
    num_batches = 0

    iterator = tqdm(loader, desc="train" if train else "eval", leave=False)
    for images, targets, target_lengths, texts, _ in iterator:
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        with torch.set_grad_enabled(train):
            log_probs = model(images)
            time_steps, batch_size, _ = log_probs.shape
            input_lengths = torch.full(
                size=(batch_size,), fill_value=time_steps, dtype=torch.long, device=device
            )
            loss = criterion(log_probs, targets, input_lengths, target_lengths)

            if not torch.isfinite(loss):
                print("Warning: non-finite CTC loss skipped. Check labels, alphabet, and input width.")
                continue

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1
        predictions.extend(decode_batch(log_probs.detach(), alphabet))
        targets_text.extend(texts)
        iterator.set_postfix(loss=f"{loss.item():.4f}")

    metrics = summarize_ocr(predictions, targets_text)
    metrics["loss"] = total_loss / max(1, num_batches)
    return metrics


def save_checkpoint(path: Path, model, cfg: dict, alphabet: CTCAlphabet, metrics: dict, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg,
            "alphabet": alphabet.chars,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/recognition.yaml")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(args.device)

    alphabet = CTCAlphabet.from_config(cfg.get("alphabet"), cfg.get("alphabet_path"))
    train_loader = make_loader(cfg, cfg.get("train_split", "train"), alphabet, train=True)
    val_split = cfg.get("val_split", "test")
    val_loader = make_loader(cfg, val_split, alphabet, train=False)

    model = build_crnn(
        architecture=cfg.get("architecture", "small"),
        vocab_size=alphabet.vocab_size,
        in_channels=1 if cfg.get("grayscale", True) else 3,
        hidden_size=cfg.get("hidden_size"),
    ).to(device)

    criterion = nn.CTCLoss(blank=alphabet.blank_idx, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=int(cfg.get("lr_patience", 3))
    )

    output_dir = Path(cfg.get("output_dir", "checkpoints/recognition"))
    exp_name = cfg.get("experiment_name", "crnn")
    history_path = output_dir / f"{exp_name}_history.csv"
    best_path = output_dir / f"{exp_name}_best.pt"
    last_path = output_dir / f"{exp_name}_last.pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Alphabet: {alphabet.chars} ({alphabet.vocab_size} CTC classes including blank)")
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Train samples: {len(train_loader.dataset)}, val samples: {len(val_loader.dataset)}")

    best_cer = math.inf
    rows = []
    for epoch in range(1, int(cfg.get("epochs", 30)) + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, device, alphabet, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, criterion, device, alphabet, train=False)
        scheduler.step(val_metrics["cer"])

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_cer": train_metrics["cer"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_cer": val_metrics["cer"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        rows.append(row)
        print(
            f"Epoch {epoch:03d}: "
            f"train_loss={row['train_loss']:.4f}, train_cer={row['train_cer']:.4f}, "
            f"val_loss={row['val_loss']:.4f}, val_cer={row['val_cer']:.4f}, "
            f"val_acc={row['val_accuracy']:.4f}"
        )

        save_checkpoint(last_path, model, cfg, alphabet, val_metrics, epoch)
        if val_metrics["cer"] < best_cer:
            best_cer = val_metrics["cer"]
            save_checkpoint(best_path, model, cfg, alphabet, val_metrics, epoch)
            print(f"Saved new best checkpoint: {best_path}")

        with history_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
