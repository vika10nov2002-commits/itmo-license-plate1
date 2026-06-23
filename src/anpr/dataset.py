from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from .alphabet import CTCAlphabet
from .transforms import PlateTransform

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class PlateOCRDataset(Dataset):
    """OCR dataset for cropped license plates.

    Supported formats:
    1. AutoRia-style JSON:
       root/train/ann/*.json, root/train/img/*.png, label in JSON field `description`.
    2. CSV:
       file with columns `image` and `text`; image paths are either absolute or relative to root.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        alphabet: CTCAlphabet,
        transform: PlateTransform | None = None,
        dataset_format: str = "autoria_json",
        csv_file: str | Path | None = None,
        max_samples: int | None = None,
        skip_empty_labels: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.alphabet = alphabet
        self.transform = transform
        self.dataset_format = dataset_format
        self.skip_empty_labels = skip_empty_labels

        if dataset_format == "autoria_json":
            self.samples = self._load_autoria_json()
        elif dataset_format == "csv":
            if csv_file is None:
                csv_file = self.root / f"{split}.csv"
            self.samples = self._load_csv(Path(csv_file))
        else:
            raise ValueError(f"Unsupported dataset_format={dataset_format!r}")

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if not self.samples:
            raise RuntimeError(
                f"No OCR samples found. root={self.root}, split={split}, format={dataset_format}. "
                "Check paths and labels."
            )

    def _load_autoria_json(self) -> list[dict[str, Any]]:
        ann_dir = self.root / self.split / "ann"
        img_dir = self.root / self.split / "img"
        samples: list[dict[str, Any]] = []
        for ann_path in sorted(ann_dir.glob("*.json")):
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            text = self.alphabet.normalize(str(data.get("description", "")))
            if self.skip_empty_labels and not text:
                continue
            stem = str(data.get("name") or ann_path.stem)
            img_path = self._find_image(img_dir, stem)
            if img_path is None:
                continue
            samples.append({"image": img_path, "text": text})
        return samples

    def _load_csv(self, csv_file: Path) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        with csv_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = self.alphabet.normalize(str(row.get("text", "")))
                if self.skip_empty_labels and not text:
                    continue
                image_raw = row.get("image") or row.get("filename") or row.get("path")
                if not image_raw:
                    continue
                img_path = Path(image_raw)
                if not img_path.is_absolute():
                    img_path = self.root / img_path
                if img_path.exists():
                    samples.append({"image": img_path, "text": text})
        return samples

    @staticmethod
    def _find_image(img_dir: Path, stem: str) -> Path | None:
        for ext in _IMAGE_EXTS:
            candidate = img_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        # Some JSON files store names with extensions already.
        direct = img_dir / stem
        if direct.exists():
            return direct
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample["image"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label_ids = self.alphabet.encode(sample["text"])
        return image, torch.tensor(label_ids, dtype=torch.long), sample["text"], str(sample["image"])


def ocr_collate_fn(batch):
    # Defensive filtering: CTCLoss cannot consume empty target sequences.
    batch = [item for item in batch if len(item[1]) > 0]
    if not batch:
        raise ValueError("Batch contains only empty labels")
    images, labels, texts, paths = zip(*batch)
    images = torch.stack(list(images), dim=0)
    label_lengths = torch.tensor([len(x) for x in labels], dtype=torch.long)
    targets = torch.cat(list(labels), dim=0)
    return images, targets, label_lengths, list(texts), list(paths)
