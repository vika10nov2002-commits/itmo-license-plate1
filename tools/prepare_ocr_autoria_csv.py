from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def find_autoria_splits(root: Path) -> list[Path]:
    return sorted(ann.parent for ann in root.rglob("ann") if ann.is_dir() and (ann.parent / "img").is_dir())


def find_image(img_dir: Path, stem_or_name: str) -> Path | None:
    candidate = img_dir / stem_or_name
    if candidate.exists() and candidate.is_file():
        return candidate
    stem = Path(stem_or_name).stem
    for ext in IMAGE_EXTS:
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def read_samples(split_dir: Path) -> list[tuple[str, str]]:
    ann_dir = split_dir / "ann"
    img_dir = split_dir / "img"
    samples: list[tuple[str, str]] = []
    for ann_path in sorted(ann_dir.glob("*.json")):
        try:
            meta = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Skip broken JSON {ann_path}: {exc}")
            continue

        text = str(meta.get("description") or meta.get("text") or "").strip().upper()
        if not text:
            continue

        image_name = str(meta.get("name") or ann_path.stem)
        img_path = find_image(img_dir, image_name)
        if img_path is None:
            img_path = find_image(img_dir, ann_path.stem)
        if img_path is None:
            continue
        samples.append((img_path.resolve().as_posix(), text))
    return samples


def split_name(path: Path) -> str | None:
    name = path.name.lower()
    if "train" in name:
        return "train"
    if "val" in name or "valid" in name:
        return "val"
    if "test" in name:
        return "test"
    return None


def write_csv(path: Path, samples: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "text"])
        writer.writerows(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert AutoRia/Nomeroff OCR folders ann+img to train/val/test CSV indexes")
    parser.add_argument("--src", required=True, help="Root with split folders containing ann/ and img/")
    parser.add_argument("--out", default="data/ocr_indexes", help="Output folder for ocr_train.csv, ocr_val.csv, ocr_test.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    args = parser.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    split_dirs = find_autoria_splits(src)
    if not split_dirs:
        raise RuntimeError(f"No ann/ + img/ pairs found inside {src}")

    by_split = {"train": [], "val": [], "test": []}
    unknown: list[tuple[str, str]] = []
    for split_dir in split_dirs:
        samples = read_samples(split_dir)
        key = split_name(split_dir)
        if key is None:
            unknown.extend(samples)
        else:
            by_split[key].extend(samples)

    # If explicit split names are absent or incomplete, make a deterministic split from all samples.
    if unknown or not by_split["train"] or not by_split["val"]:
        all_samples = by_split["train"] + by_split["val"] + by_split["test"] + unknown
        rng = random.Random(args.seed)
        rng.shuffle(all_samples)
        n = len(all_samples)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        by_split = {
            "train": all_samples[:n_train],
            "val": all_samples[n_train:n_train + n_val],
            "test": all_samples[n_train + n_val:],
        }

    alphabet = sorted({ch for samples in by_split.values() for _, text in samples for ch in text})
    for key, samples in by_split.items():
        write_csv(out / f"ocr_{key}.csv", samples)
        print(f"{key}: {len(samples)} -> {out / f'ocr_{key}.csv'}")

    (out / "alphabet.txt").write_text("".join(alphabet), encoding="utf-8")
    print(f"alphabet ({len(alphabet)}): {''.join(alphabet)}")


if __name__ == "__main__":
    main()
