from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check image/label pairs in a YOLO dataset")
    parser.add_argument("--root", required=True, help="Dataset root with train/images and train/labels")
    args = parser.parse_args()
    root = Path(args.root)
    for split in ["train", "val", "test"]:
        img_dir = root / split / "images"
        label_dir = root / split / "labels"
        if not img_dir.exists():
            continue
        images = sorted(p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        missing = [p for p in images if not (label_dir / f"{p.stem}.txt").exists()]
        empty = [p for p in images if (label_dir / f"{p.stem}.txt").exists() and (label_dir / f"{p.stem}.txt").stat().st_size == 0]
        print(f"{split}: images={len(images)}, missing_labels={len(missing)}, empty_labels={len(empty)}")
        if missing[:5]:
            print("  missing examples:", [p.name for p in missing[:5]])


if __name__ == "__main__":
    main()
