from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() == ".zip":
        tmp = source.parent / (source.stem + "_unzipped")
        if not tmp.exists():
            with zipfile.ZipFile(source, "r") as zf:
                zf.extractall(tmp)
        source = tmp
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare image folders for manual YOLO annotation. This does not create fake labels."
    )
    parser.add_argument("--source", required=True, help="Folder or zip with raw car images")
    parser.add_argument("--out", default="data/detection_raw")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    images = collect_images(Path(args.source))
    if not images:
        raise RuntimeError("No images found")

    random.shuffle(images)
    val_count = max(1, int(len(images) * args.val_ratio))
    splits = {"val": images[:val_count], "train": images[val_count:]}

    out = Path(args.out)
    for split, paths in splits.items():
        img_out = out / split / "images"
        label_out = out / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for idx, src in enumerate(paths):
            dst = img_out / f"{src.stem}_{idx:05d}{src.suffix.lower()}"
            shutil.copy2(src, dst)

    print(f"Copied {len(images)} images to {out}")
    print("Next step: annotate plates in LabelImg/CVAT/Roboflow and save YOLO .txt labels into labels/ folders.")
    print("Do not train YOLO before labels exist: empty labels mean background-only images.")


if __name__ == "__main__":
    main()
