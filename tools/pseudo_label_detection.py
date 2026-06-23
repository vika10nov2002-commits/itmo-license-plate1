from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        if min(self.train, self.val, self.test) < 0:
            raise ValueError("Split ratios must be non-negative")


def extract_if_zip(source: Path, work_dir: Path) -> Path:
    if source.is_file() and source.suffix.lower() == ".zip":
        extract_dir = work_dir / f"_{source.stem}_extracted"
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(source, "r") as zf:
                zf.extractall(extract_dir)
        return extract_dir
    return source


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def split_images(images: list[Path], ratios: SplitRatios, seed: int) -> dict[str, list[Path]]:
    ratios.validate()
    rng = random.Random(seed)
    items = images[:]
    rng.shuffle(items)
    n = len(items)
    n_train = int(n * ratios.train)
    n_val = int(n * ratios.val)
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }


def unique_image_name(src: Path, idx: int) -> str:
    # Avoid collisions when several folders contain `1.jpg`.
    return f"{src.stem}_{idx:06d}{src.suffix.lower()}"


def yolo_lines_from_result(result, class_id: int = 0, max_boxes: int | None = None) -> list[str]:
    boxes = []
    if result.boxes is None:
        return []
    for box in result.boxes:
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        cx, cy, w, h = [float(x) for x in box.xywhn[0].tolist()]
        boxes.append((conf, cx, cy, w, h))
    boxes.sort(reverse=True, key=lambda x: x[0])
    if max_boxes is not None:
        boxes = boxes[:max_boxes]
    return [f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for _, cx, cy, w, h in boxes]


def draw_preview(image_path: Path, label_lines: Iterable[str], out_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    W, H = img.size
    for line in label_lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, w, h = parts
        cx, cy, w, h = map(float, (cx, cy, w, h))
        x1 = (cx - w / 2) * W
        y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W
        y2 = (cy + h / 2) * H
        draw.rectangle([x1, y1, x2, y2], width=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def write_data_yaml(out: Path) -> None:
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: license_plate\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a YOLO license-plate dataset with pseudo-labels from a pretrained plate detector. "
            "Use this only as a bootstrap; inspect and fix labels manually before final training."
        )
    )
    parser.add_argument("--source", required=True, help="Folder or .zip with raw car images")
    parser.add_argument("--out", default="data/detection_yolo", help="Output YOLO dataset folder")
    parser.add_argument("--weights", required=True, help="Pretrained LICENSE PLATE detector weights, not plain COCO yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold for pseudo-labels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-boxes", type=int, default=1, help="License-plate datasets usually need one best box per image")
    parser.add_argument("--keep-empty", action="store_true", help="Keep images where detector found no plate as negative samples")
    parser.add_argument("--preview", type=int, default=20, help="Number of preview images with drawn pseudo-labels")
    args = parser.parse_args()

    from ultralytics import YOLO

    out = Path(args.out)
    source = extract_if_zip(Path(args.source), out)
    images = list_images(source)
    if not images:
        raise RuntimeError(f"No images found in {source}")

    ratios = SplitRatios(args.train_ratio, args.val_ratio, args.test_ratio)
    splits = split_images(images, ratios, args.seed)
    detector = YOLO(args.weights)

    for split in ["train", "val", "test"]:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped_empty = 0
    total_boxes = 0
    preview_left = args.preview

    for split, paths in splits.items():
        for local_idx, src in enumerate(paths):
            result = detector.predict(source=str(src), conf=args.conf, verbose=False)[0]
            lines = yolo_lines_from_result(result, class_id=0, max_boxes=args.max_boxes)
            if not lines and not args.keep_empty:
                skipped_empty += 1
                continue

            name = unique_image_name(src, copied)
            dst_img = out / "images" / split / name
            dst_label = out / "labels" / split / f"{Path(name).stem}.txt"
            shutil.copy2(src, dst_img)
            dst_label.write_text("\n".join(lines), encoding="utf-8")
            copied += 1
            total_boxes += len(lines)

            if preview_left > 0 and lines:
                draw_preview(dst_img, lines, out / "preview" / split / name)
                preview_left -= 1

    write_data_yaml(out)
    print(f"Input images: {len(images)}")
    print(f"Copied images: {copied}")
    print(f"Skipped images without pseudo-labels: {skipped_empty}")
    print(f"Pseudo-label boxes: {total_boxes}")
    print(f"YOLO config: {out / 'data.yaml'}")
    print(f"Preview folder: {out / 'preview'}")
    print("IMPORTANT: pseudo-labels are not ground truth. Open preview images and fix wrong/missing boxes before final metrics.")


if __name__ == "__main__":
    main()
