from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLOv8 license-plate detector")
    parser.add_argument("--data", default="configs/detection.yaml", help="YOLO dataset YAML")
    parser.add_argument("--model", default="yolov8n.pt", help="Base YOLO model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="license_plate_detector")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    device = None if args.device == "auto" else args.device
    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=10,
    )
    print(results)

    if args.eval:
        metrics = model.val(data=args.data, imgsz=args.imgsz, device=device)
        print(metrics)


if __name__ == "__main__":
    main()
