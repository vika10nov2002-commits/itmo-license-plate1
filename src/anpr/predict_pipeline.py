from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from .predict_recognizer import load_recognizer, recognize_image


def crop_with_padding(image: Image.Image, box, pad_ratio: float = 0.05) -> Image.Image:
    w, h = image.size
    x1, y1, x2, y2 = map(float, box)
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * pad_ratio, bh * pad_ratio
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))
    return image.crop((x1, y1, x2, y2))


def draw_result(image: Image.Image, detections: list[dict], output: str | Path) -> None:
    arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        text = f"{det['text']} {det['det_conf']:.2f}"
        cv2.rectangle(arr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(arr, text, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end ANPR: YOLO detector + CRNN OCR")
    parser.add_argument("--image", required=True)
    parser.add_argument("--detector", required=True, help="YOLO weights, e.g. runs/detect/.../best.pt")
    parser.add_argument("--recognizer", required=True, help="CRNN checkpoint")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/prediction.jpg")
    args = parser.parse_args()

    device_str = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_str)

    image = Image.open(args.image).convert("RGB")
    detector = YOLO(args.detector)
    recognizer, alphabet, transform = load_recognizer(args.recognizer, device)

    yolo_results = detector.predict(source=args.image, conf=args.conf, device=device_str, verbose=False)
    detections = []
    if yolo_results and yolo_results[0].boxes is not None:
        for box in yolo_results[0].boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().tolist()
            conf = float(box.conf[0].detach().cpu())
            crop = crop_with_padding(image, xyxy)
            text = recognize_image(crop, recognizer, alphabet, transform, device)
            detections.append({"box": xyxy, "det_conf": conf, "text": text})

    for i, det in enumerate(detections, start=1):
        print(f"{i}. text={det['text']!r}, det_conf={det['det_conf']:.3f}, box={list(map(int, det['box']))}")
    if not detections:
        print("No license plate detected")

    draw_result(image, detections, args.output)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
