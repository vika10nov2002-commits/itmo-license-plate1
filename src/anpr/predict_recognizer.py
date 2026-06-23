from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from .alphabet import CTCAlphabet
from .models import build_crnn
from .transforms import PlateTransform


def load_recognizer(checkpoint_path: str | Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})
    alphabet = CTCAlphabet(ckpt["alphabet"])
    model = build_crnn(
        architecture=cfg.get("architecture", "small"),
        vocab_size=alphabet.vocab_size,
        in_channels=1 if cfg.get("grayscale", True) else 3,
        hidden_size=cfg.get("hidden_size"),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    transform = PlateTransform(
        height=int(cfg.get("image_height", 32)),
        width=int(cfg.get("image_width", 128)),
        train=False,
        grayscale=bool(cfg.get("grayscale", True)),
    )
    return model, alphabet, transform


@torch.no_grad()
def recognize_image(image: Image.Image, model, alphabet: CTCAlphabet, transform: PlateTransform, device: torch.device) -> str:
    tensor = transform(image).unsqueeze(0).to(device)
    log_probs = model(tensor)
    best = log_probs.permute(1, 0, 2).argmax(dim=2)[0]
    return alphabet.decode_greedy(best)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model, alphabet, transform = load_recognizer(args.checkpoint, device)
    image = Image.open(args.image).convert("RGB")
    print(recognize_image(image, model, alphabet, transform, device))


if __name__ == "__main__":
    main()
