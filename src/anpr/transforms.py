from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter


@dataclass
class PlateTransform:
    """Resize a cropped plate without destroying its aspect ratio.

    We pad the plate to a fixed canvas instead of blindly stretching it.
    Horizontal flip is intentionally not used: mirrored license-plate text is a bad OCR target.
    """

    height: int = 32
    width: int = 128
    train: bool = False
    grayscale: bool = True
    mean: float = 0.5
    std: float = 0.5

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("L" if self.grayscale else "RGB")

        if self.train:
            image = self._augment(image)

        image = self._letterbox(image)
        arr = np.asarray(image).astype("float32") / 255.0
        if self.grayscale:
            arr = arr[None, :, :]
        else:
            arr = arr.transpose(2, 0, 1)
        tensor = torch.from_numpy(arr)
        return (tensor - self.mean) / self.std

    def _letterbox(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        scale = min(self.width / max(1, w), self.height / max(1, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        fill = 255 if self.grayscale else (255, 255, 255)
        canvas = Image.new(image.mode, (self.width, self.height), fill)
        x0 = (self.width - new_w) // 2
        y0 = (self.height - new_h) // 2
        canvas.paste(image, (x0, y0))
        return canvas

    def _augment(self, image: Image.Image) -> Image.Image:
        if random.random() < 0.35:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.35))
        if random.random() < 0.35:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
        if random.random() < 0.20:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 0.7)))
        if random.random() < 0.20:
            angle = random.uniform(-3.0, 3.0)
            image = image.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=255)
        return image
