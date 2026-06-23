from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_RU_PLATE_ALPHABET = "0123456789ABEKMHOPCTYX"
DEFAULT_FULL_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class CTCAlphabet:
    """Mapping between characters and CTC class ids.

    Class 0 is reserved for the CTC blank token. Characters start from 1.
    """

    chars: str
    blank_idx: int = 0

    def __post_init__(self) -> None:
        unique = []
        for ch in self.chars:
            if ch not in unique:
                unique.append(ch)
        object.__setattr__(self, "chars", "".join(unique))
        object.__setattr__(self, "char_to_idx", {ch: i + 1 for i, ch in enumerate(self.chars)})
        object.__setattr__(self, "idx_to_char", {i + 1: ch for i, ch in enumerate(self.chars)})

    @property
    def vocab_size(self) -> int:
        return len(self.chars) + 1

    def normalize(self, text: str, *, keep_unknown: bool = False) -> str:
        text = str(text).upper().replace(" ", "").replace("-", "")
        if keep_unknown:
            return text
        return "".join(ch for ch in text if ch in self.char_to_idx)

    def encode(self, text: str) -> list[int]:
        text = self.normalize(text)
        return [self.char_to_idx[ch] for ch in text]

    def decode_greedy(self, ids) -> str:
        """CTC greedy decode: collapse repeats and remove blank."""
        if hasattr(ids, "detach"):
            ids = ids.detach().cpu().tolist()
        prev = None
        out: list[str] = []
        for idx in ids:
            idx = int(idx)
            if idx != self.blank_idx and idx != prev:
                ch = self.idx_to_char.get(idx)
                if ch is not None:
                    out.append(ch)
            prev = idx
        return "".join(out)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.chars + "\n", encoding="utf-8")

    @classmethod
    def from_file(cls, path: str | Path) -> "CTCAlphabet":
        chars = Path(path).read_text(encoding="utf-8").strip()
        return cls(chars)

    @classmethod
    def from_config(cls, value: str | None, alphabet_path: str | None = None) -> "CTCAlphabet":
        if alphabet_path:
            return cls.from_file(alphabet_path)
        if value == "ru_plate":
            return cls(DEFAULT_RU_PLATE_ALPHABET)
        if value == "full":
            return cls(DEFAULT_FULL_ALPHABET)
        if value:
            return cls(value)
        return cls(DEFAULT_RU_PLATE_ALPHABET)
