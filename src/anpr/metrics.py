from __future__ import annotations


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            insert = cur[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (ca != cb)
            cur.append(min(insert, delete, replace))
        prev = cur
    return prev[-1]


def cer(predictions: list[str], targets: list[str]) -> float:
    total_edits = sum(edit_distance(p, t) for p, t in zip(predictions, targets))
    total_chars = sum(max(1, len(t)) for t in targets)
    return total_edits / max(1, total_chars)


def exact_match_accuracy(predictions: list[str], targets: list[str]) -> float:
    if not targets:
        return 0.0
    return sum(p == t for p, t in zip(predictions, targets)) / len(targets)


def summarize_ocr(predictions: list[str], targets: list[str]) -> dict[str, float]:
    return {
        "accuracy": exact_match_accuracy(predictions, targets),
        "cer": cer(predictions, targets),
    }
