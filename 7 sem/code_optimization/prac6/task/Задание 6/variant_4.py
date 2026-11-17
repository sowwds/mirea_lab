"""
Вариант 4.
"""
import asyncio
import random
import time
from typing import List, Tuple

SCALES = [1, 4, 8]
_ALPHABET = "abcdefghijklmnopqrstuvwxyz "


def text_features(text: str) -> Tuple[int, int, float]:
    vowels = sum(1 for c in text.lower() if c in "aeiouyаеёиоуыэюя")
    score = 0.0
    for i, c in enumerate(text):
        score += (ord(c) * 17 + i) % 97
        for _ in range(40):
            score = (score * 1.0001) % 10000
    return len(text), vowels, score


def _make_text(doc_id: int) -> str:
    rng = random.Random(doc_id)
    length = 800 + rng.randint(0, 800)
    return "".join(rng.choice(_ALPHABET) for _ in range(length))


def load_document(doc_id: int, delay_ms: float = 5.0) -> str:
    time.sleep(delay_ms / 1000.0)
    return _make_text(doc_id)


async def load_document_async(doc_id: int, delay_ms: float = 5.0) -> str:
    await asyncio.sleep(delay_ms / 1000.0)
    return _make_text(doc_id)


def run_sequential(doc_ids: List[int]) -> Tuple[float, int]:
    total_score = 0.0
    for did in doc_ids:
        text = load_document(did)
        _, _, score = text_features(text)
        total_score += score
    return total_score, len(doc_ids)


def make_workload(scale: int = 4):
    return list(range(scale * 40))


def run_task(scale: int = 4):
    ids = make_workload(scale)
    t0 = time.perf_counter()
    result = run_sequential(ids)
    elapsed = time.perf_counter() - t0
    return round(result[0], 2), result[1], round(elapsed, 4)


if __name__ == "__main__":
    for s in SCALES:
        print(f"scale={s}  {run_task(s)}")
