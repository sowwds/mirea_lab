"""
Вариант 5.
"""
import asyncio
import random
import time
from typing import List, Tuple

SCALES = [1, 4, 8]


def extract_features(tokens: List[str]) -> float:
    score = 0.0
    for i, tok in enumerate(tokens):
        h = 0
        for ch in tok:
            h = (h * 31 + ord(ch)) % 10_000_019
        score += (h % 100) / (i + 1)
        for _ in range(8):
            score = (score + 0.001) * 1.00001
    return score


def _make_tokens(doc_id: int) -> List[str]:
    rng = random.Random(doc_id)
    n = 20_000 + rng.randint(0, 20_000)
    return [f"w{rng.randint(0, 500)}" for _ in range(n)]


def load_tokens(doc_id: int, delay_ms: float = 40.0) -> List[str]:
    time.sleep(delay_ms / 1000.0)
    return _make_tokens(doc_id)


async def load_tokens_async(doc_id: int, delay_ms: float = 40.0) -> List[str]:
    await asyncio.sleep(delay_ms / 1000.0)
    return _make_tokens(doc_id)


def run_sequential(ids: List[int]) -> Tuple[float, int]:
    s = 0.0
    for i in ids:
        tokens = load_tokens(i)
        s += extract_features(tokens)
    return s, len(ids)


def make_workload(scale: int = 4):
    return list(range(scale * 4))


def run_task(scale: int = 4):
    ids = make_workload(scale)
    t0 = time.perf_counter()
    result = run_sequential(ids)
    elapsed = time.perf_counter() - t0
    return round(result[0], 4), result[1], round(elapsed, 4)


if __name__ == "__main__":
    for s in SCALES:
        print(f"scale={s}  {run_task(s)}")
