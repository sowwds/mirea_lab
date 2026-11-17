"""
Вариант 2.
"""
import asyncio
import random
import time
from typing import List, Tuple

SCALES = [1, 4, 8]


def process_block(block: List[float]) -> float:
    total = 0.0
    for x in block:
        y = x
        for _ in range(120):
            y = y * 1.0001 + 0.00001
        total += y
    return total


def _make_block(size: int) -> List[float]:
    rng = random.Random(size)
    return [rng.random() for _ in range(size)]


def read_like(size: int, delay_ms: float = 60.0) -> List[float]:
    time.sleep(delay_ms / 1000.0)
    return _make_block(size)


async def read_like_async(size: int, delay_ms: float = 60.0) -> List[float]:
    await asyncio.sleep(delay_ms / 1000.0)
    return _make_block(size)


def run_sequential(block_sizes: List[int]) -> Tuple[float, int]:
    results = []
    for sz in block_sizes:
        data = read_like(sz)
        results.append(process_block(data))
    return sum(results), len(results)


def make_workload(scale: int = 4):
    rng = random.Random(42)
    return [6000 + rng.randint(0, 6000) for _ in range(scale * 4)]


def run_task(scale: int = 4):
    sizes = make_workload(scale)
    t0 = time.perf_counter()
    result = run_sequential(sizes)
    elapsed = time.perf_counter() - t0
    return round(result[0], 4), result[1], round(elapsed, 4)


if __name__ == "__main__":
    for s in SCALES:
        print(f"scale={s}  {run_task(s)}")
