"""
Вариант 3. 
"""
import asyncio
import random
import statistics
import time
from typing import List, Tuple

SCALES = [1, 4, 8]


def analyze_packet(data: List[float]) -> dict:
    expanded = []
    for x in data:
        v = x
        for _ in range(30):
            v = (v * 1.001) % 1.0
        expanded.append(v)
    return {
        "mean": statistics.fmean(expanded),
        "stdev": statistics.pstdev(expanded) if len(expanded) > 1 else 0.0,
        "n": len(expanded),
    }


def _make_packet(n: int) -> List[float]:
    rng = random.Random(n)
    return [rng.random() for _ in range(n)]


def fetch_packet(n: int, delay_ms: float = 10.0) -> List[float]:
    time.sleep(delay_ms / 1000.0)
    return _make_packet(n)


async def fetch_packet_async(n: int, delay_ms: float = 10.0) -> List[float]:
    await asyncio.sleep(delay_ms / 1000.0)
    return _make_packet(n)


def run_sequential(sizes: List[int]) -> Tuple[float, int]:
    acc = 0.0
    for sz in sizes:
        data = fetch_packet(sz)
        stats = analyze_packet(data)
        acc += stats["mean"] + stats["stdev"]
    return acc, len(sizes)


def make_workload(scale: int = 4):
    rng = random.Random(42)
    return [30_000 + rng.randint(0, 30_000) for _ in range(scale * 4)]


def run_task(scale: int = 4):
    sizes = make_workload(scale)
    t0 = time.perf_counter()
    result = run_sequential(sizes)
    elapsed = time.perf_counter() - t0
    return round(result[0], 4), result[1], round(elapsed, 4)


if __name__ == "__main__":
    for s in SCALES:
        print(f"scale={s}  {run_task(s)}")
