"""
Вариант 9.
"""
import asyncio
import math
import random
import time
from typing import List, Tuple

SCALES = [1, 4, 8]


def _make_numbers(n: int) -> List[int]:
    rng = random.Random(n)
    return [rng.randint(1, 1000) for _ in range(12_000)]


def stage_fetch(n: int) -> List[int]:
    time.sleep(0.03)
    return _make_numbers(n)


async def stage_fetch_async(n: int) -> List[int]:
    await asyncio.sleep(0.03)
    return _make_numbers(n)


def stage_cpu(nums: List[int]) -> float:
    acc = 0.0
    for v in nums:
        x = float(v)
        for _ in range(25):
            x = math.sin(x) + math.cos(x / 2)
        acc += x
    return acc


def stage_post(value: float, state: float) -> float:
    for _ in range(400_000):
        state = (state + value * 1e-6 + 0.001) * 1.00001 % 1000.0
    return state


def run_sequential(jobs: List[int]) -> Tuple[float, int]:
    state = 0.0
    for j in jobs:
        data = stage_fetch(j)
        mid = stage_cpu(data)
        state = stage_post(mid, state)
    return state, len(jobs)


def make_workload(scale: int = 4):
    return list(range(scale * 4))


def run_task(scale: int = 4):
    jobs = make_workload(scale)
    t0 = time.perf_counter()
    result = run_sequential(jobs)
    elapsed = time.perf_counter() - t0
    return round(result[0], 4), result[1], round(elapsed, 4)


if __name__ == "__main__":
    for s in SCALES:
        print(f"scale={s}  {run_task(s)}")
