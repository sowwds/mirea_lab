"""
Вариант 6.
"""
import asyncio
import math
import random
import time
from typing import List, Tuple

SCALES = [1, 4, 8]


def short_cpu(n: int = 150_000) -> float:
    return sum(math.sin(i) for i in range(n))


def long_cpu(n: int = 1_500_000) -> float:
    acc = 0.0
    x = 0.5
    for i in range(n):
        x = math.cos(x + i * 0.00001)
        acc += x
    return acc


def io_wait(ms: float) -> int:
    time.sleep(ms / 1000.0)
    return 1


async def io_wait_async(ms: float) -> int:
    await asyncio.sleep(ms / 1000.0)
    return 1


def run_sequential(n_short: int, n_long: int, io_delays: List[float]) -> Tuple[float, int]:
    s = sum(short_cpu() for _ in range(n_short))
    s += sum(long_cpu() for _ in range(n_long))
    io_count = sum(io_wait(d) for d in io_delays)
    return s, io_count


def make_workload(scale: int = 4):
    rng = random.Random(42)
    io_delays = [30 + rng.random() * 30 for _ in range(scale * 4)]
    return scale * 8, scale * 2, io_delays


def run_task(scale: int = 4):
    n_short, n_long, io_delays = make_workload(scale)
    t0 = time.perf_counter()
    result = run_sequential(n_short, n_long, io_delays)
    elapsed = time.perf_counter() - t0
    return round(result[0], 4), result[1], round(elapsed, 4)


if __name__ == "__main__":
    for s in SCALES:
        print(f"scale={s}  {run_task(s)}")
