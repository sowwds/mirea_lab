"""
Вариант 0.
"""
import asyncio
import math
import random
import time
from typing import List, Tuple

SCALES = [1, 4, 8]


def cpu_heavy(n: int) -> float:
    acc = 0.0
    x = 1.0
    for i in range(n):
        x = math.sin(x) * math.cos(x) + 0.000001 * i
        acc += x
    return acc


def io_like(delay_ms: float, payload: int = 1000) -> int:
    time.sleep(delay_ms / 1000.0)
    s = 0
    for i in range(payload):
        s += i % 7
    return s


async def io_like_async(delay_ms: float, payload: int = 1000) -> int:
    await asyncio.sleep(delay_ms / 1000.0)
    s = 0
    for i in range(payload):
        s += i % 7
    return s


def run_sequential(cpu_tasks: List[int], io_tasks: List[float]) -> Tuple[float, int]:
    cpu_sum = sum(cpu_heavy(n) for n in cpu_tasks)
    io_sum = sum(io_like(d) for d in io_tasks)
    return cpu_sum, io_sum


def make_workload(scale: int = 4):
    rng = random.Random(42)
    cpu_tasks = [900_000 + rng.randint(0, 300_000) for _ in range(scale * 2)]
    io_tasks = [40 + rng.random() * 40 for _ in range(scale * 4)]
    return cpu_tasks, io_tasks


def run_task(scale: int = 4):
    cpu_tasks, io_tasks = make_workload(scale)
    t0 = time.perf_counter()
    result = run_sequential(cpu_tasks, io_tasks)
    elapsed = time.perf_counter() - t0
    return round(result[0], 4), result[1], round(elapsed, 4)


if __name__ == "__main__":
    for s in SCALES:
        print(f"scale={s}  {run_task(s)}")
