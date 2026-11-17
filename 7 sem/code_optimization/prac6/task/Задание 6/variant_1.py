"""
Вариант 1.
"""
import asyncio
import math
import random
import time
from typing import List, Tuple

SCALES = [1, 4, 8]


def cpu_job(seed: int, rounds: int = 1_000_000) -> float:
    acc = 0.0
    for i in range(rounds):
        acc += math.sqrt(abs(math.sin(i * 0.001 + seed)))
    return acc


def io_job(ms: float) -> str:
    time.sleep(ms / 1000.0)
    return f"ok-{ms:.1f}"


async def io_job_async(ms: float) -> str:
    await asyncio.sleep(ms / 1000.0)
    return f"ok-{ms:.1f}"


def run_sequential(cpu_seeds: List[int], io_delays: List[float]) -> Tuple[float, int]:
    s = sum(cpu_job(x) for x in cpu_seeds)
    answers = [io_job(d) for d in io_delays]
    return s, len(answers)


def make_workload(scale: int = 4):
    rng = random.Random(42)
    cpu_seeds = list(range(scale * 3))
    io_delays = [10 + rng.random() * 30 for _ in range(scale * 5)]
    return cpu_seeds, io_delays


def run_task(scale: int = 4):
    cpu_seeds, io_delays = make_workload(scale)
    t0 = time.perf_counter()
    result = run_sequential(cpu_seeds, io_delays)
    elapsed = time.perf_counter() - t0
    return round(result[0], 4), result[1], round(elapsed, 4)


if __name__ == "__main__":
    for s in SCALES:
        print(f"scale={s}  {run_task(s)}")
