"""
Вариант 7.
"""
import asyncio
import random
import time
from typing import List, Tuple

SCALES = [1, 4, 8]


def work_unit(intensity: int) -> float:
    acc = 0.0
    for i in range(intensity):
        acc += (i * i) % 97
        acc = acc * 1.0000001
    return acc


def io_unit(ms: float) -> float:
    time.sleep(ms / 1000.0)
    return ms


async def io_unit_async(ms: float) -> float:
    await asyncio.sleep(ms / 1000.0)
    return ms


def run_sequential(cpu_intensities: List[int], io_delays: List[float]) -> Tuple[float, float]:
    cpu_total = sum(work_unit(k) for k in cpu_intensities)
    io_total = sum(io_unit(d) for d in io_delays)
    return cpu_total, io_total


def make_workload(scale: int = 4):
    rng = random.Random(42)
    cpu = [200_000, 800_000, 300_000, 1_500_000, 400_000] * scale
    io = [8 + rng.random() * 25 for _ in range(scale * 4)]
    return cpu, io


def run_task(scale: int = 4):
    cpu, io = make_workload(scale)
    t0 = time.perf_counter()
    result = run_sequential(cpu, io)
    elapsed = time.perf_counter() - t0
    return round(result[0], 2), round(result[1], 2), round(elapsed, 4)


if __name__ == "__main__":
    for s in SCALES:
        print(f"scale={s}  {run_task(s)}")
