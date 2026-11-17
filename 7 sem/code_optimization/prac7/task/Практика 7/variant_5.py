"""
Вариант 5.
"""
import math
import random
from typing import List

WINDOW = 51
SIZES = [10_000, 25_000, 50_000, 100_000, 200_000]
TINY = 52


def moving_average(data: List[float], window: int) -> List[float]:
    out = []
    for i in range(len(data) - window + 1):
        s = sum(data[i:i + window])
        out.append(s / window)
    return out


def moving_std(data: List[float], window: int) -> List[float]:
    out = []
    for i in range(len(data) - window + 1):
        chunk = data[i:i + window]
        mean = sum(chunk) / window
        var = sum((x - mean) ** 2 for x in chunk) / window
        out.append(math.sqrt(var))
    return out


def local_max(data: List[float], window: int) -> List[float]:
    return [max(data[i:i + window]) for i in range(len(data) - window + 1)]


def make_workload(size: int, window: int = WINDOW):
    rng = random.Random(19)
    data = [rng.gauss(0, 1) + 0.01 * i for i in range(size)]
    return data, window


def run_baseline(workload):
    data, w = workload
    return moving_average(data, w), moving_std(data, w), local_max(data, w)


def summarize(outputs):
    ma, ms, mx = outputs
    return sum(ma), sum(ms), sum(mx)


def run_task(size: int):
    import time
    workload = make_workload(size)
    t0 = time.perf_counter()
    outputs = run_baseline(workload)
    elapsed = time.perf_counter() - t0
    return summarize(outputs), elapsed


if __name__ == "__main__":
    for s in SIZES:
        res, t = run_task(s)
        print(f"size={s:<8} {res}  time={t:.4f}s")
