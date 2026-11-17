"""
Вариант 8.
"""
import math
import random
from typing import List

MAX_LAG = 200
SIZES = [2000, 4000, 8000, 16000, 32000]
TINY = 12


def autocorrelation(signal: List[float], max_lag: int) -> List[float]:
    n = len(signal)
    mean = sum(signal) / n
    centered = [x - mean for x in signal]
    acf = []
    for lag in range(max_lag + 1):
        s = 0.0
        for i in range(n - lag):
            s += centered[i] * centered[i + lag]
        acf.append(s / (n - lag))
    return acf


def cross_correlation(a: List[float], b: List[float], max_lag: int) -> List[float]:
    n = min(len(a), len(b))
    result = []
    for lag in range(-max_lag, max_lag + 1):
        s = 0.0
        count = 0
        for i in range(n):
            j = i + lag
            if 0 <= j < n:
                s += a[i] * b[j]
                count += 1
        result.append(s / count if count else 0.0)
    return result


def make_workload(size: int):
    rng = random.Random(31)
    t = [i * 0.01 for i in range(size)]
    signal = [math.sin(2 * math.pi * 3 * ti) + 0.3 * rng.gauss(0, 1) for ti in t]
    signal2 = [math.sin(2 * math.pi * 3 * ti + 0.4) + 0.3 * rng.gauss(0, 1) for ti in t]
    return signal, signal2, min(MAX_LAG, size // 4)


def run_baseline(workload):
    sig, sig2, lag = workload
    return autocorrelation(sig, lag), cross_correlation(sig, sig2, lag)


def summarize(outputs):
    acf, xcorr = outputs
    return sum(v * v for v in acf), sum(v * v for v in xcorr)


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
        print(f"size={s:<6} {res}  time={t:.4f}s")
