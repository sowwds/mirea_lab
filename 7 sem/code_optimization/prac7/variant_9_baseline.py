"""
Вариант 9.
"""
import math
import random
from array import array
from typing import List

WINDOW = 30
SIZES = [16000, 32000, 64000, 128000, 240000]
TINY = 8


def apply_kernel(data: List[float], kernel: List[float]) -> List[float]:
    k = len(kernel)
    out = []
    for i in range(len(data) - k + 1):
        out.append(sum(data[i + j] * kernel[j] for j in range(k)))
    return out


def normalize_minmax(data: List[float]) -> List[float]:
    lo, hi = min(data), max(data)
    if hi == lo:
        return [0.0] * len(data)
    return [(x - lo) / (hi - lo) for x in data]


def pairwise_features(xs: List[float], ys: List[float]) -> array:
    feats = array('d')
    n = min(len(xs), len(ys))
    for i in range(n):
        for j in range(i + 1, min(i + WINDOW, n)):
            feats.append(xs[i] - xs[j])
            feats.append(xs[i] * ys[j])
    return feats


def make_workload(size: int):
    rng = random.Random(37)
    data = [math.sin(i * 0.05) + rng.gauss(0, 0.2) for i in range(size)]
    kernel = [0.05, 0.15, 0.6, 0.15, 0.05]
    return data, kernel


def run_baseline(workload):
    data, kernel = workload
    filtered = apply_kernel(data, kernel)
    normed = normalize_minmax(filtered)
    feats = pairwise_features(normed, filtered)
    return filtered, normed, feats


def summarize(outputs):
    filtered, normed, feats = outputs
    mean_f = sum(feats) / len(feats) if feats else 0.0
    return sum(normed), mean_f, sum(v * v for v in feats) ** 0.5


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
