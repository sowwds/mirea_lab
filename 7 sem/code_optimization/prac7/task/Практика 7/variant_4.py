"""
Вариант 4.
"""
import random
from typing import List

SIZES = [100, 200, 400, 800, 1200]
TINY = 4


def conv1d(signal: List[float], kernel: List[float]) -> List[float]:
    n, k = len(signal), len(kernel)
    out = []
    for i in range(n - k + 1):
        s = 0.0
        for j in range(k):
            s += signal[i + j] * kernel[j]
        out.append(s)
    return out


def conv2d(image: List[List[float]], kernel: List[List[float]]) -> List[List[float]]:
    h, w = len(image), len(image[0])
    kh, kw = len(kernel), len(kernel[0])
    out = []
    for i in range(h - kh + 1):
        row = []
        for j in range(w - kw + 1):
            s = 0.0
            for di in range(kh):
                for dj in range(kw):
                    s += image[i + di][j + dj] * kernel[di][dj]
            row.append(s)
        out.append(row)
    return out


def make_workload(size: int):
    rng = random.Random(17)
    signal = [rng.gauss(0, 1) for _ in range(size * 8)]
    kernel1 = [0.1, 0.2, 0.4, 0.2, 0.1]
    image = [[rng.random() for _ in range(size)] for _ in range(size)]
    kernel2 = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    return signal, kernel1, image, kernel2


def run_baseline(workload):
    signal, k1, image, k2 = workload
    return conv1d(signal, k1), conv2d(image, k2)


def summarize(outputs):
    c1, c2 = outputs
    return sum(v * v for v in c1), sum(v * v for row in c2 for v in row)


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
