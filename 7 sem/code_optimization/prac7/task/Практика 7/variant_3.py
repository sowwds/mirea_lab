"""
Вариант 3.
"""
import random
from typing import List

DIM = 40
SIZES = [500, 1500, 4000, 10000, 20000]
TINY = 2


def mean_center(data: List[List[float]]) -> List[List[float]]:
    n, d = len(data), len(data[0])
    means = [0.0] * d
    for row in data:
        for j in range(d):
            means[j] += row[j]
    means = [m / n for m in means]
    return [[row[j] - means[j] for j in range(d)] for row in data]


def covariance(data: List[List[float]]) -> List[List[float]]:
    n, d = len(data), len(data[0])
    cov = [[0.0] * d for _ in range(d)]
    for i in range(d):
        for j in range(i, d):
            s = sum(data[k][i] * data[k][j] for k in range(n))
            cov[i][j] = cov[j][i] = s / (n - 1)
    return cov


def project(data: List[List[float]], directions: List[List[float]]) -> List[List[float]]:
    return [[sum(row[j] * dr[j] for j in range(len(row))) for dr in directions] for row in data]


def make_workload(size: int):
    rng = random.Random(13)
    data = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(size)]
    directions = [[rng.gauss(0, 1) for _ in range(DIM)] for _ in range(5)]
    return data, directions


def run_baseline(workload):
    data, directions = workload
    centered = mean_center(data)
    return covariance(centered), project(centered, directions)


def summarize(outputs):
    cov, proj = outputs
    cov_norm = sum(v * v for row in cov for v in row) ** 0.5
    proj_norm = sum(v * v for row in proj for v in row) ** 0.5
    return cov_norm, proj_norm


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
