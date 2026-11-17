"""
Вариант 2.
"""
import random
from typing import List

Matrix = List[List[float]]

SIZES = [60, 120, 180, 250, 320]
TINY = 2


def matmul(A: Matrix, B: Matrix) -> Matrix:
    n, m, p = len(A), len(A[0]), len(B[0])
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        for k in range(m):
            aik = A[i][k]
            for j in range(p):
                C[i][j] += aik * B[k][j]
    return C


def frobenius_norm(M: Matrix) -> float:
    s = 0.0
    for row in M:
        for v in row:
            s += v * v
    return s ** 0.5


def matvec(A: Matrix, x: List[float]) -> List[float]:
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def make_workload(size: int):
    rng = random.Random(11)
    A = [[rng.gauss(0, 1) for _ in range(size)] for _ in range(size)]
    B = [[rng.gauss(0, 1) for _ in range(size)] for _ in range(size)]
    x = [rng.gauss(0, 1) for _ in range(size)]
    return A, B, x


def run_baseline(workload):
    A, B, x = workload
    C = matmul(A, B)
    return C, frobenius_norm(C), matvec(C, x)


def summarize(outputs):
    C, norm, y = outputs
    return norm, sum(v * v for v in y) ** 0.5


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
