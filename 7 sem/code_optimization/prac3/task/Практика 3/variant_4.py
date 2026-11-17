import time
import random


def matrix_multiply(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n = len(a)
    m = len(b[0])
    p = len(b)
    result = [[0.0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(p):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result


def row_stats(matrix: list[list[float]]) -> list[tuple[float, float, float]]:
    stats = []
    for row in matrix:
        mn = row[0]
        for x in row:
            if x < mn:
                mn = x
        mx = row[0]
        for x in row:
            if x > mx:
                mx = x
        total = 0.0
        for x in row:
            total += x
        mean = total / len(row)
        stats.append((mn, mx, mean))
    return stats


def generate_matrix(n: int) -> list[list[float]]:
    return [[random.random() for _ in range(n)] for _ in range(n)]


def run_task(n: int):
    a = generate_matrix(n)
    b = generate_matrix(n)
    product = matrix_multiply(a, b)
    stats = row_stats(product)
    return round(stats[0][2], 5), round(product[0][0], 5)


if __name__ == "__main__":
    random.seed(42)
    for n in [40, 60, 80]:
        start = time.perf_counter()
        res = run_task(n)
        elapsed = time.perf_counter() - start
        print(f"N={n:3d}  result={res}  time={elapsed:.4f}s")
