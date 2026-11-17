import time
import random


def compute_moving_average(data: list[float], window: int) -> list[float]:
    result = []
    for i in range(len(data) - window + 1):
        s = 0.0
        for j in range(window):
            s += data[i + j]
        result.append(s / window)
    return result


def pairwise_distances(points: list[tuple[float, float]]) -> list[list[float]]:
    n = len(points)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dist[i][j] = (dx * dx + dy * dy) ** 0.5
    return dist


def generate_data(n: int):
    series = [random.gauss(0, 1) for _ in range(n)]
    points = [(random.random(), random.random()) for _ in range(n // 5)]
    return series, points


def run_task(n: int):
    series, points = generate_data(n)
    ma = compute_moving_average(series, window=20)
    dist = pairwise_distances(points)
    return round(ma[0], 5) if ma else 0.0, round(dist[0][1], 5) if len(dist) > 1 else 0.0


if __name__ == "__main__":
    random.seed(42)
    for n in [1000, 2000, 4000]:
        start = time.perf_counter()
        res = run_task(n)
        elapsed = time.perf_counter() - start
        print(f"N={n:5d}  result={res}  time={elapsed:.4f}s")
