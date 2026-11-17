import time
import random


def fib(n: int) -> int:
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)


def deep_sum(d: dict) -> float:
    total = 0.0
    if isinstance(d, dict):
        for key in d:
            val = d[key]
            if isinstance(val, (int, float)):
                total += val
            elif isinstance(val, (dict, list)):
                total += deep_sum(val)
    elif isinstance(d, list):
        for item in d:
            if isinstance(item, (int, float)):
                total += item
            elif isinstance(item, (dict, list)):
                total += deep_sum(item)
    return total


def build_dict(depth: int, breadth: int) -> dict:
    if depth <= 0:
        return {f"v{i}": random.random() for i in range(breadth)}
    return {
        f"node{i}": build_dict(depth - 1, breadth)
        for i in range(breadth)
    }


def run_task(n: int):
    fib_n = min(n // 50 + 20, 32)
    fib_val = fib(fib_n)

    depth = min(4, n // 200 + 2)
    breadth = min(5, n // 100 + 2)
    nested = build_dict(depth, breadth)
    s = deep_sum(nested)
    return fib_val, round(s, 5)


if __name__ == "__main__":
    random.seed(42)
    for n in [200, 400, 600]:
        start = time.perf_counter()
        res = run_task(n)
        elapsed = time.perf_counter() - start
        print(f"N={n:4d}  result={res}  time={elapsed:.4f}s")
