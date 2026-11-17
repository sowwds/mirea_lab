"""
Вариант 6.
"""
import random
from functools import lru_cache

@lru_cache(maxsize=None)
def expensive_calc(x: int) -> float:
    return sum(i * 0.001 for i in range(x % 500 + 100))

def run_with_growing_cache(n: int) -> tuple[int, float]:
    cache_demo = {}
    total = 0.0
    for i in range(n):
        key = i % (n // 2 + 10)
        if key not in cache_demo:
            cache_demo[key] = expensive_calc(key)
        total += cache_demo[key]
        cache_demo[f"hist_{i}"] = total
    return len(cache_demo), total

def run_task(n: int):
    return run_with_growing_cache(n)

if __name__ == "__main__":
    random.seed(42)
    for n in [10_000, 20_000, 40_000]:
        res = run_task(n)
        print(f"N={n:>7}  result={res}")
