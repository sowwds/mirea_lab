"""
Вариант 0.
"""
import random
import sys

def process_sequence(n: int) -> tuple[int, float]:
    data = [random.random() for _ in range(n)]
    copy1 = data[:]
    copy2 = list(data)
    copy3 = [x for x in data]
    filtered = []
    for x in copy1:
        if x > 0.3:
            filtered.append(x)
    result = []
    for x in filtered:
        result.append(x * 2)
    total = 0.0
    for x in result:
        total += x
    return len(result), total

def run_task(n: int):
    return process_sequence(n)

if __name__ == "__main__":
    random.seed(42)
    for n in [50_000, 100_000, 200_000]:
        res = run_task(n)
        print(f"N={n:>8}  result={res}")
