"""
Вариант 1.
"""
import random

def build_and_copy(n: int) -> tuple[int, float]:
    matrix = [[random.random() for _ in range(50)] for _ in range(n)]
    m1 = [row[:] for row in matrix]
    m2 = [list(row) for row in m1]
    selected = []
    for i, row in enumerate(m2):
        if i % 3 == 0:
            selected.append(row[:])
    s = 0.0
    for row in selected:
        for x in row:
            s += x
    return len(selected), s

def run_task(n: int):
    return build_and_copy(n)

if __name__ == "__main__":
    random.seed(42)
    for n in [5_000, 10_000, 20_000]:
        res = run_task(n)
        print(f"N={n:>7}  result={res}")
