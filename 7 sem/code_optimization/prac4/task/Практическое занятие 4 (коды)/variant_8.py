"""
Вариант 8.
"""
import random

def sparse_table(n: int) -> tuple[int, float]:
    table = []
    for i in range(n):
        row = [0.0] * 100
        for _ in range(3):
            idx = random.randint(0, 99)
            row[idx] = random.random()
        table.append(row)
    table2 = [row[:] for row in table]
    s = 0.0
    for row in table2:
        for x in row:
            s += x
    return len(table), s

def run_task(n: int):
    return sparse_table(n)

if __name__ == "__main__":
    random.seed(42)
    for n in [5_000, 10_000, 20_000]:
        res = run_task(n)
        print(f"N={n:>7}  result={res}")
