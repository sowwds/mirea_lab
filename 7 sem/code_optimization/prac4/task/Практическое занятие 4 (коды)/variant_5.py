"""
Вариант 5.
"""
import random

def transform_all(n: int) -> tuple[int, float]:
    raw = [random.gauss(0, 1) for _ in range(n)]
    step1 = [x * 1.5 for x in raw]
    step2 = [x + 0.1 for x in step1 if x > -1]
    step3 = [abs(x) for x in step2]
    step4 = sorted(step3)
    result = step4[::2]
    return len(result), sum(result)

def run_task(n: int):
    return transform_all(n)

if __name__ == "__main__":
    random.seed(42)
    for n in [100_000, 200_000, 400_000]:
        res = run_task(n)
        print(f"N={n:>8}  result={res}")
