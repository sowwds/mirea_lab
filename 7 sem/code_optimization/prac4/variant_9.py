"""Практическое занятие 4, вариант 9. Потоковая обработка."""
import random


def repeated_data(n: int) -> tuple[int, float]:
    total = sum(sum(random.random() for _ in range(5)) for _ in range(n))
    return len(range(n)), total


def run_task(n: int):
    return repeated_data(n)


if __name__ == "__main__":
    random.seed(42)
    for n in [20_000, 40_000, 80_000]:
        print(f"N={n:>7}  result={run_task(n)}")
