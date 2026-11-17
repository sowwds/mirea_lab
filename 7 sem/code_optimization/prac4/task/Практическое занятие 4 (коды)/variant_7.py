"""
Вариант 7.
"""
import random

def process_with_history(n: int) -> tuple[int, float]:
    history = []
    index = {}            
    total = 0.0
    for i in range(n):
        value = random.random()
        record = {
            "id": i,
            "value": value,
            "payload": [random.random() for _ in range(20)],
        }
        history.append(record)
        index[i] = record
        if value > 0.7:
            total += value
    return len(history), total

def run_task(n: int):
    return process_with_history(n)

if __name__ == "__main__":
    random.seed(42)
    for n in [15_000, 30_000, 60_000]:
        res = run_task(n)
        print(f"N={n:>7}  result={res}")
