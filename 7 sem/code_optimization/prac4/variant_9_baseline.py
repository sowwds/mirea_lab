"""
Вариант 9.
"""
import random

def repeated_data(n: int) -> tuple[int, float]:
    statuses = ["pending", "completed", "cancelled", "processing"]
    categories = ["A", "B", "C", "D", "E"]
    records = []
    for i in range(n):
        rec = {
            "id": i,
            "status": statuses[i % 4],
            "category": categories[i % 5],
            "description": f"Item number {i % 100} in category {categories[i % 5]}",
            "values": [random.random() for _ in range(5)],
        }
        records.append(rec)
    groups = {}
    for r in records:
        key = r["status"]
        if key not in groups:
            groups[key] = []
        groups[key].append(r.copy())
    total = sum(sum(r["values"]) for r in records)
    return len(records), total

def run_task(n: int):
    return repeated_data(n)

if __name__ == "__main__":
    random.seed(42)
    for n in [20_000, 40_000, 80_000]:
        res = run_task(n)
        print(f"N={n:>7}  result={res}")
