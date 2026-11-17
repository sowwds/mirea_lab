"""
Вариант 4.
"""
import random

def load_and_process(n: int) -> tuple[int, float]:
    lines = []
    for i in range(n):
        line = f"{i},{random.random():.6f},{random.randint(0,100)},msg_{i%50}"
        lines.append(line)
    rows = []
    for line in lines:
        parts = line.split(",")
        rows.append({
            "id": int(parts[0]),
            "value": float(parts[1]),
            "code": int(parts[2]),
            "msg": parts[3],
        })
    filtered = [r for r in rows if r["value"] > 0.4]
    s = sum(r["value"] for r in filtered)
    return len(filtered), s

def run_task(n: int):
    return load_and_process(n)

if __name__ == "__main__":
    random.seed(42)
    for n in [50_000, 100_000, 200_000]:
        res = run_task(n)
        print(f"N={n:>8}  result={res}")
