"""
Вариант 3.
"""
import random

class Record:
    def __init__(self, user_id, value, tags):
        self.user_id = user_id
        self.value = value
        self.tags = tags
        self.extra = {"src": "gen", "ts": random.randint(1, 10**9)}

def process_records(n: int) -> tuple[int, float]:
    records = []
    for i in range(n):
        tags = [f"t{j}" for j in range(random.randint(1, 5))]
        records.append(Record(i % 1000, random.random(), tags))
    
    high = [r for r in records if r.value > 0.5]
    by_user = {}
    for r in high:
        if r.user_id not in by_user:
            by_user[r.user_id] = []
        by_user[r.user_id].append(r)
    
    total = 0.0
    for uid, lst in by_user.items():
        for r in lst:
            total += r.value
    return len(high), total

def run_task(n: int):
    return process_records(n)

if __name__ == "__main__":
    random.seed(42)
    for n in [20_000, 40_000, 80_000]:
        res = run_task(n)
        print(f"N={n:>7}  result={res}")
