import time
import random
from dataclasses import dataclass


@dataclass
class Event:
    id: int
    user_id: int
    event_type: str
    value: float
    timestamp: int


def process_events(events: list[Event], event_type: str) -> dict:
    selected = []
    for e in events:
        if e.event_type == event_type:
            selected.append(e)
    n = len(selected)
    for i in range(n):
        for j in range(0, n - i - 1):
            if selected[j].timestamp > selected[j + 1].timestamp:
                selected[j], selected[j + 1] = selected[j + 1], selected[j]
    user_ids = []
    for e in selected:
        if e.user_id not in user_ids:
            user_ids.append(e.user_id)
    per_user = {}
    for uid in user_ids:
        vals = []
        for e in selected:
            if e.user_id == uid:
                vals.append(e.value)
        total = 0.0
        for v in vals:
            total += v
        per_user[uid] = {
            "count": len(vals),
            "sum": total,
            "avg": total / len(vals) if vals else 0.0,
        }
    top = []
    for uid, stats in per_user.items():
        top.append((uid, stats["sum"]))
    for i in range(len(top)):
        max_idx = i
        for j in range(i + 1, len(top)):
            if top[j][1] > top[max_idx][1]:
                max_idx = j
        top[i], top[max_idx] = top[max_idx], top[i]
    top5 = top[:5]
    return {
        "total_selected": len(selected),
        "unique_users": len(user_ids),
        "top5": top5,
    }


def generate_events(n: int) -> list[Event]:
    types = ["click", "view", "purchase", "login", "click", "view"]
    events = []
    for i in range(n):
        events.append(Event(
            id=i,
            user_id=random.randint(1, max(50, n // 20)),
            event_type=random.choice(types),
            value=round(random.uniform(0.1, 100.0), 2),
            timestamp=random.randint(1_000_000, 2_000_000),
        ))
    return events


def run_task(n: int):
    events = generate_events(n)
    result = process_events(events, event_type="click")
    return result["total_selected"], result["unique_users"]


if __name__ == "__main__":
    random.seed(42)
    for n in [1000, 2000, 4000]:
        start = time.perf_counter()
        res = run_task(n)
        elapsed = time.perf_counter() - start
        print(f"N={n:5d}  result={res}  time={elapsed:.4f}s")
