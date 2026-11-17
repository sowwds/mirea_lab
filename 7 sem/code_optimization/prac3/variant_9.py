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
    """Стабильная сортировка, один проход и первые пять шагов отбора."""
    selected = sorted(
        (event for event in events if event.event_type == event_type),
        key=lambda event: event.timestamp,
    )
    totals = {}
    for event in selected:
        totals[event.user_id] = totals.get(event.user_id, 0.0) + event.value

    top = list(totals.items())
    for i in range(min(5, len(top))):
        max_idx = i
        for j in range(i + 1, len(top)):
            if top[j][1] > top[max_idx][1]:
                max_idx = j
        top[i], top[max_idx] = top[max_idx], top[i]
    return {
        "total_selected": len(selected),
        "unique_users": len(totals),
        "top5": top[:5],
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
