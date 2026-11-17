import time
import random
from collections import defaultdict


def parse_and_count(logs: list[str]) -> dict:
    levels = ["INFO", "WARN", "ERROR", "DEBUG"]
    level_counts = {}
    for level in levels:
        count = 0
        for log in logs:
            parts = log.split()
            if parts and parts[0] == level:
                count += 1
        level_counts[level] = count
    hour_counts = {}
    for hour in range(24):
        count = 0
        for log in logs:
            parts = log.split()
            if len(parts) >= 2:
                try:
                    h = int(parts[1].split(":")[0])
                    if h == hour:
                        count += 1
                except (ValueError, IndexError):
                    pass
        hour_counts[hour] = count
    messages = []
    for log in logs:
        parts = log.split(maxsplit=2)
        if len(parts) >= 3 and len(parts[2]) > 40:
            messages.append(parts[2])

    return {
        "levels": level_counts,
        "hours": hour_counts,
        "long_count": len(messages),
    }


def generate_logs(n: int) -> list[str]:
    levels = ["INFO", "WARN", "ERROR", "DEBUG"]
    messages = [
        "User logged in successfully",
        "Failed to connect to database after multiple retries and timeout",
        "Cache miss for key",
        "Request processed",
        "Unexpected exception occurred while processing payment transaction details",
        "Heartbeat ok",
    ]
    logs = []
    for _ in range(n):
        level = random.choice(levels)
        h = random.randint(0, 23)
        m = random.randint(0, 59)
        s = random.randint(0, 59)
        msg = random.choice(messages)
        if random.random() < 0.3:
            msg = msg + " " + "x" * random.randint(10, 30)
        logs.append(f"{level} {h:02d}:{m:02d}:{s:02d} {msg}")
    return logs


def run_task(n: int):
    logs = generate_logs(n)
    result = parse_and_count(logs)
    return result["levels"]["ERROR"], result["long_count"]


if __name__ == "__main__":
    random.seed(42)
    for n in [2000, 5000, 10000]:
        start = time.perf_counter()
        res = run_task(n)
        elapsed = time.perf_counter() - start
        print(f"N={n:6d}  result={res}  time={elapsed:.4f}s")
