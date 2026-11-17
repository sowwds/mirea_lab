"""
Вариант 8.
"""
import asyncio
import random
import time
from typing import List, Tuple

SCALES = [1, 4, 8]


def _make_data(item_id: int) -> List[float]:
    rng = random.Random(item_id) 
    return [rng.random() for _ in range(12_000)]


def stage_load(item_id: int) -> List[float]:
    time.sleep(0.04)
    return _make_data(item_id)


async def stage_load_async(item_id: int) -> List[float]:
    await asyncio.sleep(0.04)
    return _make_data(item_id)


def stage_transform(data: List[float]) -> List[float]:
    out = []
    for x in data:
        y = x
        for _ in range(100):
            y = (y * 1.001 + 0.0001) % 1.0
        out.append(y)
    return out


def stage_reduce(data: List[float]) -> float:
    return sum(data) / len(data) if data else 0.0


def run_sequential(item_ids: List[int]) -> Tuple[float, int]:
    total = 0.0
    for iid in item_ids:
        raw = stage_load(iid)
        transformed = stage_transform(raw)
        total += stage_reduce(transformed)
    return total, len(item_ids)


def make_workload(scale: int = 4):
    return list(range(scale * 4))


def run_task(scale: int = 4):
    ids = make_workload(scale)
    t0 = time.perf_counter()
    result = run_sequential(ids)
    elapsed = time.perf_counter() - t0
    return round(result[0], 4), result[1], round(elapsed, 4)


if __name__ == "__main__":
    for s in SCALES:
        print(f"scale={s}  {run_task(s)}")
