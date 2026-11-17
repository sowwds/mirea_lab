"""
Вариант 2.
"""
import random
from dataclasses import dataclass

class Point:
    def __init__(self, x, y, z, label):
        self.x = x
        self.y = y
        self.z = z
        self.label = label

def create_points(n: int) -> tuple[int, float]:
    points = []
    for i in range(n):
        p = Point(
            random.random(),
            random.random(),
            random.random(),
            f"point_{i}"
        )
        points.append(p)
    
    by_label = {}
    for p in points:
        by_label[p.label] = p
    coords = [(p.x, p.y, p.z) for p in points]
    s = sum(p.x + p.y + p.z for p in points)
    return len(points), s

def run_task(n: int):
    return create_points(n)

if __name__ == "__main__":
    random.seed(42)
    for n in [30_000, 60_000, 100_000]:
        result = run_task(n)
        print(f"N={n:>7}  result={result}")
