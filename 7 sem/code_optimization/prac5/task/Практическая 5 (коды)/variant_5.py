"""
Вариант 5. 
"""
import random
import time

class PathCounter:
    def __init__(self, obstacle_factor: float = 1.0):
        self.obstacle_factor = obstacle_factor

    def paths(self, x: int, y: int) -> float:
        if x < 0 or y < 0:
            return 0.0
        if x == 0 and y == 0:
            return self.obstacle_factor
        return (self.paths(x - 1, y) + self.paths(x, y - 1)) * self.obstacle_factor

    def set_factor(self, f: float) -> None:
        self.obstacle_factor = f


def run_task(size: int = 12, queries: int = 60):
    random.seed(42)
    pc = PathCounter(1.0)
    pts = [(random.randint(5, size), random.randint(5, size)) for _ in range(queries)]
    pts = pts + pts[: queries // 2]

    t0 = time.perf_counter()
    s = sum(pc.paths(x, y) for x, y in pts)
    t1 = time.perf_counter()

    pc.set_factor(0.95)
    s2 = sum(pc.paths(x, y) for x, y in pts[:15])
    return round(s, 2), round(s2, 2), round(t1 - t0, 4)


if __name__ == "__main__":
    for sz in [10, 11, 12]:
        print(f"size={sz}  {run_task(size=sz)}")
