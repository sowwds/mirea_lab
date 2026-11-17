"""
Вариант 7.
"""
import math
import random
from array import array
from typing import List

STEPS = 100
HALF_LIFE = 5.0
DT = 0.1
BINS, LO, HI = 40, -8.0, 8.0
SIZES = [6000, 12000, 24000, 48000, 96000]
TINY = 2


def simulate_decay(draws, n_particles: int, steps: int, half_life: float, dt: float) -> List[int]:
    p_decay = 1.0 - math.exp(-math.log(2) * dt / half_life)
    alive = [True] * n_particles
    n_alive = n_particles
    history = [n_alive]
    for s in range(steps):
        base = s * n_particles
        for k in range(n_particles):
            if alive[k] and draws[base + k] < p_decay:
                alive[k] = False
                n_alive -= 1
        history.append(n_alive)
    return history


def histogram(values: List[float], bins: int, lo: float, hi: float) -> List[int]:
    counts = [0] * bins
    width = (hi - lo) / bins
    for v in values:
        if lo <= v < hi:
            idx = int((v - lo) / width)
            if 0 <= idx < bins:
                counts[idx] += 1
    return counts


def make_workload(size: int):
    rng = random.Random(29)
    draws = array('d', (rng.random() for _ in range(size * STEPS)))
    samples = [rng.gauss(0, 2) for _ in range(size * 20)]
    return draws, size, samples


def run_baseline(workload):
    draws, n_particles, samples = workload
    hist = histogram(samples, BINS, LO, HI)
    history = simulate_decay(draws, n_particles, STEPS, HALF_LIFE, DT)
    return hist, history


def summarize(outputs):
    hist, history = outputs
    return sum(hist), history[-1], sum(history)


def run_task(size: int):
    import time
    workload = make_workload(size)
    t0 = time.perf_counter()
    outputs = run_baseline(workload)
    elapsed = time.perf_counter() - t0
    return summarize(outputs), elapsed


if __name__ == "__main__":
    for s in SIZES:
        res, t = run_task(s)
        print(f"size={s:<6} {res}  time={t:.4f}s")
