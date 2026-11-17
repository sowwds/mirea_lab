"""Замеры памяти и времени. Запуск: python benchmark.py."""
import gc
import os
from pathlib import Path
import random
import statistics
import time
import tracemalloc

os.environ.setdefault("MPLCONFIGDIR", "/tmp/code-optimization-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from memory_profiler import profile

import variant_9 as optimized
import variant_9_baseline as baseline

ASSETS = Path(__file__).resolve().parent / "assets"
SIZES = [1000, 5000, 20000, 40000, 80000]


def measure_memory(function, n):
    gc.collect()
    random.seed(42)
    tracemalloc.start()
    result = function(n)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return current, peak, result


def measure_time(function, n):
    samples = []
    for _ in range(5):
        gc.collect()
        random.seed(42)
        start = time.perf_counter()
        function(n)
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main():
    ASSETS.mkdir(exist_ok=True)
    before_peaks, after_peaks = [], []
    print("| N | Пик до, МиБ | Пик после, КиБ | Остаток до / после, КиБ | Снижение пика | До, с | После, с |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for n in SIZES:
        before_current, before_peak, before_result = measure_memory(baseline.run_task, n)
        after_current, after_peak, after_result = measure_memory(optimized.run_task, n)
        assert before_result == after_result
        before_time = measure_time(baseline.run_task, n)
        after_time = measure_time(optimized.run_task, n)
        before_peaks.append(before_peak / 2**20)
        after_peaks.append(after_peak / 2**20)
        print(f"| {n} | {before_peak / 2**20:.3f} | {after_peak / 1024:.3f} | "
              f"{before_current / 1024:.3f} / {after_current / 1024:.3f} | "
              f"{before_peak / after_peak:.0f}× | {before_time:.6f} | {after_time:.6f} |", flush=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(SIZES, before_peaks, "o-", label="Исходная версия")
    ax.plot(SIZES, after_peaks, "o-", label="Оптимизированная версия")
    ax.set(xlabel="N", ylabel="Пиковая память Python, МиБ", title="Практика 4, вариант 9")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "memory_comparison.png", dpi=160)
    plt.close(fig)

    for name, function in [("before", baseline.repeated_data), ("after", optimized.repeated_data)]:
        gc.collect()
        random.seed(42)
        with (ASSETS / f"profile_memory_{name}.txt").open("w") as stream:
            profile(function, stream=stream, precision=4)(SIZES[-1])
    print("Построчные профили memory_profiler сохранены в assets.")


if __name__ == "__main__":
    main()
