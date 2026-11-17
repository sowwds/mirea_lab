"""Замеры времени и профилирование. Запуск: python benchmark.py."""
import cProfile
import gc
import io
import os
from pathlib import Path
import pstats
import random
import statistics
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/code-optimization-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from line_profiler import LineProfiler

import variant_9 as optimized
import variant_9_baseline as baseline

ASSETS = Path(__file__).resolve().parent / "assets"
SIZES = [1000, 2000, 4000, 8000, 12000]


def measure_time(function, events):
    samples = []
    for _ in range(5):
        gc.collect()
        start = time.perf_counter()
        function(events, "click")
        samples.append(time.perf_counter() - start)
    return statistics.median(samples)


def main():
    ASSETS.mkdir(exist_ok=True)
    before_times, after_times = [], []
    print("| N | До, с | После, с | Ускорение |")
    print("| --- | --- | --- | --- |")
    for n in SIZES:
        random.seed(42)
        events = baseline.generate_events(n)
        assert baseline.process_events(events, "click") == optimized.process_events(events, "click")
        before = measure_time(baseline.process_events, events)
        after = measure_time(optimized.process_events, events)
        before_times.append(before)
        after_times.append(after)
        print(f"| {n} | {before:.6f} | {after:.6f} | {before / after:.2f}× |")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(SIZES, before_times, "o-", label="Исходная версия")
    ax.plot(SIZES, after_times, "o-", label="Оптимизированная версия")
    ax.set(xlabel="N", ylabel="Время, с", title="Практика 3, вариант 9")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS / "time_comparison.png", dpi=160)
    plt.close(fig)

    profiler = cProfile.Profile()
    profiler.runcall(baseline.process_events, events, "click")
    output = io.StringIO()
    pstats.Stats(profiler, stream=output).strip_dirs().sort_stats("cumulative").print_stats(10)
    (ASSETS / "profile_cprofile.txt").write_text(output.getvalue())

    lines = LineProfiler()
    lines.add_function(baseline.process_events)
    lines.runcall(baseline.process_events, events, "click")
    with (ASSETS / "profile_lines.txt").open("w") as stream:
        lines.print_stats(stream=stream)


if __name__ == "__main__":
    main()
