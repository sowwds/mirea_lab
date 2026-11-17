"""Четыре модели: прогрев, пять повторений, медиана и min–max."""
import os
from pathlib import Path
import pickle
import platform
import statistics
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/code-optimization-mpl")
from variant_9_baseline import stage_fetch, stage_cpu, stage_post, _make_numbers, run_sequential
from variant_9 import run_threads, run_processes, run_async, WORKERS

ASSETS = Path(__file__).resolve().parent / "assets"
# 4, 16, 32 задания соответствуют SCALES 1, 4, 8; 1 и 64 — дополнительные.
SIZES = [1, 4, 16, 32, 64]
MODELS = [("Последовательно", run_sequential, 1), ("Потоки", run_threads, WORKERS),
          ("Процессы", run_processes, WORKERS), ("asyncio + процессы", run_async, WORKERS)]


def stage_times(jobs):
    times = [0.0, 0.0, 0.0]
    state = 0.0
    for job in jobs:
        start = time.perf_counter()
        data = stage_fetch(job)
        times[0] += time.perf_counter() - start
        start = time.perf_counter()
        value = stage_cpu(data)
        times[1] += time.perf_counter() - start
        start = time.perf_counter()
        state = stage_post(value, state)
        times[2] += time.perf_counter() - start
    return times


def main():
    ASSETS.mkdir(exist_ok=True)
    print(f"Python {sys.version.split()[0]}; {platform.platform()}; GIL={sys._is_gil_enabled()}", flush=True)
    print(f"spawn; workers={WORKERS}; объём списка 12000 int в pickle={len(pickle.dumps(_make_numbers(0)))} байт", flush=True)
    # Общий прогрев всех моделей перед оценкой этапов.
    for _, function, _ in MODELS:
        function(list(range(4)))
    print("\n## Доли этапов и предварительная оценка Амдала", flush=True)
    print("| Заданий | Fetch, с | CPU, с | Post, с | Fetch, % | CPU, % | Post, % | S2 | S4 | S8 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for n in SIZES:
        jobs = list(range(n))
        stage_times(jobs)  # первый запуск не учитываем
        samples = [stage_times(jobs) for _ in range(5)]
        fetch, cpu, post = [statistics.median(s[i] for s in samples) for i in range(3)]
        total = fetch + cpu + post
        f = post / total
        bounds = [1 / (f + (1-f)/p) for p in (2, 4, 8)]
        print(f"| {n} | {fetch:.4f} | {cpu:.4f} | {post:.4f} | {100*fetch/total:.1f} | "
              f"{100*cpu/total:.1f} | {100*f:.1f} | {bounds[0]:.2f} | {bounds[1]:.2f} | {bounds[2]:.2f} |", flush=True)
    print("\n## Модели", flush=True)
    print("| Заданий | Модель | Медиана, с | Min–max, с | S | E |")
    print("| --- | --- | --- | --- | --- | --- |")
    results = []
    for n in SIZES:
        jobs = list(range(n))  # вход готов до таймера
        expected = None
        # Сначала прогреваем все модели, только затем измеряем.
        for _, function, _ in MODELS:
            result = function(jobs)
            if expected is None:
                expected = result
            assert result[1] == expected[1] and abs(result[0] - expected[0]) < 1e-9
        samples = {label: [] for label, _, _ in MODELS}
        for repeat in range(5):
            # Меняем порядок, чтобы не закреплять прогрев и частоту CPU за одной моделью.
            offset = repeat % len(MODELS)
            order = MODELS[offset:] + MODELS[:offset]
            for label, function, _ in order:
                start = time.perf_counter()
                result = function(jobs)
                samples[label].append(time.perf_counter() - start)
                assert result[1] == expected[1] and abs(result[0] - expected[0]) < 1e-9
        baseline_time = statistics.median(samples[MODELS[0][0]])
        for label, _, workers in MODELS:
            elapsed = statistics.median(samples[label])
            speedup = baseline_time / elapsed
            print(f"| {n} | {label} | {elapsed:.4f} | {min(samples[label]):.4f}–{max(samples[label]):.4f} | "
                  f"{speedup:.2f} | {speedup/workers:.2f} |", flush=True)
            results.append((n, label, elapsed, speedup))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for column, ylabel, filename in [(2, "Время, с", "time_comparison.png"),
                                     (3, "Ускорение T₁ / Tₚ", "speedup_comparison.png")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for label, _, _ in MODELS:
            rows = [r for r in results if r[1] == label]
            ax.plot([r[0] for r in rows], [r[column] for r in rows], "o-", label=label)
        ax.set(xlabel="Число заданий", ylabel=ylabel, title="Практика 6, вариант 9")
        ax.legend()
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(ASSETS / filename, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
