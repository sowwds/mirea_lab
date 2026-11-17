"""Сравнение двух кэшей с исходником. Запуск: python benchmark.py."""
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
from variant_9_baseline import Analytics
from variant_9 import LruAnalytics, ManualAnalytics

ASSETS = Path(__file__).resolve().parent / "assets"
CLASSES = [("Без кэша", Analytics), ("LRU", LruAnalytics), ("Ручной LRU", ManualAnalytics)]
SIZES = [40, 60, 80]


def make_requests(n_days, pattern):
    rng = random.Random(123)
    if pattern == "cold":
        return [([d], [], None) for d in range(n_days)] + [([], [p], None) for p in range(80)]
    requests = []
    for i in range(40):
        day_range = range(5) if pattern in ("hot", "updates") else range(n_days)
        product_range = range(8) if pattern in ("hot", "updates") else range(80)
        days = rng.sample(day_range, 5)
        products = rng.sample(product_range, 8)
        update = (i % 5, i % 8, i) if pattern == "updates" and i % 5 == 0 else None
        requests.append((days, products, update))
    return requests + requests[:20] if pattern == "original" else requests


def execute(obj, requests):
    checksum = 0.0
    for days, products, update in requests:
        if update is not None:
            obj.update_sale(*update)
        checksum += obj.summary_report(days, products)["combo"]
    return checksum


def create(cls, n_days):
    random.seed(42)
    return cls(n_days, 80)


def measure(cls, n_days, requests):
    samples = []
    for _ in range(5):
        gc.collect()
        obj = create(cls, n_days)
        start = time.perf_counter()
        result = execute(obj, requests)
        samples.append(time.perf_counter() - start)
    gc.collect()
    obj = create(cls, n_days)
    tracemalloc.start()
    execute(obj, requests)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return statistics.median(samples), current, peak, result


def main():
    ASSETS.mkdir(exist_ok=True)
    plot_rows = []
    for pattern in ("original", "hot", "updates", "cold"):
        print(f"\n## {pattern}", flush=True)
        print("| Дней | Стратегия | Время, мс | Пик, КиБ | Удержано, КиБ | Ускорение |")
        print("| --- | --- | --- | --- | --- | --- |")
        for n_days in (SIZES if pattern == "original" else [60]):
            requests = make_requests(n_days, pattern)
            expected = None
            original_time = None
            for label, cls in CLASSES:
                elapsed, current, peak, result = measure(cls, n_days, requests)
                if expected is None:
                    expected, original_time = result, elapsed
                assert result == expected
                print(f"| {n_days} | {label} | {elapsed * 1000:.3f} | {peak / 1024:.3f} | "
                      f"{current / 1024:.3f} | {original_time / elapsed:.2f}× |", flush=True)
                if pattern == "original":
                    plot_rows.append((n_days, label, elapsed * 1000, peak / 1024))
    for column, ylabel, filename in [(2, "Время, мс", "time_comparison.png"),
                                     (3, "Пик выделений Python, КиБ", "memory_comparison.png")]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for label, _ in CLASSES:
            rows = [row for row in plot_rows if row[1] == label]
            ax.plot([r[0] for r in rows], [r[column] for r in rows], "o-", label=label)
        ax.set(xlabel="Число дней (80 товаров)", ylabel=ylabel, title="Практика 5, вариант 9")
        ax.legend()
        ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(ASSETS / filename, dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    main()
