"""Полный эксперимент: время, cold/warm, память, графики в assets."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/code-optimization-mpl")
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import numpy as np
import numba
import psutil
from variant_9_baseline import make_workload, run_baseline, SIZES, TINY
from variant_9 import prepare, run_numpy, run_numba, run_parallel, run_processes, THREADS

ASSETS = Path(__file__).resolve().parent / "assets"
METHODS = {"Python": run_baseline, "NumPy": run_numpy, "Numba": run_numba,
           "prange": run_parallel, "Процессы": run_processes}
ACCELERATED = {"NumPy", "Numba", "prange"}


def check(left, right):
    for a, b in zip(left, right):
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-12)


def measure(function, workload, repeats=5):
    start = time.perf_counter()
    result = function(workload)  # прогрев не учитывается
    pilot = time.perf_counter() - start
    del result
    batch = min(1000, max(1, int(0.02 / max(pilot, 1e-9))))
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(batch):
            result = function(workload)
            del result
        samples.append((time.perf_counter() - start) / batch)
    return statistics.median(samples), min(samples), max(samples)


def worker(mode, name, size):
    if mode == "--cold":
        workload = prepare(make_workload(TINY))
        start = time.perf_counter()
        result = METHODS[name](workload)
        cold = time.perf_counter() - start
        warm = measure(METHODS[name], workload)
        print(cold, *warm)
    elif name == "probe":
        before = psutil.Process().memory_info().rss
        block = np.ones(8 * 1024 * 1024, dtype=np.float64)
        after = psutil.Process().memory_info().rss
        assert block.sum() > 0 and after - before >= 48 * 1024**2
        print("PROBE", before, after)
    else:
        workload = make_workload(size)
        if name in {"Numba", "prange"}:
            METHODS[name](prepare(make_workload(TINY)))
        result = METHODS[name](workload)
        print("DONE", flush=True)
        time.sleep(0.1)  # результат остаётся жив для внешнего счётчика RSS
        assert len(result) == 3


def peak_rss(name, size):
    proc = subprocess.Popen([sys.executable, "-B", __file__, "--memory", name, str(size)],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    root = psutil.Process(proc.pid)
    peak = 0
    while proc.poll() is None:
        try:
            processes = [root] + root.children(recursive=True)
            current = 0
            for child in processes:
                try:
                    current += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, current)
        except psutil.NoSuchProcess:
            pass
        time.sleep(0.005)
    out, err = proc.communicate()
    if proc.returncode:
        raise RuntimeError(err)
    return peak / 1024**2


def main():
    ASSETS.mkdir(exist_ok=True)
    print(f"Python {platform.python_version()}; NumPy {np.__version__}; Numba {numba.__version__}", flush=True)
    print(f"{platform.platform()}; cores={psutil.cpu_count(logical=False)}/{psutil.cpu_count()}; "
          f"RAM={psutil.virtual_memory().total / 1024**3:.2f} GiB; Numba threads={THREADS}; BLAS threads=1", flush=True)
    probe = subprocess.check_output([sys.executable, "-B", __file__, "--memory", "probe", "0"], text=True)
    print(probe.strip(), flush=True)
    cold_rows = []
    print("\n## Cold/warm на TINY (отдельные свежие процессы)")
    print("| Метод | Cold, мс | Warm, мс | Warm min–max, мс |")
    print("| --- | --- | --- | --- |")
    for name in ["Numba", "prange"]:
        text = subprocess.check_output([sys.executable, "-B", __file__, "--cold", name, "0"], text=True)
        cold, warm, low, high = map(float, text.split())
        cold_rows.append((name, cold, warm))
        print(f"| {name} | {cold*1000:.4f} | {warm*1000:.6f} | {low*1000:.6f}–{high*1000:.6f} |", flush=True)
    # Cold измерен отдельно, здесь JIT прогревается до замеров.
    run_numba(prepare(make_workload(TINY)))
    run_parallel(prepare(make_workload(TINY)))
    rows, memories = [], []
    for size in [TINY] + SIZES:
        workload = make_workload(size)
        arrays = prepare(workload)
        expected = run_baseline(workload)
        for name, function in METHODS.items():
            check(expected, function(workload))
        del expected
        measured = {}
        for name, function in METHODS.items():
            if name in ACCELERATED:
                core = measure(function, arrays)
                end = measure(function, workload)
            else:
                core = end = measure(function, workload)
            measured[name] = (core, end)
        print(f"\n## N={size}", flush=True)
        print("| Метод | Ядро, мс | Min–max ядра, мс | E2E, мс | Min–max E2E, мс | S ядра | S E2E |")
        print("| --- | --- | --- | --- | --- | --- | --- |")
        base = measured["Python"][0][0]
        for name, (core, end) in measured.items():
            print(f"| {name} | {core[0]*1000:.4f} | {core[1]*1000:.4f}–{core[2]*1000:.4f} | "
                  f"{end[0]*1000:.4f} | {end[1]*1000:.4f}–{end[2]*1000:.4f} | {base/core[0]:.3g} | {base/end[0]:.3g} |", flush=True)
            rows.append((size, name, core[0], end[0], base/core[0], base/end[0]))
        relative = measured['Numba'][0][0] / measured['prange'][0][0]
        print(f"prange относительно Numba: S={relative:.4f}, E={relative/THREADS:.4f}", flush=True)
        print("| Метод | Пик RSS, МиБ |")
        print("| --- | --- |")
        for name in METHODS:
            memory = peak_rss(name, size)
            memories.append((size, name, memory))
            print(f"| {name} | {memory:.3f} |", flush=True)
        del workload, arrays
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for columns, labels, filename in [((2,3), ("Ядро, мс", "End-to-end, мс"), "time_comparison.png"),
                                      ((4,5), ("Ускорение ядра", "Ускорение end-to-end"), "speedup_comparison.png")]:
        fig, axes = plt.subplots(1, 2, figsize=(11,4.3))
        for ax, column, label in zip(axes, columns, labels):
            for name in METHODS:
                points = [r for r in rows if r[1] == name]
                scale = 1000 if column < 4 else 1
                ax.plot([r[0] for r in points], [r[column]*scale for r in points], "o-", label=name)
            ax.set(xlabel="N", ylabel=label, xscale="log", yscale="log")
            ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(ASSETS/filename, dpi=160); plt.close(fig)
    plot_memory(memories)
    fig, ax = plt.subplots(figsize=(7,4))
    x = np.arange(len(cold_rows))
    ax.bar(x-.18,[r[1]*1000 for r in cold_rows],width=.36,label="Cold")
    ax.bar(x+.18,[r[2]*1000 for r in cold_rows],width=.36,label="Warm")
    ax.set(xticks=x,xticklabels=[r[0] for r in cold_rows],ylabel="Время, мс",yscale="log")
    ax.legend();fig.tight_layout();fig.savefig(ASSETS/'cold_warm.png',dpi=160);plt.close(fig)


def plot_memory(memories):
    ASSETS.mkdir(exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,4.5))
    for name in METHODS:
        points = [r for r in memories if r[1] == name]
        ax.plot([r[0] for r in points], [r[2] for r in points], "o-", label=name)
    ax.set(xlabel="N", ylabel="Пик RSS, МиБ", xscale="log")
    ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
    fig.savefig(ASSETS/'memory_comparison.png', dpi=160); plt.close(fig)


def memory_only():
    memories = []
    for size in [TINY] + SIZES:
        print(f"\n## RSS N={size}", flush=True)
        print("| Метод | Пик RSS, МиБ |")
        print("| --- | --- |")
        for name in METHODS:
            value = peak_rss(name, size)
            memories.append((size, name, value))
            print(f"| {name} | {value:.3f} |", flush=True)
    plot_memory(memories)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--memory-only":
        memory_only()
    elif len(sys.argv) > 1:
        worker(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    else:
        main()
