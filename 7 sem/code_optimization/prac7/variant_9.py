"""Вариант 9: NumPy, Numba и процессы для обработки сигнала."""
from array import array
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads
from variant_9_baseline import WINDOW, make_workload, run_baseline

THREADS = min(4, get_num_threads())
set_num_threads(THREADS)


def prepare(workload):
    return tuple(np.ascontiguousarray(part, dtype=np.float64) for part in workload)


def run_numpy(workload):
    data, kernel = prepare(workload)
    if kernel.size == 0:
        filtered = np.zeros(data.size + 1, dtype=np.float64)
    else:
        if data.size < kernel.size:
            raise ValueError("Нет элементов после фильтрации")
        filtered = np.correlate(data, kernel, mode="valid")
    lo, hi = filtered.min(), filtered.max()
    normed = np.zeros_like(filtered) if lo == hi else (filtered - lo) / (hi - lo)
    n = filtered.size
    w = min(WINDOW - 1, n - 1)
    feats = np.empty(2 * (w * n - w * (w + 1) // 2), dtype=np.float64)
    # Представление скользящих окон не копирует исходный массив.
    full = max(0, n - WINDOW + 1)
    if full:
        ys = np.lib.stride_tricks.sliding_window_view(filtered, WINDOW)
        xs = np.lib.stride_tricks.sliding_window_view(normed, WINDOW)
        head = feats[:2 * full * (WINDOW - 1)].reshape(full, WINDOW - 1, 2)
        head[:, :, 0] = normed[:full, None] - xs[:, 1:]
        head[:, :, 1] = normed[:full, None] * ys[:, 1:]
    offset = 2 * full * (WINDOW - 1)
    # Только короткий хвост (не больше 29 строк) имеет окна разной длины.
    for i in range(full, n):
        end = min(i + WINDOW, n)
        count = end - i - 1
        block = feats[offset:offset + 2 * count].reshape(count, 2)
        block[:, 0] = normed[i] - normed[i + 1:end]
        block[:, 1] = normed[i] * filtered[i + 1:end]
        offset += 2 * count
    return filtered, normed, feats


@njit
def _filter_normalize(data, kernel):
    n = len(data) - len(kernel) + 1
    if n <= 0:
        raise ValueError("Нет элементов после фильтрации")
    filtered = np.empty(n, dtype=np.float64)
    for i in range(n):
        total = 0.0
        for j in range(len(kernel)):
            total += data[i + j] * kernel[j]
        filtered[i] = total
    lo, hi = filtered.min(), filtered.max()
    normed = np.empty(n, dtype=np.float64)
    for i in range(n):
        normed[i] = 0.0 if hi == lo else (filtered[i] - lo) / (hi - lo)
    return filtered, normed


@njit
def _numba_core(data, kernel):
    filtered, normed = _filter_normalize(data, kernel)
    n = len(filtered)
    w = min(WINDOW - 1, n - 1)
    feats = np.empty(2 * (w * n - w * (w + 1) // 2), dtype=np.float64)
    offset = 0
    for i in range(n):
        for j in range(i + 1, min(i + WINDOW, n)):
            feats[offset] = normed[i] - normed[j]
            feats[offset + 1] = normed[i] * filtered[j]
            offset += 2
    return filtered, normed, feats


@njit(parallel=True)
def _parallel_core(data, kernel):
    filtered, normed = _filter_normalize(data, kernel)
    n = len(filtered)
    w = min(WINDOW - 1, n - 1)
    full = n - w
    feats = np.empty(2 * (w * n - w * (w + 1) // 2), dtype=np.float64)
    for row in prange(n):
        i = np.int64(row)
        tail = max(0, i - full)
        # Номер первой пары строки i: полный прямоугольник и короткий хвост.
        offset = 2 * (min(i, full) * w + tail * (w - 1) - tail * (tail - 1) // 2)
        for j in range(i + 1, min(i + WINDOW, n)):
            feats[offset] = normed[i] - normed[j]
            feats[offset + 1] = normed[i] * filtered[j]
            offset += 2
    return filtered, normed, feats


def run_numba(workload):
    return _numba_core(*prepare(workload))


def run_parallel(workload):
    return _parallel_core(*prepare(workload))


def _features_chunk(task):
    xs, ys, owned = task
    feats = array('d')
    for i in range(owned):
        for j in range(i + 1, min(i + WINDOW, len(xs))):
            feats.append(xs[i] - xs[j])
            feats.append(xs[i] * ys[j])
    return feats


def run_processes(workload):
    from variant_9_baseline import apply_kernel, normalize_minmax
    data, kernel = workload
    filtered = apply_kernel(data, kernel)
    normed = normalize_minmax(filtered)
    n = len(filtered)
    chunk = max(1, (n + THREADS - 1) // THREADS)
    # Окно соседних элементов за правой границей порции необходимо для её последних строк.
    tasks = [(normed[start:min(start + chunk + WINDOW - 1, n)],
              filtered[start:min(start + chunk + WINDOW - 1, n)], min(chunk, n - start))
             for start in range(0, n, chunk)]
    feats = array('d')
    with ProcessPoolExecutor(max_workers=THREADS, mp_context=get_context("spawn")) as pool:
        for part in pool.map(_features_chunk, tasks):
            feats.extend(part)
    return filtered, normed, feats


def run_task(size):
    import time
    workload = make_workload(size)
    start = time.perf_counter()
    outputs = run_numba(workload)
    elapsed = time.perf_counter() - start
    _, normed, feats = outputs
    # В отличие от списка, ndarray нельзя проверять условием «if feats».
    summary = (float(sum(normed)), float(sum(feats) / len(feats)) if len(feats) else 0.0,
               float(sum(value * value for value in feats) ** 0.5))
    return summary, elapsed


if __name__ == "__main__":
    from variant_9_baseline import TINY
    for size in [TINY, 16000]:
        result, elapsed = run_task(size)
        print(f"size={size} result={result} time={elapsed:.6f}s")
