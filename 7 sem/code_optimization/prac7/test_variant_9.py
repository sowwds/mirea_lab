import numpy as np
import pytest
from numba import set_num_threads
from variant_9_baseline import make_workload, run_baseline, TINY, SIZES
from variant_9 import run_numpy, run_numba, run_parallel, run_processes, THREADS, prepare

FUNCTIONS = [run_numpy, run_numba, run_parallel, run_processes]


def equal(expected, actual):
    assert len(actual) == 3
    for left, right in zip(expected, actual):
        np.testing.assert_allclose(left, right, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("size", [TINY, SIZES[0], SIZES[1]])
def test_full_outputs(function, size):
    workload = make_workload(size)
    equal(run_baseline(workload), function(workload))


@pytest.mark.parametrize("function", FUNCTIONS)
def test_repeatability(function):
    workload = make_workload(40)
    first = function(workload)
    for _ in range(2):
        for a, b in zip(first, function(workload)):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("workload", [([2.0] * 5, [0.05, 0.15, 0.6, 0.15, 0.05]),
                                       ([3.0] * 40, [1.0]),
                                       ([1.0, -2.0, 4.0, 8.0], [2.0, -1.0]),
                                       ([1.0, 2.0], [])])
def test_boundaries(function, workload):
    equal(run_baseline(workload), function(workload))


@pytest.mark.parametrize("function", [run_baseline] + FUNCTIONS)
def test_too_short(function):
    with pytest.raises(ValueError):
        function(([1.0], [1.0, 2.0]))


@pytest.mark.parametrize("function", [run_numpy, run_numba, run_parallel])
def test_arrays_and_input_preserved(function):
    source = np.arange(80, dtype=np.float32)[::2]
    kernel = np.array([0.2, 0.8], dtype=np.float32)
    original = source.copy()
    prepared = prepare((source, kernel))
    assert all(a.dtype == np.float64 and a.flags.c_contiguous for a in prepared)
    actual = function((source, kernel))
    equal(run_baseline((source.tolist(), kernel.tolist())), actual)
    np.testing.assert_array_equal(source, original)
    assert all(a.dtype == np.float64 and a.flags.c_contiguous for a in actual)


@pytest.mark.parametrize("threads", [1, 2, THREADS])
def test_thread_count(threads):
    workload = make_workload(100)
    try:
        set_num_threads(threads)
        equal(run_baseline(workload), run_parallel(workload))
    finally:
        set_num_threads(THREADS)


def test_task_summary():
    from variant_9 import run_task
    from variant_9_baseline import summarize
    expected = summarize(run_baseline(make_workload(TINY)))
    actual, elapsed = run_task(TINY)
    np.testing.assert_allclose(expected, actual, rtol=1e-11, atol=1e-12)
    assert elapsed >= 0
