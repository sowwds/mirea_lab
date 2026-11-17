import pytest
from variant_9_baseline import run_sequential
from variant_9 import run_threads, run_processes, run_async


@pytest.mark.parametrize("jobs", [[], [0], [2, 0, 2]])
@pytest.mark.parametrize("function", [run_threads, run_processes, run_async])
def test_equivalence(function, jobs):
    expected = run_sequential(jobs)
    actual = function(jobs)
    assert actual[1] == expected[1]
    assert actual[0] == pytest.approx(expected[0], rel=1e-12, abs=1e-12)
