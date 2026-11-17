import random
import pytest
import variant_9 as optimized
import variant_9_baseline as baseline


@pytest.mark.parametrize("module", [baseline, optimized])
@pytest.mark.parametrize("n", [-3, 0])
def test_empty(module, n):
    assert module.run_task(n) == (0, 0)


@pytest.mark.parametrize("n", [-3, 0, 1, 10, 100, 1000, 20000])
def test_exact_result_and_random_state(n):
    random.seed(42)
    expected = baseline.run_task(n)
    state = random.getstate()
    random.seed(42)
    assert optimized.run_task(n) == expected
    assert random.getstate() == state
