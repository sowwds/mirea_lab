import random
import pytest
import variant_9 as optimized
import variant_9_baseline as baseline


@pytest.mark.parametrize("module", [baseline, optimized])
@pytest.mark.parametrize("events, expected", [
    ([], {"total_selected": 0, "unique_users": 0, "top5": []}),
    ([baseline.Event(0, 7, "view", 5.0, 0)],
     {"total_selected": 0, "unique_users": 0, "top5": []}),
    ([baseline.Event(0, 7, "click", 5.0, 0)],
     {"total_selected": 1, "unique_users": 1, "top5": [(7, 5.0)]}),
])
def test_known_results(module, events, expected):
    assert module.process_events(events, "click") == expected


@pytest.mark.parametrize("n", [0, 1, 10, 100, 1000])
def test_generated_events(n):
    random.seed(42)
    events = baseline.generate_events(n)
    original = list(events)
    assert optimized.process_events(events, "click") == baseline.process_events(events, "click")
    assert events == original


def test_selection_sort_ties():
    # После переноса максимума в начало исходная сортировка меняет порядок
    # равных элементов: пользователь 2 опережает пользователя 1.
    events = [baseline.Event(i, i + 1, "click", value, i)
              for i, value in enumerate([1., 1., 2.])]
    expected = {"total_selected": 3, "unique_users": 3,
                "top5": [(3, 2.), (2, 1.), (1, 1.)]}
    assert baseline.process_events(events, "click") == expected
    assert optimized.process_events(events, "click") == expected


def test_float_addition_order():
    events = [baseline.Event(i, 1, "click", value, timestamp)
              for i, (value, timestamp) in enumerate([(1., 3), (1e16, 1), (-1e16, 2)])]
    assert optimized.process_events(events, "click") == baseline.process_events(events, "click")
    assert optimized.process_events(events, "click")["top5"] == [(1, 1.)]
