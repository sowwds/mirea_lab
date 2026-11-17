```bash
cd "/home/aleksey/Documents/mirea_lab/7 sem/software_testing/pr7"
.venv/bin/python -m pytest tests warmup --collect-only -q
bash -o pipefail -c 'WARMUP_APP=app_v0 .venv/bin/python -m pytest warmup/test_warmup.py -v --randomly-seed=20261112 --color=no | tee warmup_failure.txt'
.venv/bin/python -m pytest warmup/test_warmup.py -v
.venv/bin/python -m pytest tests/test_api.py::test_viewer_cannot_write -v
.venv/bin/python -m pytest 'tests/test_api.py::test_seats_conflict_does_not_change_state[above-300]' -v
bash -o pipefail -c '.venv/bin/python -m pytest tests warmup -q --randomly-seed=731 --color=no | tee random_order_output.txt'
bash -o pipefail -c '.venv/bin/python -m pytest tests warmup -v --randomly-seed=20261112 --cov=app --cov-report=term-missing --cov-report=html:htmlcov --cov-report=json:coverage.json --cov-fail-under=85 --color=no | tee pytest_output.txt'
```
