```bash
cd "/home/aleksey/Documents/mirea_lab/7 sem/software_testing/pr4"
.venv/bin/python -m pytest test_calculator.py --collect-only -q
.venv/bin/python -m pytest 'test_calculator.py::test_add_integers[positive]' -v
.venv/bin/python -m pytest test_calculator.py::test_divide_by_zero -v
.venv/bin/python -m pytest test_calculator.py -v --cov=calculator --cov-report=term-missing --cov-report=html:htmlcov --cov-report=json:coverage.json --cov-fail-under=90
```
