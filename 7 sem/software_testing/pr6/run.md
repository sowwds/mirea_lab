```bash
cd "/home/aleksey/Documents/mirea_lab/7 sem/software_testing/pr6"
.venv/bin/python -m pytest test_temperature_sensor.py --collect-only -q
.venv/bin/python -m pytest test_temperature_sensor.py::test_temperature_change -v --setup-show
.venv/bin/python -m pytest test_temperature_sensor.py::test_unsubscribe_unknown test_temperature_sensor.py::test_repeated_unsubscribe -v
.venv/bin/python -m pytest test_temperature_sensor.py::test_subscriber_exception -v
bash -o pipefail -c '.venv/bin/python -m pytest test_temperature_sensor.py -v --color=no | tee pytest_output.txt'
```
