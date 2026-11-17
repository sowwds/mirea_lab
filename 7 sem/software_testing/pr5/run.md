```bash
cd "/home/aleksey/Documents/mirea_lab/7 sem/software_testing/pr5"
.venv/bin/python -m pytest test_order_service.py --collect-only -q
.venv/bin/python -m pytest test_order_service.py::test_place_order_success -v
.venv/bin/python -m pytest test_order_service.py::test_place_order_save_failed -v
.venv/bin/python -m pytest test_order_service.py::test_place_order_storage_exception -v
.venv/bin/python -m pytest test_order_service.py -k get_order -v
bash -o pipefail -c '.venv/bin/python -m pytest test_order_service.py -v --color=no | tee pytest_output.txt'
```
