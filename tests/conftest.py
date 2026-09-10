"""Put `src/` and `tools/` on the path for every test.

Three files import `gate_measures` and `run_gates`, which live in `tools/`, and without this the
plain `pytest tests` invocation stopped at collection with zero tests run -- the suite was only ever
runnable as `PYTHONPATH=src:tools pytest tests`, a fact recorded in one markdown file."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("src", "tools"):
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
