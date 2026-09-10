import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _sub in ("src", "tools"):
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def pytest_addoption(parser):
    parser.addoption("--quick", action="store_true", default=False,
                     help="the three cheapest working points, shortened cuts (a pre-push check)")


def pytest_configure(config):
    config.addinivalue_line("markers", "regression: reruns working points; needs CUDA and minutes")
