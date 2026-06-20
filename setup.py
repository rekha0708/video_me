"""Setuptools shim plus GPU setup convenience command.

Packaging still reads metadata from pyproject.toml. For runtime setup, use:
    python setup.py gpu --install-python-deps
or:
    python -m scripts.setup_gpu --install-python-deps
"""

from __future__ import annotations

import sys


if len(sys.argv) > 1 and sys.argv[1] == "gpu":
    from scripts.setup_gpu import main

    raise SystemExit(main(sys.argv[2:]))

from setuptools import setup

setup()
