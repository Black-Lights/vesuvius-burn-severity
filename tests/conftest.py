"""Loaded by pytest before any test file. burnsev must be imported before rasterio (see
burnsev/__init__.py), and in each test file the import sorter puts rasterio and rioxarray first."""

import importlib.util

import burnsev  # noqa: F401

# The agent's tests need the bonus A packages (pip install -r requirements-agent.txt). Without them
# that file is left out, so the core checks still run.
if importlib.util.find_spec("langgraph") is None:
    collect_ignore = ["test_agent.py"]
