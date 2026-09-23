"""Loaded by pytest before any test file. burnsev must be imported before rasterio (see
burnsev/__init__.py), and in each test file the import sorter puts rasterio and rioxarray first."""

import burnsev  # noqa: F401
