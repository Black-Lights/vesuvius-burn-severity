"""burnsev: burn severity and post-fire erosion priority from Sentinel-2 L2A.

The notebook imports this package; the MCP server (bonus A) wraps the same functions,
so there is one implementation of every step.
"""

import os

# rasterio and pyproj ship their own copy of the PROJ database. If the machine also has a
# system-wide PROJ_LIB, PROJ_DATA or GDAL_DATA (conda, PostGIS and QGIS installers set them),
# GDAL reads that other copy and every reprojection fails with "proj.db ... comes from another
# PROJ installation". Removing the variables for this process makes the libraries fall back to
# their bundled data. It has no effect on a machine where they are not set. This must run
# before rasterio or pyproj is imported, which is why it sits at the top of the package:
# import burnsev before any other geospatial library, and restart a kernel started earlier.
for _name in ("PROJ_LIB", "PROJ_DATA", "GDAL_DATA"):
    os.environ.pop(_name, None)

from . import aoi, catalog, indices, ingest, reference  # noqa: F401

__version__ = "0.1.0"
