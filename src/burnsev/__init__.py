"""burnsev: burn severity and post-fire erosion priority from Sentinel-2 L2A.

The notebook imports this package; the MCP server (bonus A) wraps the same functions,
so there is one implementation of every step.
"""

from . import aoi, catalog, indices, ingest  # noqa: F401

__version__ = "0.1.0"
