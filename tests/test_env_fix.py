"""The PROJ fix at the top of burnsev/__init__.py must hold on a machine whose environment
points PROJ and GDAL at the wrong data directory. Runs in a subprocess, because the fix acts
at import time and this process has already imported everything."""

import os
import subprocess
import sys

CODE = (
    "import burnsev, rasterio\n"
    "from rasterio.warp import transform\n"
    "print(transform('EPSG:4326', 'EPSG:32633', [14.4], [40.8]))\n"
)


def test_reprojection_works_with_bogus_proj_and_gdal_variables(tmp_path):
    bogus = str(tmp_path)  # an empty directory: no proj.db, no gdal data
    env = dict(os.environ, PROJ_LIB=bogus, PROJ_DATA=bogus, GDAL_DATA=bogus)
    result = subprocess.run(
        [sys.executable, "-c", CODE], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr[-800:]
    assert "449386" in result.stdout  # easting of (14.4 E, 40.8 N) in UTM 33N
