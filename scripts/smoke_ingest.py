"""Day-1 smoke test: search, print the scene table, load two scenes, compute NBR.

Run:  .venv/Scripts/python.exe scripts/smoke_ingest.py
"""

import time

import numpy as np

from burnsev import aoi, catalog, indices, ingest

t0 = time.time()
items = catalog.search_scenes(aoi.BBOX, aoi.PRE_WINDOW[0], aoi.POST_WINDOW[1], aoi.MAX_CLOUD)
table = catalog.scene_table(items)
print(f"{len(items)} scenes in {time.time() - t0:.1f}s")
print(table.to_string(index=False))

# raster:bands presence, to know where the offset really comes from
a = items[-1].assets["B04"]
print("\nraster:bands on B04:", a.extra_fields.get("raster:bands"))
print("asset href:", a.href[:90])

# two scenes: one pre-fire, one post-fire
pick = [i for i in items if i.datetime.date().isoformat() in ("2025-08-06", "2025-08-17")]
print("\nloading", [i.id for i in pick])
t0 = time.time()
ds = ingest.load_cube(pick)
print(f"cube in {time.time() - t0:.1f}s:", dict(ds.sizes), "crs", ds.odc.crs, "res", ds.odc.geobox.resolution)
offsets = {str(np.datetime64(t, 'D')): catalog.boa_offset(pick[0]) for t in ds.time.values}
refl = ingest.mask_and_scale(ds, offsets)
print("valid fraction per date:", refl["valid_fraction"].values.round(3))
nbr = indices.nbr(refl)
print("mean NBR per date:", nbr.mean(dim=("y", "x")).values.round(3))
print("B04 reflectance range:", float(refl["B04"].min()), float(refl["B04"].max()))
