"""Area of interest, time windows and constants for the Vesuvius 2025 fire.

Everything a user might want to change lives here, with the reason next to it.
"""

# Bounding box in EPSG:4326 as (west, south, east, north).
# Covers the cone of Vesuvius and its south-eastern flank (Terzigno, Ottaviano,
# Boscotrecase, Trecase), where the August 2025 fire burned. About 13 x 11 km.
BBOX = (14.35, 40.77, 14.50, 40.87)

# The fire started on the evening of 8 August 2025 and was contained by 12 August.
# EFFIS dates it from 7 August. The 7 August image covers only the west of the box and none of
# the fire, so no number depends on which of the two dates is right.
FIRE_START = "2025-08-08"
FIRE_END = "2025-08-12"

# Pre-fire and post-fire windows for the median composites.
# Pre: same season as the fire, ends the day before it. It starts on 1 July, not 1 June: the
# unburned reference stays flat all summer, but inside the future burn NBR slides from 0.50 in
# June to 0.38 on 6 August (drying understory, small early-August fires), so a June-to-August
# median overstates the pre-fire state and inflates dNBR. Severe area (dNBR >= 0.27): 611 ha
# with June, 564 ha with July, 525 ha with 6 August alone. July keeps a median with at least two
# observations per pixel.
# Post: starts after containment and lasts one month. Recovery begins within weeks (mean NBR in
# the burn -0.16 on 14 Aug, -0.06 by early September, +0.07 by 13 Oct), so a longer median mixes
# recovery into severity and halves the high class (91 ha with a median to 15 Oct, 149 ha to
# 15 Sep, 204 ha from the first clear image; total burned area 564, 591 and 628 ha). One month is
# the timing of the Key and Benson initial assessment and still a median over several dates.
PRE_WINDOW = ("2025-07-01", "2025-08-07")
POST_WINDOW = ("2025-08-13", "2025-09-15")

# Scenes are loaded over a longer span than the composites use, so the NBR time series in step 5
# shows the June slide before the fire and the recovery after it. The composites ignore the extra
# dates.
SERIES_START = "2025-06-01"
SERIES_END = "2025-10-15"

# Scene-level cloud cover ceiling for the STAC search (percent). Generous on purpose:
# the per-pixel SCL mask does the real work, and a 25% scene can be clear over the AOI.
MAX_CLOUD = 25.0

# Bands, each with a job: B02, B03, B04 (blue, green, red, 10 m) for the pictures; B8A (near
# infrared) and B12 (short-wave infrared 2), both 20 m, for the burn index; SCL, the scene
# classification (20 m), for the cloud mask. B11 (short-wave infrared 1, 20 m) is not used by
# the core: the foundation model in bonus B expects the six Harmonized Landsat Sentinel bands
# (B02, B03, B04, B8A, B11, B12), so it is downloaded once with the others.
BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12", "SCL"]

# Working resolution in metres. NBR needs B8A and B12, both native 20 m, so the cube is
# built at 20 m rather than upsampling SWIR to 10 m and pretending it carries 10 m detail.
RESOLUTION = 20

# The UTM zone of tile 33TVF. Loading straight into it avoids a resample on export.
CRS = "EPSG:32633"

# Sentinel-2 L2A number format, from the ESA product definition (not a choice made here). A stored
# value is reflectance times BOA_QUANTIFICATION_VALUE, plus 1000 since processing baseline 04.00:
# reflectance = (DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE. The offset of each product is
# read from the catalogue (catalog.boa_add_offset); the divisor is fixed.
BOA_QUANTIFICATION_VALUE = 10000

# SCL classes to mask out (Sentinel-2 L2A scene classification):
# 0 no data, 1 saturated/defective, 3 cloud shadow, 8 cloud medium probability,
# 9 cloud high probability, 10 thin cirrus, 11 snow/ice.
SCL_MASK = (0, 1, 3, 8, 9, 10, 11)

# dNBR burn severity classes, USGS / FIREMON (Key and Benson 2006), thresholds in dNBR units.
# Order matters: the first break a value is below gives its class.
SEVERITY_CLASSES = [
    ("unburned", -9.0, 0.10),
    ("low", 0.10, 0.27),
    ("moderate-low", 0.27, 0.44),
    ("moderate-high", 0.44, 0.66),
    ("high", 0.66, 9.0),
]

# Decision rule: severe burn on steep ground.
#
# Source: the USGS post-fire debris-flow likelihood model M1 (Staley et al. 2017, Geomorphology 278).
# Its terrain term is the proportion of upslope area burned at moderate or high severity with a
# slope of 23 degrees or more. This notebook computes only that terrain term. The full model also
# needs rainfall intensity and soil erodibility, which are out of scope; the notebook says so.
#
# "Moderate or high" starts at dNBR 0.27 (the moderate-low class) in the Key and Benson breaks above.
# USGS itself uses per-fire BARC maps for that class; the fixed breaks stand in for them here, and the
# notebook lists that as a limitation. SEVERE_CLASSES is derived from the cut, so there is one number
# to change and one number to defend.
SEVERITY_CUT_DNBR = 0.27
SEVERE_CLASSES = tuple(name for name, low, _high in SEVERITY_CLASSES if low >= SEVERITY_CUT_DNBR)

# Slope threshold in degrees, from the same model. The ranking is rerun at each value in
# SLOPE_SENSITIVITY_DEG to show that the top cells do not change when the threshold moves a little.
SLOPE_THRESHOLD_DEG = 23.0
SLOPE_SENSITIVITY_DEG = (20.0, 23.0, 26.0)

# Side of the grid cell that groups pixels into planning units. 250 m is about 12 x 12 pixels at
# 20 m: enough to average out single-pixel noise, small enough to point a crew at one slope.
GRID_CELL_M = 250

# Cuts between the three priorities, as the share of a cell that is severe and steep: at least
# half, treat first; a quarter to a half, treat next; less, monitor. A planning choice, not a
# published threshold: M1 turns the share into a probability together with rainfall intensity,
# which is not modelled here. Step 9 shows how the lists move with the slope threshold.
PRIORITY_SHARES = (0.50, 0.25)
