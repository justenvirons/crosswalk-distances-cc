"""Shared constants and helpers for the Cook County crossing-distance comparison.

Imagery candidates (Phase 1, Aim 1a):
  A  Google Maps satellite tiles (reference methodology's original source)
  B  Cook County CookOrtho2025, 3-band RGB (leaf-off, single vintage)

Both candidates are segmented by the same model (SAM 3, text prompt), so any
difference in error is attributable to the imagery. The county's 4-band product
is used only to screen candidate intersections for canopy cover (NDVI) when
drawing the stratified sample ("pool" clips); it is not a segmentation input.

The clip ID scheme and 25 m clip geometry follow agupta01/crossing-distances
(inference/utils.py), so footprints match the reference study's.
"""

import math

PRECISION = 6          # decimal places kept on lat/lon (~11 cm)
RADIUS_M = 25.0        # half-width of the square clip, metres
MODAL_ENV = "cook"     # Modal environment used for everything in this project

# Modal object names
IMAGERY_VOLUME = "cc-imagery"
WEIGHTS_VOLUME = "cc-sam3-weights"
RESULTS_VOLUME = "cc-results"
HF_SECRET = "huggingface"          # Modal secret holding HF_TOKEN (gated SAM 3 weights)

# Sub-folders inside the imagery and results volumes
SUBDIRS = {
    "A": "A_google",       # candidate A: Google basemap, RGB
    "B": "B_cook_rgb",     # candidate B: Cook County, RGB
    "pool": "pool_rgbn",   # 4-band county clips for the canopy screen only
}
NDVI_VEG = 0.20            # canopy screen: share of pixels above this NDVI = veg_frac

COOK_IMAGESERVER = (
    "https://gis.cookcountyil.gov/imagery/rest/services/CookOrtho2025/ImageServer"
)

# SAM 3 (image model). SAM 3.1's new checkpoint is a video-tracking update;
# still images use the SAM 3 image checkpoint.
SAM3_REPO = "facebook/sam3"
SAM3_FILES = ("sam3.pt", "config.json")
SAM3_CKPT = "sam3/sam3.pt"          # path inside the weights volume
DEFAULT_PROMPT = "road"             # confirm with the prompt pilot before the full run
DEFAULT_CONFIDENCE = 0.5            # SAM 3 Sam3Processor default
MIN_ROAD_PART_M2 = 5.0              # drop isolated road fragments smaller than this


def crosswalk_id(lat: float, lon: float) -> str:
    """Same encoding as the reference repo, e.g. 41878123N_87629456W."""
    lat, lon = round(lat, PRECISION), round(lon, PRECISION)
    return (
        f"{int(round(abs(lat) * 10**PRECISION))}{'N' if lat > 0 else 'S'}_"
        f"{int(round(abs(lon) * 10**PRECISION))}{'E' if lon > 0 else 'W'}"
    )


def _offset(lat, lon, dist, heading):
    lat_r, lon_r, h = map(math.radians, (lat, lon, heading))
    ang = dist / 6371000.0
    lat2 = math.asin(math.sin(lat_r) * math.cos(ang)
                     + math.cos(lat_r) * math.sin(ang) * math.cos(h))
    lon2 = lon_r + math.atan2(math.sin(h) * math.sin(ang) * math.cos(lat_r),
                              math.cos(ang) - math.sin(lat_r) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


def bbox_lonlat(lat: float, lon: float, radius: float = RADIUS_M):
    """(west, south, east, north) of the 2*radius square clip around a point."""
    d = math.sqrt(2) * radius
    n, w = _offset(lat, lon, d, 315)
    s, e = _offset(lat, lon, d, 135)
    return (min(w, e), min(s, n), max(w, e), max(s, n))
