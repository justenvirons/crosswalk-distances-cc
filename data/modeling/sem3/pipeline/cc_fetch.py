"""Fetch 50 m x 50 m clips around each intersection (Modal, CPU).

  source A    -> candidate A: Google basemap, zoom 22 satellite tiles (the same
                 tiles, zoom and bbox the reference repo fetched with
                 segment-geospatial's tms_to_geotiff)
  source B    -> candidate B: Cook County CookOrtho2025, bands R,G,B
  source pool -> Cook County R,G,B,NIR, used only for the canopy (NDVI) screen

Every clip is written as a georeferenced GeoTIFF (EPSG:3857) to the `cc-imagery`
volume under /<subdir>/crosswalk_<id>.tif, and a per-clip summary CSV is written
locally.

Usage (from ~/crosswalk/pipeline, with MODAL_ENVIRONMENT=cook):
  uv run modal run cc_fetch.py --source pool --input ../data/sample/candidates.csv \
      --out-csv ../data/sample/pool_ndvi.csv
  uv run modal run cc_fetch.py --source B --input ../data/sample/sample.csv \
      --out-csv ../logs/fetch_B.csv
"""

import modal

from cc_common import (
    COOK_IMAGESERVER, IMAGERY_VOLUME, NDVI_VEG, SUBDIRS, bbox_lonlat, crosswalk_id,
)

app = modal.App("cc-crosswalk-fetch")
imagery = modal.Volume.from_name(IMAGERY_VOLUME, create_if_missing=True)

cook_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("requests", "numpy", "rasterio", "pyproj", "pillow")
    .add_local_python_source("cc_common")
)

# Google satellite tiles, as segment-geospatial's source="Satellite".
GOOGLE_SATELLITE = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
MERC_HALF = 20037508.342789244     # half the EPSG:3857 world width, metres


def _summary(path, cid, lat, lon):
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        res_x = abs(src.transform.a)
        out = {
            "id": cid, "lat": lat, "lon": lon, "path": path,
            "width": src.width, "height": src.height, "bands": src.count,
            # ground metres per pixel (Web Mercator units scaled by cos(lat))
            "ground_res_m": res_x * np.cos(np.radians(lat)),
        }
        if src.count >= 4:
            r = src.read(1).astype("float32")
            nir = src.read(4).astype("float32")
            ndvi = (nir - r) / (nir + r + 1e-6)
            out["ndvi_mean"] = float(ndvi.mean())
            out["veg_frac"] = float((ndvi > NDVI_VEG).mean())
    return out


@app.function(image=cook_image, volumes={"/imagery": imagery},
              max_containers=8, timeout=600, retries=2)
def fetch_cook(lat: float, lon: float, subdir: str, bands: str, size: int = 1024):
    import os
    import time

    import numpy as np
    import rasterio
    import requests
    from pyproj import Transformer
    from rasterio.io import MemoryFile
    from rasterio.transform import from_bounds

    cid = crosswalk_id(lat, lon)
    w, s, e, n = bbox_lonlat(lat, lon)
    t = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x0, y0 = t.transform(w, s)
    x1, y1 = t.transform(e, n)

    params = {
        "bbox": f"{x0},{y0},{x1},{y1}", "bboxSR": 3857, "imageSR": 3857,
        "size": f"{size},{size}", "format": "tiff", "pixelType": "U8",
        "bandIds": bands, "interpolation": "RSP_BilinearInterpolation", "f": "image",
    }
    for attempt in range(4):
        r = requests.get(f"{COOK_IMAGESERVER}/exportImage", params=params, timeout=120)
        if r.ok and r.headers.get("content-type", "").startswith("image"):
            break
        time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"exportImage failed for {cid}: {r.status_code} {r.text[:200]}")

    with MemoryFile(r.content) as mf, mf.open() as src:
        data = src.read()

    # Always write our own georeferencing from the requested bbox, so the clip is
    # correct even if the server's TIFF tags are missing or in another SR.
    os.makedirs(f"/imagery/{subdir}", exist_ok=True)
    path = f"/imagery/{subdir}/crosswalk_{cid}.tif"
    profile = dict(driver="GTiff", width=data.shape[2], height=data.shape[1],
                   count=data.shape[0], dtype="uint8", crs="EPSG:3857",
                   transform=from_bounds(x0, y0, x1, y1, data.shape[2], data.shape[1]),
                   compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.uint8))
    imagery.commit()
    return _summary(path, cid, lat, lon)


def _google_mosaic(w, s, e, n, zoom, get_tile):
    """Stitch the zoom-level tiles covering a lon/lat bbox and crop to it.

    Replaces segment-geospatial's tms_to_geotiff, whose recent releases need
    GDAL's Python bindings. Returns (rgb uint8 array of shape (3, h, w),
    (x0, y0, x1, y1) bounds of the cropped pixels in EPSG:3857).
    """
    import io
    import math

    import numpy as np
    from PIL import Image

    world = 256 * 2 ** zoom                       # world width in pixels

    def to_px(lon, lat):
        siny = math.sin(math.radians(lat))
        x = (lon + 180.0) / 360.0 * world
        y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * world
        return x, y

    fx0, fy0 = to_px(w, n)                        # top-left
    fx1, fy1 = to_px(e, s)                        # bottom-right
    px0, py0 = math.floor(fx0), math.floor(fy0)
    px1, py1 = math.ceil(fx1), math.ceil(fy1)
    tx0, ty0, tx1, ty1 = px0 // 256, py0 // 256, (px1 - 1) // 256, (py1 - 1) // 256

    mosaic = Image.new("RGB", ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256))
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            tile = Image.open(io.BytesIO(get_tile(tx, ty, zoom))).convert("RGB")
            mosaic.paste(tile, ((tx - tx0) * 256, (ty - ty0) * 256))
    crop = mosaic.crop((px0 - tx0 * 256, py0 - ty0 * 256,
                        px1 - tx0 * 256, py1 - ty0 * 256))
    data = np.asarray(crop, dtype=np.uint8).transpose(2, 0, 1)

    def to_merc(p):                               # pixel edge -> metres
        return p / world * 2 * MERC_HALF - MERC_HALF

    bounds = (to_merc(px0), -to_merc(py1), to_merc(px1), -to_merc(py0))
    return data, bounds


@app.function(image=cook_image, volumes={"/imagery": imagery},
              max_containers=20, timeout=600, retries=2)
def fetch_google(lat: float, lon: float, subdir: str, zoom: int = 22):
    import os
    import time

    import numpy as np
    import rasterio
    import requests
    from rasterio.transform import from_bounds

    cid = crosswalk_id(lat, lon)
    w, s, e, n = bbox_lonlat(lat, lon)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (cc-crosswalk-fetch)"

    def get_tile(x, y, z):
        url = GOOGLE_SATELLITE.format(x=x, y=y, z=z)
        for attempt in range(4):
            r = session.get(url, timeout=60)
            if r.ok and r.headers.get("content-type", "").startswith("image"):
                return r.content
            time.sleep(2 ** attempt)
        raise RuntimeError(f"tile z{z}/{x}/{y} failed for {cid}: {r.status_code}")

    # Zoom 22 "Satellite" tiles over the clip bbox, output in EPSG:3857
    data, (x0, y0, x1, y1) = _google_mosaic(w, s, e, n, zoom, get_tile)

    os.makedirs(f"/imagery/{subdir}", exist_ok=True)
    path = f"/imagery/{subdir}/crosswalk_{cid}.tif"
    profile = dict(driver="GTiff", width=data.shape[2], height=data.shape[1],
                   count=3, dtype="uint8", crs="EPSG:3857",
                   transform=from_bounds(x0, y0, x1, y1, data.shape[2], data.shape[1]),
                   compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(np.uint8))
    imagery.commit()
    return _summary(path, cid, lat, lon)


@app.local_entrypoint()
def main(source: str, input: str, out_csv: str, limit: int = 0, subdir: str = ""):
    """--subdir overrides the target folder (e.g. pilot_B for the prompt pilot)."""
    import pandas as pd

    df = pd.read_csv(input)
    if limit:
        df = df.head(limit)
    lats, lons = df["lat"].round(6).tolist(), df["lon"].round(6).tolist()
    subdir = subdir or SUBDIRS[source]
    print(f"Fetching {len(df)} clips for source {source} -> /{subdir}")

    if source == "A":
        results = fetch_google.map(lats, lons, kwargs={"subdir": subdir},
                                   return_exceptions=True)
    else:
        bands = "0,1,2" if source == "B" else "0,1,2,3"
        results = fetch_cook.map(lats, lons, kwargs={"subdir": subdir, "bands": bands},
                                 return_exceptions=True)

    rows = []
    for (la, lo), r in zip(zip(lats, lons), results):
        if isinstance(r, Exception):
            rows.append({"id": crosswalk_id(la, lo), "lat": la, "lon": lo, "error": repr(r)})
        else:
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    n_err = int(out["error"].notna().sum()) if "error" in out else 0
    print(f"done: {len(out) - n_err} ok, {n_err} failed -> {out_csv}")
