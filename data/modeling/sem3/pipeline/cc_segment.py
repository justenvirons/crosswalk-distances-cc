"""Segment the drivable surface with SAM 3 and export polygons in the format the
reference grow-cut step expects (Modal, GPU).

Per clip:
  RGB GeoTIFF -> SAM 3 (backbone once) -> road probability for the chosen
  Config (cc_sam3: mode text / box / text+box; fusion instance / semantic / fused;
  optional negative classes) -> drop fragments < MIN_ROAD_PART_M2
  -> polygons in the clip's georeferencing -> EPSG:4326
  -> cross_walks.geojson (one row per image: image_name, geometry)

Box prompts are 5 m squares on OSM road centre-lines ~15 m from the clip centre
(cc_geom.centerline_boxes); they need /inputs/osm_roads.geojson on the results
volume (made by 02b_osm_roads.py).

--osm-clip keeps road only within a corridor around each OSM drivable centre-line
(half-width by highway class + --osm-margin-m), cutting parking lots and driveways
set back behind the sidewalk. It uses the same osm_roads.geojson. Clips with no
centre-line nearby are left unclipped (osm_clip = False in segment_stats.csv).

Modes
  normal  one Config -> cross_walks.geojson, segment_stats.csv, QA overlays
  pilot   a grid of Configs on pilot clips -> QA overlays + stats per Config

Usage (from ~/crosswalk/pipeline, MODAL_ENVIRONMENT=cook):
  uv run modal run cc_segment.py --source B --input-dir pilot_B --pilot --run-name pilot
  uv run modal run cc_segment.py --source A --mode text --fusion fused --prompts road \
      --fill-holes-m2 10 --run-name v1
  uv run modal run cc_segment.py --source B --input-dir pilot_B --pilot --run-name pilot_osm \
      --modes text --fusions semantic --prompt-sets road --negative-sets "" \
      --prob-thds 0.4 --fill-holes 25 --osm-clip --delta finetuned/ft1/seg_head_last.pt
"""

import modal

from cc_common import (
    DEFAULT_CONFIDENCE, IMAGERY_VOLUME, MIN_ROAD_PART_M2, RESULTS_VOLUME, SAM3_CKPT,
    SUBDIRS, WEIGHTS_VOLUME,
)

app = modal.App("cc-crosswalk-segment")
imagery = modal.Volume.from_name(IMAGERY_VOLUME, create_if_missing=True)
weights = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
results = modal.Volume.from_name(RESULTS_VOLUME, create_if_missing=True)

OSM_ROADS = "/results/inputs/osm_roads.geojson"

# Half-width (m) of the corridor kept around an OSM centre-line by --osm-clip, by
# `highway` class, before --osm-margin-m is added. Deliberately wider than typical
# curb-to-curb half-widths so real carriageway is never cut: the target is lots and
# driveways set back behind the sidewalk, not the curb itself.
OSM_HALF_WIDTH_M = {"motorway": 20.0, "trunk": 18.0, "primary": 15.0,
                    "secondary": 13.0, "tertiary": 11.0}
OSM_HALF_WIDTH_DEFAULT_M = 9.0      # residential, unclassified, missing tag
# Not used for the corridor: parking aisles, alleys and paths would re-admit the
# areas the clip is meant to remove.
OSM_SKIP_CLASSES = {"service", "track", "path", "footway", "cycleway", "pedestrian",
                    "steps", "bridleway", "corridor"}

# SAM 3 needs Python >= 3.12, PyTorch >= 2.7 and CUDA >= 12.6 (repo README).
sam3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.10.0", "torchvision",
                 index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("git+https://github.com/facebookresearch/sam3.git",
                 "setuptools<81",   # sam3 imports pkg_resources
                 "psutil", "einops",
                 # imported on SAM 3's model-loading path but not declared as
                 # core dependencies: pycocotools (module level, via sam3.train),
                 # scikit-image (connected-components fallback without cc_torch)
                 "pycocotools", "scikit-image")
    .pip_install("numpy>=1.26,<2", "opencv-python-headless", "rasterio",
                 "geopandas", "shapely", "pyproj", "pandas", "scipy", "pillow")
    # fail at image build (CPU, seconds) rather than on a GPU container if an
    # import is still missing
    .run_commands("python -c 'from sam3.model_builder import build_sam3_image_model; "
                  "from sam3.model.sam3_image_processor import Sam3Processor; print(\"sam3 imports OK\")'")
    .add_local_python_source("cc_common", "cc_geom", "cc_sam3")
)


def drop_small_parts(mask, px_m2, min_m2):
    """Remove connected road fragments smaller than min_m2 (8-connectivity)."""
    import numpy as np
    from scipy import ndimage

    lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1)) * px_m2
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_m2
    return keep[lab]


def mask_to_geometry(mask, transform):
    """Union of polygons for True pixels, in the raster's CRS."""
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    polys = [shape(g) for g, v in shapes(mask.astype("uint8"), mask=mask,
                                         transform=transform) if v == 1]
    return unary_union(polys) if polys else None


def clip_lat(bounds, crs):
    import geopandas as gpd
    from shapely.geometry import box

    return gpd.GeoSeries([box(*bounds)], crs=crs).centroid.to_crs(4326).y.iloc[0]


def ground_pixel_area_m2(transform, crs, bounds):
    """Ground area of one pixel. Web Mercator units shrink by cos(lat) on the ground."""
    import math

    a = abs(transform.a) * abs(transform.e)
    if crs is not None and crs.to_epsg() == 3857:
        a *= math.cos(math.radians(clip_lat(bounds, crs))) ** 2
    return a


def qa_overlay(rgb, mask, path, boxes=None, max_px=1024, corridor=None):
    """Road tinted, road edge magenta, prompt boxes yellow, OSM corridor edge white."""
    import cv2
    import numpy as np

    from cc_geom import cxcywh_to_pixels

    img = rgb.copy()
    img[mask] = (0.55 * img[mask] + 0.45 * np.array([0, 200, 255])).astype(np.uint8)
    if corridor is not None:
        img[cv2.Canny(corridor.astype(np.uint8) * 255, 50, 150) > 0] = (255, 255, 255)
    edge = cv2.Canny(mask.astype(np.uint8) * 255, 50, 150) > 0
    img[edge] = (255, 0, 255)
    h, w = img.shape[:2]
    for b in boxes or []:
        x0, y0, x1, y1 = cxcywh_to_pixels(b, w, h)
        cv2.rectangle(img, (x0, y0), (x1, y1), (255, 230, 0), max(2, w // 400))
    if max(h, w) > max_px:
        s = max_px / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])


def prompt_boxes_for_clip(roads, bounds, crs):
    """Normalized exemplar boxes for one clip from OSM road lines (EPSG:3857 GeoDataFrame)."""
    from shapely.geometry import Point, box

    from cc_geom import boxes_to_cxcywh, centerline_boxes, mercator_units_per_m

    if roads is None or crs is None or crs.to_epsg() != 3857:
        return []
    frame = box(*bounds)
    lines = [g.intersection(frame) for g in roads.geometry.iloc[roads.sindex.query(frame)]]
    lines = [ln for ln in lines if not ln.is_empty and ln.geom_type in ("LineString", "MultiLineString")]
    flat = []
    for ln in lines:
        flat.extend(getattr(ln, "geoms", [ln]))
    center = Point((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
    u = mercator_units_per_m(clip_lat(bounds, crs))
    return boxes_to_cxcywh(centerline_boxes(flat, center, u, bounds=tuple(bounds)), bounds)


def _highway_class(v):
    """OSM `highway` value as a single string (osmnx can store lists)."""
    if isinstance(v, (list, tuple)):
        v = v[0] if v else ""
    v = str(v or "").strip("[]'\" ")
    return v.split(",")[0].strip("'\" ")


def osm_corridor(roads, bounds, crs, out_shape, transform, margin_m):
    """True within each OSM drivable centre-line's corridor, on the clip's pixel grid.

    Returns None when the clip is not EPSG:3857 or no centre-line reaches it, so the
    caller leaves that mask unclipped rather than erasing it.
    """
    import math

    from rasterio.features import rasterize
    from shapely.geometry import box

    if roads is None or crs is None or crs.to_epsg() != 3857:
        return None
    u = 1.0 / math.cos(math.radians(clip_lat(bounds, crs)))     # Mercator units per m
    reach = (max(OSM_HALF_WIDTH_M.values()) + margin_m) * u
    frame = box(*bounds)
    near = roads.iloc[roads.sindex.query(frame.buffer(reach))]
    shapes_ = []
    for geom, hw in zip(near.geometry, near.get("highway", [None] * len(near))):
        cls = _highway_class(hw)
        if geom is None or geom.is_empty or cls in OSM_SKIP_CLASSES:
            continue
        half = OSM_HALF_WIDTH_M.get(cls.removesuffix("_link"), OSM_HALF_WIDTH_DEFAULT_M)
        buf = geom.buffer((half + margin_m) * u).intersection(frame)
        if not buf.is_empty:
            shapes_.append((buf, 1))
    if not shapes_:
        return None
    return rasterize(shapes_, out_shape=out_shape, transform=transform,
                     fill=0, dtype="uint8").astype(bool)


@app.function(image=sam3_image, gpu="L40S", timeout=6 * 3600,
              volumes={"/imagery": imagery, "/weights": weights, "/results": results})
def segment(subdir: str, run_name: str, configs: list[dict], pilot: bool = False,
            limit: int = 0, checkpoint: str = SAM3_CKPT, delta: str = "",
            osm_clip: bool = False, osm_margin_m: float = 2.0):
    import glob
    import json
    import os
    import time

    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import rasterio
    import torch

    from cc_geom import fill_small_holes
    from cc_sam3 import Config, ImageSession, load

    cfgs = [Config(**{**c, "prompts": tuple(c["prompts"]), "negatives": tuple(c["negatives"])})
            for c in configs]
    needs_boxes = any(c.mode != "text" for c in cfgs)
    roads = None
    if needs_boxes or osm_clip:
        if not os.path.exists(OSM_ROADS):
            raise FileNotFoundError(f"{OSM_ROADS} missing: run 02b_osm_roads.py and upload it")
        roads = gpd.read_file(OSM_ROADS).to_crs(3857)
        uses = [u for u, on in (("box prompts", needs_boxes), ("the OSM clip", osm_clip)) if on]
        print(f"Loaded {len(roads)} OSM road lines for {' and '.join(uses)}")
        if osm_clip and "highway" not in roads.columns:
            print(f"WARNING: {OSM_ROADS} has no 'highway' column; every corridor uses "
                  f"{OSM_HALF_WIDTH_DEFAULT_M} m, which can cut wide arterials")

    model, proc = load(f"/weights/{checkpoint}", delta=f"/weights/{delta}" if delta else None)
    base = f"/results/{subdir}/{run_name}"
    folder = {c: os.path.join(base, c.key) if pilot else base for c in cfgs}
    for f in folder.values():
        os.makedirs(os.path.join(f, "qa"), exist_ok=True)

    tifs = sorted(glob.glob(f"/imagery/{subdir}/*.tif"))
    if limit:
        tifs = tifs[:limit]
    if not tifs:
        raise FileNotFoundError(f"no clips in /imagery/{subdir}: run cc_fetch.py for this "
                                "source first and check its out-csv for errors")
    print(f"Segmenting {len(tifs)} clips from /imagery/{subdir}; {len(cfgs)} config(s)")

    records = {c: [] for c in cfgs}
    stats = []
    t0 = time.time()
    for i, tif in enumerate(tifs):
        name = os.path.basename(tif)
        with rasterio.open(tif) as src:
            rgb = np.transpose(src.read([1, 2, 3]), (1, 2, 0)).astype(np.uint8)
            transform, crs, bounds = src.transform, src.crs, src.bounds
        px_m2 = ground_pixel_area_m2(transform, crs, bounds)
        boxes = prompt_boxes_for_clip(roads, bounds, crs) if needs_boxes else []
        corridor = (osm_corridor(roads, bounds, crs, rgb.shape[:2], transform, osm_margin_m)
                    if osm_clip else None)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            sess = ImageSession(model, proc, rgb)
            for c in cfgs:
                mask, st = sess.road(c, boxes)
                mask = fill_small_holes(mask, px_m2, c.fill_holes_m2)
                raw_frac = float(mask.mean())
                if corridor is not None:
                    mask = mask & corridor
                mask = drop_small_parts(mask, px_m2, MIN_ROAD_PART_M2)
                records[c].append({"image_name": name, "geometry": mask_to_geometry(mask, transform)})
                stats.append({"image_name": name, "config": c.key, **st,
                              "road_frac": float(mask.mean()),
                              "road_m2": float(mask.sum() * px_m2),
                              "empty": bool(not mask.any()),
                              "road_frac_unclipped": raw_frac,
                              "osm_clip": corridor is not None})
                qa_overlay(rgb, mask, os.path.join(folder[c], "qa", name.replace(".tif", ".jpg")),
                           boxes if c.mode != "text" else None, corridor=corridor)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(tifs)}  ({(time.time() - t0) / (i + 1):.1f} s/clip)")

    for c, recs in records.items():
        gdf = gpd.GeoDataFrame(recs, geometry="geometry", crs=crs)
        gdf = gdf[gdf.geometry.notna()].to_crs("EPSG:4326")
        gdf.to_file(os.path.join(folder[c], "cross_walks.geojson"), driver="GeoJSON")
    st = pd.DataFrame(stats)
    st.to_csv(os.path.join(base, "segment_stats.csv"), index=False)
    if pilot:
        (st.groupby("config")[["road_frac", "empty", "n_instances", "box_fallback_to_text",
                                "road_frac_unclipped", "osm_clip"]]
           .mean().to_csv(os.path.join(base, "pilot_summary.csv")))
    with open(os.path.join(base, "run.json"), "w") as f:
        json.dump({"subdir": subdir, "pilot": pilot, "configs": configs,
                   "checkpoint": checkpoint, "delta": delta, "min_road_part_m2": MIN_ROAD_PART_M2,
                   "osm_clip": osm_clip, "osm_margin_m": osm_margin_m,
                   "n_images": len(tifs)}, f, indent=2)
    results.commit()
    return base


def _split(s, sep=","):
    return [x.strip() for x in s.split(sep) if x.strip()]


@app.local_entrypoint()
def main(source: str, run_name: str, input_dir: str = "", pilot: bool = False,
         mode: str = "text+box", fusion: str = "fused", prompts: str = "road",
         negatives: str = "", prob_thd: float = 0.3,
         confidence: float = DEFAULT_CONFIDENCE, limit: int = 0,
         rule: str = "positive", fill_holes_m2: float = 0.0,
         prob_thds: str = "", fill_holes: str = "0", rules: str = "positive",
         modes: str = "text,box,text+box", fusions: str = "instance,fused",
         prompt_sets: str = "road;street;roadway;asphalt road",
         negative_sets: str = ";sidewalk,grass,building", delta: str = "",
         osm_clip: bool = False, osm_margin_m: float = 2.0):
    """Normal run: --mode/--fusion/--prompts/--negatives pick one Config.
    Pilot run (--pilot): every combination of --modes x --fusions x --prompt-sets
    (';' separates sets, ',' unions phrases within a set) x --negative-sets
    (';' separates sets; an empty set means no negatives).
    --delta finetuned/<run>/seg_head_best.pt uses a cc_finetune.py head.
    --osm-clip keeps road only near OSM drivable centre-lines (needs
    /inputs/osm_roads.geojson); --osm-margin-m widens that corridor."""
    from dataclasses import asdict

    from cc_sam3 import Config, config_grid

    subdir = input_dir or SUBDIRS[source]
    if pilot:
        cfgs = config_grid(_split(modes), _split(fusions),
                           [_split(s) for s in prompt_sets.split(";") if s.strip()],
                           [_split(s) for s in negative_sets.split(";")],
                           prob_thd, confidence,
                           prob_thds=[float(x) for x in _split(prob_thds)] or None,
                           fill_holes=[float(x) for x in _split(fill_holes)] or [0.0],
                           rules=_split(rules))
    else:
        cfgs = [Config(mode, fusion, tuple(_split(prompts)), tuple(_split(negatives)),
                       prob_thd, confidence, rule, fill_holes_m2)]
    print(f"{len(cfgs)} config(s):", *[c.key for c in cfgs], sep="\n  ")
    base = segment.remote(subdir, run_name, [asdict(c) for c in cfgs], pilot=pilot, limit=limit,
                          delta=delta, osm_clip=osm_clip, osm_margin_m=osm_margin_m)
    print(f"done -> volume {RESULTS_VOLUME}:{base.removeprefix('/results')}")
