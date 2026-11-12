"""Define the crossings to be measured (units of analysis) and the ground-truth layer (server).

A unit is one OSM crossing edge (crosswalk_edges_v2_0, the edges grow-cut measures
along) that lies wholly inside the 50 m clip of a screened sample intersection, so
both imagery sources show the whole crossing. Units are fixed here, from OSM and
the clip geometry only, before any model output or hand measurement is examined.
An edge inside two overlapping clips is assigned to the clip whose centre is nearest.

Writes to --out-dir:
  units.csv                 unit_id, sample_id, stratum, added (no lengths)
  gt_google_subset.csv      random --google-frac of units per stratum (seed
                            --seed) to measure a second time on Google imagery
  groundtruth.gpkg
    layer clips             the 50 m clip squares, with sample_id (navigation)
    layer units             the unit edges, with unit_id and sample_id (reference)
    layer gt_crossings      empty line layer (EPSG:3435, US ft) to digitize the
                            hand measurements into: one line per crossing piece,
                            curb face to curb face; fields unit_id, piece, island,
                            curb_visible, imagery, measurer, notes

Usage:
  uv run python 04_crossing_units.py
"""

import argparse
import os

import pandas as pd

from cc_common import RADIUS_M, bbox_lonlat

STRATA = ["geography", "canopy", "tile_pos"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="../data/sample/sample_screened.csv")
    ap.add_argument("--edges", default="../data/osm/crosswalk_edges_v2_0.shp")
    ap.add_argument("--out-dir", default="../data/groundtruth")
    ap.add_argument("--margin-m", type=float, default=0.5,
                    help="an edge must lie this far inside the clip edge")
    ap.add_argument("--google-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--overwrite-gt", action="store_true",
                    help="replace an existing gt_crossings layer (destroys measurements)")
    args = ap.parse_args()

    import geopandas as gpd
    from shapely.geometry import box

    os.makedirs(args.out_dir, exist_ok=True)
    gpkg = os.path.join(args.out_dir, "groundtruth.gpkg")

    s = pd.read_csv(args.sample)
    s["id"] = s["id"].astype(str)
    clips = gpd.GeoDataFrame(
        s[["id", *STRATA, "added"]].rename(columns={"id": "sample_id"}),
        geometry=[box(*bbox_lonlat(la, lo, RADIUS_M)) for la, lo in zip(s.lat, s.lon)],
        crs=4326).to_crs(3435)
    centres = gpd.GeoSeries(gpd.points_from_xy(s.lon, s.lat), crs=4326).to_crs(3435)
    clips["cx"], clips["cy"] = centres.x.values, centres.y.values

    edges = gpd.read_file(args.edges).to_crs(3435)
    edges["osmid"] = edges["osmid"].astype(str)
    edges = edges[~edges.geometry.normalize().duplicated()].reset_index(drop=True)

    margin_ft = args.margin_m * 3.28084
    rows, truncated = [], 0
    sidx = edges.sindex
    for c in clips.itertuples():
        inner = c.geometry.buffer(-margin_ft)
        cand = edges.iloc[sidx.query(c.geometry)]
        cand = cand[cand.intersects(c.geometry)]
        inside = cand[cand.within(inner)]
        truncated += len(cand) - len(inside)
        for e in inside.itertuples():
            mid = e.geometry.interpolate(0.5, normalized=True)
            rows.append({"osmid": e.osmid, "sample_id": c.sample_id,
                         "d_centre": ((mid.x - c.cx) ** 2 + (mid.y - c.cy) ** 2) ** 0.5,
                         "geometry": e.geometry})
    u = gpd.GeoDataFrame(rows, crs=3435)
    if u.empty:
        raise SystemExit("no OSM crossing edge lies wholly inside any clip: check --edges")
    u = u.sort_values("d_centre").drop_duplicates("osmid").drop(columns="d_centre")
    u = u.merge(clips[["sample_id", *STRATA, "added"]], on="sample_id")
    u = u.sort_values(["sample_id", "osmid"]).reset_index(drop=True)
    u.insert(0, "unit_id", u["osmid"])

    u.drop(columns="geometry").to_csv(os.path.join(args.out_dir, "units.csv"), index=False)
    sub = u.drop(columns="geometry").groupby(STRATA).sample(frac=args.google_frac,
                                                            random_state=args.seed)
    sub[["unit_id", "sample_id", *STRATA]].sort_values(["sample_id", "unit_id"]).to_csv(
        os.path.join(args.out_dir, "gt_google_subset.csv"), index=False)
    clips.drop(columns=["cx", "cy"]).to_file(gpkg, layer="clips", driver="GPKG")
    u[["unit_id", "sample_id", "geometry"]].to_file(gpkg, layer="units", driver="GPKG")

    import pyogrio
    layers = pyogrio.list_layers(gpkg)[:, 0].tolist()
    if "gt_crossings" in layers and not args.overwrite_gt:
        n = len(gpd.read_file(gpkg, layer="gt_crossings"))
        print(f"kept existing gt_crossings layer ({n} lines); use --overwrite-gt to replace it")
    else:
        empty = gpd.GeoDataFrame(
            {"unit_id": pd.Series(dtype="str"), "piece": pd.Series(dtype="int32"),
             "island": pd.Series(dtype="str"), "curb_visible": pd.Series(dtype="str"),
             "imagery": pd.Series(dtype="str"), "measurer": pd.Series(dtype="str"),
             "notes": pd.Series(dtype="str")},
            geometry=gpd.GeoSeries([], crs=3435))
        empty.to_file(gpkg, layer="gt_crossings", driver="GPKG", engine="pyogrio",
                      geometry_type="LineString")

    per = u.groupby("sample_id").size()
    print(f"{len(u)} crossing units at {per.size} of {len(clips)} clips "
          f"(median {per.median():.0f}, max {per.max()} per clip); "
          f"{truncated} edges cut by a clip edge were left out")
    print(u.groupby(STRATA).size().rename("units").to_string())
    missing = sorted(set(clips.sample_id) - set(u.sample_id))
    if missing:
        print(f"{len(missing)} clips have no whole crossing edge:", ", ".join(missing))
    print(f"{len(sub)} units in the Google re-measurement subset -> {args.out_dir}/gt_google_subset.csv")
    print(f"-> {args.out_dir}/units.csv, {gpkg}")


if __name__ == "__main__":
    main()
