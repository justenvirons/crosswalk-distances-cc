"""Restrict the evaluation sample to street-street intersections and top it up (server).

A point is eligible if
  1. it lies within --max-dist-ft of a drivable OSM centre-line (parking-lot and
     driveway nodes fail this), and
  2. drivable OSM roads within --radius-ft of it run in at least --min-dirs
     directions that differ by more than --tol-deg (alley mouths, driveways and
     mid-block points fail this: alleys and service roads are not "drivable" here).

Ineligible sample points are dropped and each stratum (geography x canopy x tile
position) is refilled to its original size from eligible, unused candidates in the
same stratum (pilot points excluded). Canopy terciles and the tile split are
recomputed from the candidate pool exactly as in 02_select_sample.py / Appendix A,
and checked against the labels stored in sample.csv.

Needs drivable centre-lines around every candidate, e.g.
  uv run python 02b_osm_roads.py --inputs ../data/sample/candidates.csv \
      --out ../data/sample/osm_roads_pool.geojson

Usage:
  uv run python 02c_street_screen.py --roads ../data/sample/osm_roads_pool.geojson

Writes to --out-dir:
  pool_screen.csv       every candidate with dist_road_ft, n_dirs, eligible, stratum
  sample_excluded.csv   dropped sample points and the reason
  sample_added.csv      replacement points (fetch, segment and grow-cut these)
  sample_screened.csv   the final evaluation sample (kept + added, column `added`)
"""

import argparse

import numpy as np
import pandas as pd

STRATA = ["geography", "canopy", "tile_pos"]


def assign_strata(cand, ndvi):
    pool = cand.merge(ndvi[["id", "veg_frac", "ndvi_mean"]], on="id")
    breaks = np.quantile(pool["veg_frac"], [1 / 3, 2 / 3])  # = R quantile type 7
    pool["canopy"] = pd.cut(pool["veg_frac"], [-1, *breaks, 2],
                            labels=["low", "mid", "high"]).astype(str)
    pool["tile_pos"] = np.where(pool["tile_offset"] >= pool["tile_offset"].median(),
                                "edge", "centre")
    return pool


def local_bearings(pt, near, step_ft=10.0):
    """Direction (0-180 deg) of each nearby line where it passes closest to pt."""
    out = []
    for g in near.geometry:
        for ln in getattr(g, "geoms", [g]):
            d = ln.project(pt)
            a = ln.interpolate(max(d - step_ft, 0.0))
            b = ln.interpolate(min(d + step_ft, ln.length))
            if a.distance(b) > 0:
                out.append(np.degrees(np.arctan2(b.y - a.y, b.x - a.x)) % 180)
    return out


def n_directions(bearings, tol_deg):
    dirs = []
    for x in bearings:
        if all(min(abs(x - y), 180 - abs(x - y)) > tol_deg for y in dirs):
            dirs.append(x)
    return len(dirs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roads", required=True)
    ap.add_argument("--candidates", default="../data/sample/candidates.csv")
    ap.add_argument("--pool-ndvi", default="../data/sample/pool_ndvi.csv")
    ap.add_argument("--sample", default="../data/sample/sample.csv")
    ap.add_argument("--pilot", default="../data/sample/pilot.csv")
    ap.add_argument("--out-dir", default="../data/sample")
    ap.add_argument("--max-dist-ft", type=float, default=50.0)
    ap.add_argument("--radius-ft", type=float, default=60.0)
    ap.add_argument("--min-dirs", type=int, default=2)
    ap.add_argument("--tol-deg", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=44)
    args = ap.parse_args()

    import geopandas as gpd

    cand = pd.read_csv(args.candidates)
    ndvi = pd.read_csv(args.pool_ndvi)
    if "error" in ndvi:
        ndvi = ndvi[ndvi["error"].isna()]
    sample, pilot = pd.read_csv(args.sample), pd.read_csv(args.pilot)

    pool = assign_strata(cand, ndvi)
    chk = sample.merge(pool[["id", "canopy", "tile_pos"]], on="id", suffixes=("", "_re"))
    if len(chk) != len(sample) or not ((chk.canopy == chk.canopy_re).all()
                                       and (chk.tile_pos == chk.tile_pos_re).all()):
        raise SystemExit("recomputed strata do not match sample.csv: check the input files")

    roads = gpd.read_file(args.roads).to_crs(3435)            # Illinois East, US ft
    pts = gpd.GeoDataFrame(pool, geometry=gpd.points_from_xy(pool.lon, pool.lat),
                           crs=4326).to_crs(3435)
    sidx = roads.sindex
    dist, ndirs = [], []
    for p in pts.geometry:
        idx = sidx.query(p.buffer(max(args.radius_ft, args.max_dist_ft) * 4))
        near_all = roads.iloc[idx]
        dist.append(float(near_all.distance(p).min()) if len(near_all) else np.inf)
        near = near_all[near_all.distance(p) <= args.radius_ft]
        ndirs.append(n_directions(local_bearings(p, near), args.tol_deg))
    pool["dist_road_ft"] = np.round(dist, 1)
    pool["n_dirs"] = ndirs
    no_roads = int(np.isinf(pool["dist_road_ft"]).sum())
    if no_roads:
        print(f"WARNING: {no_roads} candidates have no OSM road nearby: is --roads built "
              "from candidates.csv?")
    pool["reason"] = np.select(
        [pool.dist_road_ft > args.max_dist_ft, pool.n_dirs < args.min_dirs],
        [f"> {args.max_dist_ft:g} ft from a drivable centre-line",
         f"fewer than {args.min_dirs} road directions within {args.radius_ft:g} ft"], "")
    pool["eligible"] = pool["reason"] == ""
    pool.to_csv(f"{args.out_dir}/pool_screen.csv", index=False)

    in_sample = pool["id"].isin(sample["id"])
    excluded = pool[in_sample & ~pool.eligible]
    kept = pool[in_sample & pool.eligible]
    target = sample.groupby(STRATA).size()

    spare = pool[pool.eligible & ~in_sample & ~pool["id"].isin(pilot["id"])
                 & (pool.canopy != "mid")]
    rng = np.random.default_rng(args.seed)
    added, rows = [], []
    for key, n in target.items():
        have = kept.set_index(STRATA).index.isin([key]).sum()
        pick = spare.set_index(STRATA).index.isin([key])
        cands = spare[pick]
        need = n - have
        take = cands.iloc[rng.permutation(len(cands))[:max(need, 0)]]
        added.append(take)
        rows.append({**dict(zip(STRATA, key)), "target": n, "kept": have,
                     "added": len(take), "final": have + len(take),
                     "spare_eligible": len(cands)})
    added = pd.concat(added)
    cols = list(sample.columns)
    final = pd.concat([kept[cols].assign(added=False), added[cols].assign(added=True)])

    excluded[["id", "lat", "lon", *STRATA, "dist_road_ft", "n_dirs", "reason"]].to_csv(
        f"{args.out_dir}/sample_excluded.csv", index=False)
    added[cols].to_csv(f"{args.out_dir}/sample_added.csv", index=False)
    final.to_csv(f"{args.out_dir}/sample_screened.csv", index=False)

    elig = pool[pool.canopy != "mid"]
    print(f"candidates eligible: {int(pool.eligible.sum())} of {len(pool)} "
          f"(strata only: {int(elig.eligible.sum())} of {len(elig)})")
    print(f"sample: {len(kept)} kept, {len(excluded)} excluded "
          f"({(excluded.reason.str.startswith('>')).sum()} off-street, "
          f"{(excluded.reason.str.startswith('fewer')).sum()} not a street-street junction)")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nfinal sample {len(final)}; {len(added)} new points -> {args.out_dir}/sample_added.csv")
    short = [r for r in rows if r["final"] < r["target"]]
    if short:
        print("WARNING: strata still short:", short)


if __name__ == "__main__":
    main()
