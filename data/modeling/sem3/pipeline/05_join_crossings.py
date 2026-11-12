"""Score each imagery source's grow-cut output against the crossing units (server).

Each grow-cut line is assigned to the unit edge it runs along: its direction within
--max-angle of the edge's, and either at least --min-overlap of the line within
--buffer-ft of the edge, or at least --min-overlap of the edge within --buffer-ft
of the line (so a line that overshoots the curbs still measures its crossing, with
the overshoot scored as error). Lines inside a sample clip that run along no unit
edge are classed as on a crossing cut by the clip edge (not a unit; not counted)
or as false detections (along no OSM crossing edge). Pieces assigned to one unit are ordered along
the edge; pieces separated by less than --merge-gap-ft are one crossing (the gap is
added to the length), larger gaps are reported as separate groups (refuge island or
a hole in the mask; the ground truth decides which). Model errors are not edited.

Inputs: 04_crossing_units.py output, refined_crosswalks_v3_0_4.csv per source
(03_grow_cut.sh), and, once measured, the gt_crossings layer of groundtruth.gpkg.
Ground-truth lines are split by their `imagery` field: --gt-imagery (county) gives
gt_len_ft, the reference; --check-imagery (google) gives gt_google_len_ft, the
re-measurement of the random subset in gt_google_subset.csv.

Writes to --out-dir:
  crossing_status.csv     per unit: found/missed and piece counts per source,
                          ground truth measured or not (no lengths; safe to open)
  false_detections.csv    lines matching no unit, per source and clip
  crossing_lengths.csv    per unit: model and ground-truth lengths. Do not open
                          until the ground-truth pass is finished.

Usage:
  uv run python 05_join_crossings.py
"""

import argparse
import os

import numpy as np
import pandas as pd

from cc_common import RADIUS_M, bbox_lonlat

STRATA = ["geography", "canopy", "tile_pos"]


def bearing(a, b):
    return np.degrees(np.arctan2(b[1] - a[1], b[0] - a[0])) % 180


def angle_diff(x, y):
    d = abs(x - y) % 180
    return min(d, 180 - d)


def edge_bearing_at(edge, pt, step=5.0):
    d = edge.project(pt)
    a = edge.interpolate(max(d - step, 0.0))
    b = edge.interpolate(min(d + step, edge.length))
    return bearing((a.x, a.y), (b.x, b.y))


def read_lines(path):
    import geopandas as gpd
    from shapely import wkt
    c = pd.read_csv(path)
    g = c["geometry"].map(wkt.loads)
    x0 = abs(g.iloc[0].coords[0][0])
    crs = 4326 if x0 <= 180 else 3857
    return gpd.GeoDataFrame(c[["length_ft"]], geometry=g, crs=crs).to_crs(3435)


def assign(lines, units, buffer_ft, min_overlap, max_angle):
    """unit_id (or None) and overlap fraction for each line."""
    sidx = units.sindex
    out_id, out_frac = [], []
    for ln in lines.geometry:
        if ln.length == 0:
            out_id.append(None); out_frac.append(0.0); continue
        lb = bearing(ln.coords[0], ln.coords[-1])
        mid = ln.interpolate(0.5, normalized=True)
        best = (None, 0.0, np.inf)
        for j in sidx.query(ln.buffer(buffer_ft)):
            e = units.geometry.iloc[j]
            frac = max(ln.intersection(e.buffer(buffer_ft)).length / ln.length,
                       e.intersection(ln.buffer(buffer_ft)).length / max(e.length, 1e-9))
            if frac < min_overlap:
                continue
            if angle_diff(lb, edge_bearing_at(e, mid)) > max_angle:
                continue
            dist = e.distance(mid)
            if (frac, -dist) > (best[1], -best[2]):
                best = (units.unit_id.iloc[j], frac, dist)
        out_id.append(best[0]); out_frac.append(best[1])
    return out_id, out_frac


def combine(pieces, edge, merge_gap_ft):
    """Model length, number of separate groups and largest gap for one unit."""
    iv = []
    for ln, L in zip(pieces.geometry, pieces.length_ft):
        s0, s1 = sorted((edge.project(ln.interpolate(0)), edge.project(ln.interpolate(1, normalized=True))))
        iv.append([s0, s1, L])
    iv.sort()
    total, groups, max_gap, end = iv[0][2], 1, 0.0, iv[0][1]
    for s0, s1, L in iv[1:]:
        gap = max(s0 - end, 0.0)
        max_gap = max(max_gap, gap)
        if gap >= merge_gap_ft:
            groups += 1
        else:
            total += gap
        total += L
        end = max(end, s1)
    return total, groups, max_gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="../data/groundtruth")
    ap.add_argument("--sample", default="../data/sample/sample_screened.csv")
    ap.add_argument("--results-root", default="../data/results")
    ap.add_argument("--edges", default="../data/osm/crosswalk_edges_v2_0.shp")
    ap.add_argument("--sources", default="A:A_google,B:B_cook_rgb")
    ap.add_argument("--run", default="v1")
    ap.add_argument("--buffer-ft", type=float, default=15.0)
    ap.add_argument("--min-overlap", type=float, default=0.6)
    ap.add_argument("--max-angle", type=float, default=20.0)
    ap.add_argument("--merge-gap-ft", type=float, default=3.0)
    ap.add_argument("--gt-imagery", default="county",
                    help="imagery value of the primary ground-truth lines")
    ap.add_argument("--check-imagery", default="google",
                    help="imagery value of the re-measurement lines (subset)")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    import geopandas as gpd
    import pyogrio
    from shapely.geometry import box

    out_dir = args.out_dir or args.gt_dir
    gpkg = os.path.join(args.gt_dir, "groundtruth.gpkg")
    units = gpd.read_file(gpkg, layer="units")
    units["unit_id"] = units["unit_id"].astype(str)
    meta = pd.read_csv(os.path.join(args.gt_dir, "units.csv"), dtype={"unit_id": str})
    table = meta[["unit_id", "sample_id", *STRATA, "added"]].copy()
    geom = dict(zip(units.unit_id, units.geometry))

    s = pd.read_csv(args.sample)
    clips = gpd.GeoDataFrame({"sample_id": s["id"].astype(str)},
                             geometry=[box(*bbox_lonlat(la, lo, RADIUS_M)) for la, lo in zip(s.lat, s.lon)],
                             crs=4326).to_crs(3435)

    edges = gpd.read_file(args.edges).to_crs(3435)
    edges["unit_id"] = edges["osmid"].astype(str)
    others = edges[~edges.unit_id.isin(set(units.unit_id))].reset_index(drop=True)

    false_rows = []
    for item in args.sources.split(","):
        key, folder = item.split(":")
        lines = read_lines(os.path.join(args.results_root, folder, args.run,
                                        "refined_crosswalks_v3_0_4.csv"))
        lines["unit_id"], lines["overlap"] = assign(lines, units, args.buffer_ft,
                                                    args.min_overlap, args.max_angle)
        mids = lines.copy()
        mids["geometry"] = lines.geometry.interpolate(0.5, normalized=True)
        in_clip = gpd.sjoin(mids, clips, predicate="within", how="left")
        in_clip = in_clip[~in_clip.index.duplicated()]
        loose = in_clip[in_clip.unit_id.isna() & in_clip.sample_id.notna()]
        on_other, _ = assign(lines.loc[loose.index], others, args.buffer_ft,
                             args.min_overlap, args.max_angle)
        loose = loose.assign(kind=np.where(pd.notna(on_other), "crossing cut by clip edge",
                                           "false detection"))
        fd = loose[loose.kind == "false detection"]
        for (sid, kind), n in loose.groupby(["sample_id", "kind"]).size().items():
            false_rows.append({"source": key, "sample_id": sid, "kind": kind, "n_lines": int(n)})

        recs = []
        for uid, g in lines[lines.unit_id.notna()].groupby("unit_id"):
            L, groups, gap = combine(g, geom[uid], args.merge_gap_ft)
            recs.append({"unit_id": uid, f"{key}_len_ft": L, f"{key}_lines": len(g),
                         f"{key}_groups": groups, f"{key}_max_gap_ft": gap})
        r = pd.DataFrame(recs, columns=["unit_id", f"{key}_len_ft", f"{key}_lines",
                                        f"{key}_groups", f"{key}_max_gap_ft"])
        table = table.merge(r, on="unit_id", how="left")
        table[f"{key}_status"] = np.where(table[f"{key}_len_ft"].notna(), "found", "missed")
        print(f"{key}: {len(lines)} lines; {int(lines.unit_id.notna().sum())} on units, "
              f"{len(fd)} false detections and {len(loose) - len(fd)} on crossings cut by "
              f"the clip edge in sample clips, "
              f"{int((lines.unit_id.isna() & in_clip.sample_id.isna()).sum())} outside sample clips | "
              f"units found {int((table[f'{key}_status'] == 'found').sum())} of {len(table)}, "
              f"split into 2+ groups {int((table[f'{key}_groups'] > 1).sum())}")

    gt_n = 0
    gt_cols = []
    if "gt_crossings" in pyogrio.list_layers(gpkg)[:, 0].tolist():
        gt = gpd.read_file(gpkg, layer="gt_crossings")
        if len(gt):
            gt = gt.to_crs(3435)
            gt["unit_id"] = gt["unit_id"].astype(str)
            gt["imagery"] = gt["imagery"].fillna("").astype(str).str.strip().str.lower()
            gt["len"] = gt.length
            bad = sorted(set(gt.imagery) - {args.gt_imagery, args.check_imagery})
            if bad:
                print(f"WARNING: gt_crossings lines with imagery {bad} are ignored; "
                      f"use '{args.gt_imagery}' or '{args.check_imagery}'")
            unknown = sorted(set(gt.unit_id) - set(table.unit_id))
            if unknown:
                print(f"WARNING: {len(unknown)} gt_crossings unit_id values match no unit:", unknown[:10])
            for img, pre in ((args.gt_imagery, "gt"), (args.check_imagery, f"gt_{args.check_imagery}")):
                g = gt[gt.imagery == img]
                if g.empty:
                    continue
                agg = g.groupby("unit_id").agg(
                    **{f"{pre}_len_ft": ("len", "sum"), f"{pre}_pieces": ("len", "size"),
                       f"{pre}_island": ("island", lambda x: "yes" if (x == "yes").any() else "no"),
                       f"{pre}_curb_visible": ("curb_visible", lambda x: "/".join(sorted(set(map(str, x)))))})
                table = table.merge(agg.reset_index(), on="unit_id", how="left")
                gt_cols.append(pre)
            if "gt" in gt_cols:
                gt_n = int(table.gt_len_ft.notna().sum())
    if "gt_len_ft" not in table:
        table["gt_len_ft"] = np.nan
    table["gt_status"] = np.where(table.gt_len_ft.notna(), "measured", "not yet")
    chk = f"gt_{args.check_imagery}"
    sub_path = os.path.join(args.gt_dir, f"gt_{args.check_imagery}_subset.csv")
    if os.path.exists(sub_path):
        sub = set(pd.read_csv(sub_path, dtype={"unit_id": str}).unit_id)
        table[f"in_{args.check_imagery}_subset"] = table.unit_id.isin(sub)
        done = int(table[f"{chk}_len_ft"].notna().sum()) if f"{chk}_len_ft" in table else 0
        extra = int((table[f"{chk}_len_ft"].notna() & ~table[f"in_{args.check_imagery}_subset"]).sum()) \
            if f"{chk}_len_ft" in table else 0
        print(f"{args.check_imagery} re-measurement: {done} of {len(sub)} subset units"
              + (f" (+{extra} outside the subset)" if extra else ""))
    print(f"ground truth: {gt_n} of {len(table)} units measured")

    keys = [i.split(":")[0] for i in args.sources.split(",")]
    status_cols = ["unit_id", "sample_id", *STRATA, "added", "gt_status"] + \
        [c for c in table.columns if c.startswith("in_") and c.endswith("_subset")] + \
        [c for k in keys for c in (f"{k}_status", f"{k}_lines", f"{k}_groups")]
    table[status_cols].to_csv(os.path.join(out_dir, "crossing_status.csv"), index=False)
    pd.DataFrame(false_rows, columns=["source", "sample_id", "kind", "n_lines"]).to_csv(
        os.path.join(out_dir, "false_detections.csv"), index=False)
    table.to_csv(os.path.join(out_dir, "crossing_lengths.csv"), index=False)

    both = (table[[f"{k}_status" for k in keys]] == "found").all(axis=1)
    print(f"found by all sources: {int(both.sum())} of {len(table)}")
    print(f"-> {out_dir}/crossing_status.csv, false_detections.csv, "
          "crossing_lengths.csv (lengths: open only after the ground-truth pass)")


if __name__ == "__main__":
    main()
