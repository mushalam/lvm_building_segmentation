#!/usr/bin/env python3
"""Build COCO building-instance tiles from IGN BD ORTHO sheets + BD TOPO.

A port of the generator that produced ign_building_v2 (LVM_datasets/
create_instance_seg_dataset_v2.py on the team share), with the same
conventions so new tiles match v2's labels exactly:

  * 1024 px tiles stepped from each JP2 sheet's origin; partial edge tiles
    dropped
  * every BATIMENT polygon intersecting a tile is clipped to it and rasterised
    with all_touched=True; instances under 4 px are dropped; tiles left with no
    instance are skipped
  * the first three image bands, as uint8 PNG (for the IRC product: NIR, R, G)

Two guards v2 did not need, because new départements border Paris:

  * --within: keep a tile only if it lies entirely inside the union of these
    départements (from the GPKGs' DEPARTEMENT layer). This removes every tile
    touching Paris -- so no pixel can overlap a v2 train/valid/test tile -- and
    every tile reaching into a neighbour whose buildings we have no labels
    for, which would otherwise teach the model that real buildings are
    background.
  * --exclude-sheets: names of JP2 sheets whose footprints no tile may touch
    (v2's 13 Paris sheets), checked independently as a second line.

    python tools/build_ign_dataset.py \\
        --jp2 data_local/ign_raw/BDORTHO_*_D09[234]_2024-01-01 \\
        --gpkg data_local/ign_raw/BDTOPO_*_D09[234]_2026-06-15 \\
        --within 92 93 94 --exclude-sheets-file tools/v2_paris_sheets.txt \\
        --prefix idf --out data_local/ign_idf/building/train
"""
import argparse, json, re, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

TILE, MIN_PX = 1024, 4
_G = {}


def sheet_bbox(name):
    """IGN sheet name -> Lambert-93 bbox. 'XXXX-YYYY' is the sheet's top-left
    corner in km; sheets are 5 km square. Verified against the JP2
    georeferencing in main() before it is used."""
    m = re.search(r"-(\d{4})-(\d{4})-LA93", name)
    x, y = int(m.group(1)) * 1000, int(m.group(2)) * 1000
    return (x, y - 5000, x + 5000, y)


def _init(gpkgs, within, excluded):
    import geopandas as gpd, pandas as pd
    from shapely.geometry import box
    from shapely.ops import unary_union
    from shapely.prepared import prep
    b = pd.concat([gpd.read_file(g, layer="batiment", columns=["geometry"]) for g in gpkgs],
                  ignore_index=True)
    b = b[b.geometry.notna() & ~b.geometry.is_empty]
    _G["b"] = gpd.GeoDataFrame(b, geometry="geometry", crs=b.crs)
    _G["sindex"] = _G["b"].sindex
    deps = pd.concat([gpd.read_file(g, layer="departement") for g in gpkgs], ignore_index=True)
    col = next(c for c in ("code_insee", "insee_dep", "code") if c in deps.columns)
    keep = deps[deps[col].astype(str).isin(within)]
    _G["within"] = prep(unary_union(keep.geometry.values))
    _G["excluded"] = [box(*bb) for bb in excluded]
    _G["dep_codes"] = sorted(set(keep[col].astype(str)))


def _parts(g):
    if g is None or g.is_empty: return []
    if g.geom_type == "Polygon": return [g]
    if g.geom_type in ("MultiPolygon", "GeometryCollection"):
        return [p for s in g.geoms for p in _parts(s)]
    return []


def _sheet(job):
    jp2, out_dir, prefix, sheet_idx = job
    import rasterio
    from rasterio.features import rasterize
    from rasterio.windows import Window, bounds as wbounds
    from shapely.geometry import box
    from PIL import Image
    recs, stats = [], dict(tiles=0, outside=0, excluded=0, empty=0, kept=0, tiny=0)
    with rasterio.open(jp2) as src:
        b = _G["b"] if _G["b"].crs == src.crs else _G["b"].to_crs(src.crs)
        sidx = b.sindex
        for ty in range(src.height // TILE):
            for tx in range(src.width // TILE):
                stats["tiles"] += 1
                win = Window(tx * TILE, ty * TILE, TILE, TILE)
                tr = src.window_transform(win)
                tb = box(*wbounds(win, src.transform))
                if not _G["within"].contains(tb):
                    stats["outside"] += 1; continue
                if any(tb.intersects(e) for e in _G["excluded"]):
                    stats["excluded"] += 1; continue
                anns = []
                for i in sidx.query(tb, predicate="intersects"):
                    parts = _parts(b.geometry.iloc[i].intersection(tb))
                    if not parts: continue
                    m = rasterize([(p, 1) for p in parts], out_shape=(TILE, TILE), transform=tr,
                                  fill=0, dtype="uint8", all_touched=True)
                    n = int(m.sum())
                    if n < MIN_PX:
                        stats["tiny"] += 1; continue
                    ys, xs = np.where(m > 0)
                    inv = ~tr
                    seg = []
                    for p in parts:
                        ring = [round(c, 2) for x, y, *_ in p.exterior.coords for c in inv * (x, y)]
                        if len(ring) >= 6: seg.append(ring)
                    if not seg: continue
                    anns.append({"segmentation": seg, "area": n,
                                 "bbox": [int(xs.min()), int(ys.min()),
                                          int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
                                 "category_id": 1, "iscrowd": 0})
                if not anns:
                    stats["empty"] += 1; continue
                img = src.read(window=win)[:3].transpose(1, 2, 0)
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                name = f"{prefix}_{sheet_idx:03d}_{ty:02d}{tx:02d}.png"
                Image.fromarray(img).save(Path(out_dir) / name)
                recs.append({"file_name": name, "anns": anns,
                             "bounds": list(tb.bounds), "sheet": Path(jp2).name})
                stats["kept"] += 1
    return recs, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jp2", nargs="+", required=True, help="dirs searched recursively for *.jp2")
    ap.add_argument("--gpkg", nargs="+", required=True, help="dirs or files; *.gpkg found recursively")
    ap.add_argument("--within", nargs="+", required=True, help="département codes, e.g. 92 93 94")
    ap.add_argument("--exclude-sheets-file", help="text file, one JP2 sheet name per line")
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit-sheets", type=int, default=0, help="smoke test: first N sheets only")
    args = ap.parse_args()

    import rasterio
    find = lambda roots, pat: sorted({p for r in roots for p in
                                      ([Path(r)] if Path(r).is_file() else Path(r).rglob(pat))})
    jp2s, gpkgs = find(args.jp2, "*.jp2"), find(args.gpkg, "*.gpkg")
    assert jp2s and gpkgs, (len(jp2s), len(gpkgs))
    # Neighbouring départements' deliveries both carry the sheets on their
    # shared border -- the same 5 km grid square twice. Keep one per square,
    # or every border tile would appear twice in training.
    by_square = {}
    for j in jp2s:
        by_square.setdefault(sheet_bbox(j.name), j)
    dupes = len(jp2s) - len(by_square)
    jp2s = sorted(by_square.values())
    if args.limit_sheets:
        jp2s = jp2s[:args.limit_sheets]
    # the sheet-name convention must match the actual georeferencing
    for j in jp2s[:3]:
        with rasterio.open(j) as s:
            got = tuple(round(v) for v in s.bounds)
        assert got == sheet_bbox(j.name), f"{j.name}: bounds {got} != name-derived {sheet_bbox(j.name)}"
    excluded = []
    if args.exclude_sheets_file:
        excluded = [sheet_bbox(l.strip()) for l in open(args.exclude_sheets_file) if l.strip()]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"{len(jp2s)} sheets ({dupes} duplicate border squares dropped), {len(gpkgs)} GPKGs, "
          f"within {args.within}, {len(excluded)} excluded sheets; "
          f"sheet-name georeferencing verified", flush=True)

    t0, recs, tot = time.time(), [], {}
    with ProcessPoolExecutor(args.workers, initializer=_init,
                             initargs=([str(g) for g in gpkgs], args.within, excluded)) as ex:
        for k, (r, st) in enumerate(ex.map(_sheet, [(str(j), str(out), args.prefix, i)
                                                      for i, j in enumerate(jp2s)])):
            recs += r
            for a, v in st.items(): tot[a] = tot.get(a, 0) + v
            print(f"  [{k + 1}/{len(jp2s)}] {jp2s[k].name}: {st}  ({time.time() - t0:.0f}s)", flush=True)

    images, annotations = [], []
    for iid, r in enumerate(recs):
        images.append({"id": iid, "file_name": r["file_name"], "height": TILE, "width": TILE,
                       "bounds_lambert93": r["bounds"], "sheet": r["sheet"]})
        for a in r["anns"]:
            annotations.append(dict(a, id=len(annotations), image_id=iid))
    doc = {"info": {"description": f"IGN buildings, départements {' '.join(args.within)}",
                    "generator": "lvm_EOSC tools/build_ign_dataset.py",
                    "conventions": "as ign_building_v2: 1024 px, all_touched, >=4 px, no empty tiles"},
           "categories": [{"id": 1, "name": "building", "supercategory": "none"}],
           "images": images, "annotations": annotations}
    (out / "_annotations.coco.json").write_text(json.dumps(doc))
    print(f"done: {len(images)} tiles, {len(annotations)} buildings "
          f"({len(annotations) / max(1, len(images)):.1f}/tile); totals {tot}; "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
