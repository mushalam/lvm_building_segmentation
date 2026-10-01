#!/usr/bin/env python3
"""Build a dataset whose train split is the union of several splits.

    python tools/merge_coco.py --out data_local/ign_paris_idf/building \\
        --train data/ign_building_v2/building/train data_local/ign_idf/building/train \\
        --valid data/ign_building_v2/building/valid --test data/ign_building_v2/building/test

Images are symlinked (no copy). Image and annotation ids are renumbered. File
names must not collide across inputs. valid and test are linked as whole
directories, so they stay byte-identical to their source.

Before writing, every train image is MD5-hashed against every valid and
test image. Any identical tile aborts the merge. This is the leakage check
behind the geographic guard in build_ign_dataset.py, made independently.
"""
import argparse, hashlib, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def load(d):
    return json.loads((Path(d) / "_annotations.coco.json").read_text())


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--test", required=True)
    args = ap.parse_args()
    out = Path(args.out)

    imgs, anns, seen, src_paths = [], [], set(), []
    for d in args.train:
        doc = load(d)
        remap = {}
        for im in doc["images"]:
            assert im["file_name"] not in seen, f"file name collision: {im['file_name']}"
            seen.add(im["file_name"])
            remap[im["id"]] = len(imgs)
            imgs.append(dict(im, id=len(imgs)))
            src_paths.append(Path(d).resolve() / im["file_name"])
        for a in doc["annotations"]:
            anns.append(dict(a, id=len(anns), image_id=remap[a["image_id"]]))
        print(f"train += {d}: {len(doc['images'])} images, {len(doc['annotations'])} annotations")

    held = [Path(args.valid).resolve() / i["file_name"] for i in load(args.valid)["images"]] + \
           [Path(args.test).resolve() / i["file_name"] for i in load(args.test)["images"]]
    with ThreadPoolExecutor(32) as ex:
        h_train = dict(zip(ex.map(md5, src_paths), src_paths))
        h_held = dict(zip(ex.map(md5, held), held))
    leak = sorted(set(h_train) & set(h_held))
    if leak:
        sys.exit(f"LEAKAGE: {len(leak)} train tiles are byte-identical to valid/test, e.g. "
                 f"{h_train[leak[0]]} == {h_held[leak[0]]}")
    print(f"leakage check: 0 of {len(src_paths)} train tiles identical to any of "
          f"{len(held)} valid/test tiles")

    (out / "train").mkdir(parents=True, exist_ok=True)
    for im, p in zip(imgs, src_paths):
        link = out / "train" / im["file_name"]
        if not link.exists():
            os.symlink(p, link)
    doc = {"info": {"description": "merged train: " + " + ".join(args.train)},
           "categories": [{"id": 1, "name": "building", "supercategory": "none"}],
           "images": imgs, "annotations": anns}
    (out / "train" / "_annotations.coco.json").write_text(json.dumps(doc))
    for split, src in (("valid", args.valid), ("test", args.test)):
        link = out / split
        if not link.exists():
            os.symlink(Path(src).resolve(), link)
    print(f"wrote {out}: train {len(imgs)} images / {len(anns)} annotations; "
          f"valid -> {args.valid}; test -> {args.test}")


if __name__ == "__main__":
    main()
