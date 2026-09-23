#!/usr/bin/env python3
"""Remap a COCO file's category_id to match the target dataset's.

The public pretraining corpus is a Roboflow export using category_id 0; the
IGN target uses 1. lvm.train.validate and lvm.score both emit predictions with
category_id 1, so scoring the public corpus produced segm AP of exactly 0.0000
-- not a weak model but a category mismatch, with every prediction filed under
a class the ground truth does not contain.

Worth noting how this presents: AP 0.0000 reads as "the model learned nothing",
which is a plausible enough story for a from-scratch stage that it invites the
wrong diagnosis. The tell is that it is exactly zero rather than merely small.
"""
import json, sys
from pathlib import Path


def main(path, new_id=1):
    d = json.loads(Path(path).read_text())
    old = sorted({a["category_id"] for a in d["annotations"]})
    assert len(old) == 1, f"expected one category, found {old}"
    if old[0] == new_id:
        print(f"{path}: already category_id {new_id}, unchanged")
        return
    for a in d["annotations"]:
        a["category_id"] = new_id
    for c in d["categories"]:
        c["id"] = new_id
    assert len(d["categories"]) == 1, d["categories"]
    Path(path).write_text(json.dumps(d))
    print(f"{path}: category_id {old[0]} -> {new_id} "
          f"({len(d['annotations'])} annotations)")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
