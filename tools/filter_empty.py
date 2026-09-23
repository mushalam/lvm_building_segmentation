#!/usr/bin/env python3
"""Drop images with no annotations from a COCO file.

The public pretraining corpus has 832 of 6,764 images (12.3%) containing no
buildings at all, while the IGN target has 1 in 4,903. A pretraining stage is
only useful insofar as its distribution resembles the target's, and 12%
all-background tiles teach the detector to suppress detections -- the opposite
of what a 77-instances-per-image target needs.
"""
import json, sys
from collections import Counter
from pathlib import Path


def main(src, dst):
    d = json.loads(Path(src).read_text())
    per = Counter(a["image_id"] for a in d["annotations"])
    keep = [i for i in d["images"] if per.get(i["id"], 0) > 0]
    kept_ids = {i["id"] for i in keep}
    anns = [a for a in d["annotations"] if a["image_id"] in kept_ids]
    assert len(anns) == len(d["annotations"]), (
        f"dropped {len(d['annotations']) - len(anns)} annotations, expected 0 "
        "-- empty images cannot own annotations")
    out = dict(d, images=keep, annotations=anns)
    Path(dst).write_text(json.dumps(out))
    print(f"{src}\n  images {len(d['images'])} -> {len(keep)} "
          f"(dropped {len(d['images']) - len(keep)})")
    print(f"  annotations {len(d['annotations'])} (unchanged)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
