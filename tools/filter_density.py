#!/usr/bin/env python3
"""Keep only tiles at or above a minimum instance count.

Runs A+B showed in-domain detector pretraining buying convergence speed but no
final accuracy, and attributed that to a density mismatch: the public corpus
averages 13.2 instances/tile against IGN's 77.0. This filters the corpus toward
the target's density so the explanation can be tested rather than asserted.

Filtering trades density for volume, so the follow-up run matches *gradient
steps* rather than epochs. Density and data quantity would otherwise move
together, which is the confound that made runs 3 and 4 unattributable.
"""
import json, sys
from collections import Counter
from pathlib import Path


def main(src, dst, min_inst):
    d = json.loads(Path(src).read_text())
    per = Counter(a["image_id"] for a in d["annotations"])
    keep = [i for i in d["images"] if per.get(i["id"], 0) >= min_inst]
    kept = {i["id"] for i in keep}
    anns = [a for a in d["annotations"] if a["image_id"] in kept]
    assert keep, f"no tiles with >= {min_inst} instances"
    mean = len(anns) / len(keep)
    Path(dst).write_text(json.dumps(dict(d, images=keep, annotations=anns)))
    print(f"{Path(src).parent.name}: {len(d['images'])} -> {len(keep)} tiles, "
          f"{len(d['annotations'])} -> {len(anns)} instances, "
          f"mean {mean:.1f}/tile (IGN target 77.0)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
