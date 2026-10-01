#!/usr/bin/env python3
"""Download and unpack IGN products from the Géoplateforme download service.

    python tools/fetch_ign.py BDORTHO_2-0_IRC-0M20_JP2-E080_LAMB93_D092_2024-01-01 \\
        BDTOPO_3-5_TOUSTHEMES_GPKG_LAMB93_D092_2026-06-15 --out data_local/ign_raw

Each dataset name is a resource at data.geopf.fr/telechargement/resource/
<PRODUCT>/<name>. Its Atom listing names a single .7z and an .md5. The archive
is downloaded with resume, checked against the md5, then unpacked to
<out>/<name>/. Already-verified archives are not fetched again.

Uses data.geopf.fr (the Géoplateforme API), not geoportail.gouv.fr, which
closed on 2026-09-30. Needs requests and py7zr (the .venv_geo environment).
"""
import argparse, hashlib, re, sys, time
from pathlib import Path

import requests

RESOURCE = "https://data.geopf.fr/telechargement/resource"
S = requests.Session(); S.headers["User-Agent"] = "lvm_EOSC-fetch/1.0"


def files_of(name):
    product = name.split("_")[0]
    xml = S.get(f"{RESOURCE}/{product}/{name}", timeout=60).text
    links = re.findall(r'href="(https://data\.geopf\.fr/telechargement/download/[^"]+)"', xml)
    arch = [u for u in links if u.endswith(".7z") or re.search(r"\.7z\.\d+$", u)]
    md5 = [u for u in links if u.endswith(".md5")]
    if not arch:
        sys.exit(f"{name}: no .7z in listing")
    return sorted(arch), (md5[0] if md5 else None)


def download(url, dst):
    have = dst.stat().st_size if dst.exists() else 0
    head = S.head(url, timeout=60, allow_redirects=True)
    total = int(head.headers.get("Content-Length", 0))
    if total and have == total:
        return
    hdr = {"Range": f"bytes={have}-"} if have else {}
    with S.get(url, headers=hdr, stream=True, timeout=120) as r:
        r.raise_for_status()
        mode = "ab" if have and r.status_code == 206 else "wb"
        done, t0 = (have if mode == "ab" else 0), time.time()
        with open(dst, mode) as f:
            for chunk in r.iter_content(8 << 20):
                f.write(chunk); done += len(chunk)
        print(f"  {dst.name}: {done / 2**30:.2f} GB in {time.time() - t0:.0f}s", flush=True)


def md5sum(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--out", default="data_local/ign_raw")
    args = ap.parse_args()
    import py7zr

    for name in args.names:
        dest = Path(args.out) / name
        dest.mkdir(parents=True, exist_ok=True)
        archives, md5url = files_of(name)
        print(f"{name}: {len(archives)} archive part(s)", flush=True)
        expected = {}
        r = S.get(md5url, timeout=60) if md5url else None
        # The listing can advertise an .md5 that 404s ("content no longer
        # exists", seen for BD TOPO 2026-06-15). Integrity then rests on the
        # per-file CRCs that py7zr verifies while extracting.
        if r is not None and r.status_code == 200 and not r.text.lstrip().startswith("{"):
            for line in r.text.splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    expected[parts[-1].lstrip("*")] = parts[0].lower()
        for url in archives:
            f = dest / url.rsplit("/", 1)[1]
            download(url, f)
            want = expected.get(f.name)
            got = md5sum(f)
            if want and got != want:
                f.unlink()
                sys.exit(f"  md5 MISMATCH for {f.name}: got {got}, want {want} -- deleted, rerun")
            print(f"  {f.name}: md5 {'verified' if want else 'unavailable, relying on 7z CRCs'} "
                  f"({got})", flush=True)
        marker = dest / ".extracted"
        if not marker.exists():
            first = sorted(dest.glob("*.7z*"))[0]
            with py7zr.SevenZipFile(first, "r") as z:
                z.extractall(dest)
            marker.touch()
        n = sum(1 for _ in dest.rglob("*.jp2")) + sum(1 for _ in dest.rglob("*.gpkg"))
        print(f"  extracted -> {dest} ({n} .jp2/.gpkg files)", flush=True)


if __name__ == "__main__":
    main()
