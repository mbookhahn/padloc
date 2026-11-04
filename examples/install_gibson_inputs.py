#!/usr/bin/env python3
"""Install the Gibson floor plans and poses, which the fixture package does not ship.

    python examples/install_gibson_inputs.py SRC [--fixtures DIR]

The Gibson cells read ``map.png`` and ``poses.txt`` of the nine ``gibson_t`` test scenes of
F3Loc's Gibson Floorplan Localization Dataset (doi:10.3929/ethz-b-000670118). ``SRC`` holds
that dataset's per-scene zips (``SRC/Spencerville.zip``, ...) or the extracted scenes
(``SRC/<scene>/``, ``SRC/gibson_t/<scene>/`` or ``SRC/Gibson Floorplan Localization
Dataset/gibson_t/<scene>/``). Every file is checked against the SHA-256 of the copy behind
the paper's numbers and written next to the scene's depth fixtures.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = Path("Gibson Floorplan Localization Dataset") / "gibson_t"
SHA256 = {   # scene: (map.png, poses.txt)
    "Spencerville": ("ffa969235ad7e23ee3c7f1f2dec4436c03663ea1b262764c8dc4d2109abb7093", "9bcace966ad3ab1a6524036bce0a013fc4084d1e7d2fdbcba8a6e1dc66cd78c8"),
    "Spotswood": ("8d7dfd2d06bb70c1bc7420aed0c313c67adaa2b6baab6d43b9bbc2a1f898819b", "63996fc4d2d6a5896fa3fe0d1efe5026561b3a31afe0fb4111a8556f65a055ff"),
    "Springhill": ("2c361e95eab59e7a186fa83ccae248c6518caaab2af0760d65d5aecb0335f667", "7808b7bbb7dd354eaefdf8fa439825889b623b2ee22a24361fcc3a13e8aff150"),
    "Stilwell": ("6906a48b9c9ad1bf92aac5b0187c2ea8520532906db80ccad1bcdd032b91300f", "39d8a0d3954476b4517bc3c373c6dd14a9fadb34abb6cf7fea89567ee3c3294d"),
    "Stokes": ("e9cd5c9fca556dec80bdb31c1a92116b166700f87de81ff3ff350c0d42837acf", "f7a1b8bcea5caf4feb43714535443c789858fccd4e292f424a7250b0d30514b5"),
    "Sumas": ("a49f969a813ae9dfc42939d4b14ebe8c5f78149d695e743815d7e2083cd8638b", "5c62a4866b9e824b2ba227a2e33755b08c362571b34bd2aa43ad547090d5cb83"),
    "Superior": ("90b18494a90ac5dc96b83383cf7375a13a49c07cebba7942754e843f2c1727a8", "b93878be119b656ff1bd5e38f9b4a6c7b295b72250a6af24fc5e5f32ea3a6805"),
    "Swormville": ("548d00d5848b6a83c862e428cd98755bff6a0822ae58f714d0ec51df382a267a", "b0a316db7491595ae312db9d4f091ce947d743029b5ab95732e4215f11987a25"),
    "Woonsocket": ("a6d2dcf4742a2d66a3c95cb5dfc24eaf99e3808dccd4d46b9f10ee6d2d288a2c", "c25651e2298e85968fc10ea45b864abca23f75aea0df67bb4e256be0619b89ff"),
}
FILES = ("map.png", "poses.txt")


def read_scene(src: Path, scene: str) -> dict:
    """The scene's two files as bytes, from its zip or its extracted folder."""
    z = src / f"{scene}.zip"
    if z.is_file():
        with zipfile.ZipFile(z) as zf:
            return {f: zf.read(f"{scene}/{f}") for f in FILES}
    for d in (src / scene, src / "gibson_t" / scene, src / DATASET / scene):
        if all((d / f).is_file() for f in FILES):
            return {f: (d / f).read_bytes() for f in FILES}
    raise FileNotFoundError(f"no {scene}.zip or {scene}/ with {' and '.join(FILES)} under {src}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="folder with the per-scene zips or the extracted scenes of F3Loc's gibson_t")
    ap.add_argument("--fixtures", default=str(HERE.parent / "fixtures"))
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.fixtures) / "gibson" / DATASET
    bad = []
    for scene, sums in SHA256.items():
        try:
            data = read_scene(src, scene)
        except (OSError, KeyError, zipfile.BadZipFile) as e:
            bad.append(f"{scene}: {e}")
            continue
        wrong = [f for f, want in zip(FILES, sums) if hashlib.sha256(data[f]).hexdigest() != want]
        if wrong:
            bad.append(f"{scene}: SHA-256 of {', '.join(wrong)} differs from the paper's copy")
            continue
        (dst / scene).mkdir(parents=True, exist_ok=True)
        for f in FILES:
            (dst / scene / f).write_bytes(data[f])
    if bad:
        sys.exit("\n".join(["not installed:"] + bad))
    print(f"installed {' and '.join(FILES)} of {len(SHA256)} scenes into {dst}")


if __name__ == "__main__":
    main()
