#!/usr/bin/env python3
"""Extract Aneumo STL meshes from zip shards into HemoData/Vascular_STL/aneumo.

This is a lightweight companion to full Aneumo unpacking. It lets the STL
geometry corpus become available before the full CFD npy/VTK contents finish
expanding from the large zip shards.
"""

import argparse
import json
import os
import re
import shutil
import zipfile
from pathlib import Path


ROOT = Path(".")
DEFAULT_ZIP_ROOT = ROOT / "totaldata" / "unpacked" / "dataset_2" / "aneumo"
DEFAULT_OUT = ROOT / "HemoData" / "Vascular_STL" / "aneumo"
DEFAULT_MANIFEST = ROOT / "results" / "full_pretrain" / "aneumo_1000case_prepare" / "aneumo_stl_zip_extract_manifest.jsonl"
STL_RE = re.compile(r"^(?P<case>[0-9]+)/Stl/(?P<name>[^/]+\.stl)$", re.IGNORECASE)


def safe_name(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._=-" else "_" for c in text)


def natural_zip_key(path: Path):
    stem = path.stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_member(zf: zipfile.ZipFile, member: str, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return "skip_existing"
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with zf.open(member, "r") as src, tmp.open("wb") as out:
        shutil.copyfileobj(src, out, length=1024 * 1024)
    os.replace(tmp, dst)
    return "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_root", type=Path, default=DEFAULT_ZIP_ROOT)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reset_manifest", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.reset_manifest and args.manifest.exists():
        args.manifest.unlink()

    zips = sorted(args.zip_root.glob("*.zip"), key=natural_zip_key)
    total_ok = total_skip = total_fail = total_members = 0
    print(f"zip_root={args.zip_root}")
    print(f"out_dir={args.out_dir}")
    print(f"manifest={args.manifest}")
    print(f"zip_shards={len(zips)}")

    for idx, zip_path in enumerate(zips, 1):
        ok = skip = fail = members = 0
        print(f"[{idx}/{len(zips)}] {zip_path.name}", flush=True)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                names = sorted(n for n in zf.namelist() if STL_RE.match(n))
                for member in names:
                    members += 1
                    match = STL_RE.match(member)
                    case = match.group("case")
                    name = safe_name(Path(match.group("name")).stem)
                    dst = args.out_dir / f"aneumo_{case}_{name}.stl"
                    try:
                        status = extract_member(zf, member, dst)
                        if status == "ok":
                            ok += 1
                        else:
                            skip += 1
                    except Exception as exc:
                        status = "failed"
                        fail += 1
                        write_jsonl(args.manifest, {
                            "zip": str(zip_path),
                            "member": member,
                            "case": case,
                            "dst": str(dst),
                            "status": status,
                            "error": repr(exc),
                        })
                        continue
                    write_jsonl(args.manifest, {
                        "zip": str(zip_path),
                        "member": member,
                        "case": case,
                        "dst": str(dst),
                        "status": status,
                        "size": dst.stat().st_size if dst.exists() else 0,
                    })
        except Exception as exc:
            fail += 1
            write_jsonl(args.manifest, {
                "zip": str(zip_path),
                "status": "zip_failed",
                "error": repr(exc),
            })
        total_ok += ok
        total_skip += skip
        total_fail += fail
        total_members += members
        print(f"  members={members} ok={ok} skip={skip} fail={fail}", flush=True)

    print("=" * 60)
    print(f"total_members={total_members} ok={total_ok} skip={total_skip} fail={total_fail}")
    print(f"out_stl={len(list(args.out_dir.glob('*.stl')))}")


if __name__ == "__main__":
    main()
