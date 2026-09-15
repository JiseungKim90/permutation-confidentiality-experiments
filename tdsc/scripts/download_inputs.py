#!/usr/bin/env python3
"""Download public experiment inputs and verify their complete SHA-256 digests."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "model": {
        "filename": "cifar10_resnet20.pt",
        "url": (
            "https://github.com/chenyaofo/pytorch-cifar-models/releases/"
            "download/resnet/cifar10_resnet20-4118986f.pt"
        ),
        "sha256": "4118986f0df73003d572b0e397f0ac7b3f60af1f31aff3d2da164536e36f6ec8",
    },
    "cifar": {
        "filename": "cifar-10-python.tar.gz",
        "url": "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        "sha256": "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce",
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download(name, spec, data_dir, force=False):
    target = data_dir / spec["filename"]
    if target.exists():
        actual = sha256_file(target)
        if actual == spec["sha256"]:
            return {"name": name, "path": str(target), "status": "present", "sha256": actual}
        if not force:
            raise RuntimeError(
                "%s exists with SHA-256 %s; expected %s (use --force to replace)"
                % (target, actual, spec["sha256"])
            )

    part = target.with_name(target.name + ".part")
    if part.exists():
        part.unlink()
    request = Request(spec["url"], headers={"User-Agent": "p050-tdsc-reproduction/1"})
    try:
        with urlopen(request, timeout=60) as response, part.open("wb") as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                output.write(block)
        actual = sha256_file(part)
        if actual != spec["sha256"]:
            raise RuntimeError(
                "downloaded %s with SHA-256 %s; expected %s"
                % (name, actual, spec["sha256"])
            )
        os.replace(str(part), str(target))
    finally:
        if part.exists():
            part.unlink()
    return {"name": name, "path": str(target), "status": "downloaded", "sha256": actual}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--only", choices=("all", "model", "cifar"), default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    names = tuple(INPUTS) if args.only == "all" else (args.only,)
    records = [download(name, INPUTS[name], args.data_dir, args.force) for name in names]
    print(json.dumps({"success": True, "inputs": records}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
