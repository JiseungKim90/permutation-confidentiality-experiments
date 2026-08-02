"""Verify the journal-extension SHA-256 manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def verify_manifest(manifest: Path, base: Path) -> int:
    base = base.resolve()
    checked = 0
    for line_number, raw in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise ValueError(f"{manifest}:{line_number}: malformed checksum line")
        expected, relative = fields
        target = (base / relative).resolve()
        if base != target and base not in target.parents:
            raise ValueError(f"{manifest}:{line_number}: path escapes manifest root")
        if not target.is_file():
            raise FileNotFoundError(target)
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        actual = digest.hexdigest()
        if actual != expected.lower():
            raise RuntimeError(f"checksum mismatch: {relative}")
        checked += 1
    return checked


def main() -> None:
    results = {
        "journal_files": verify_manifest(
            ROOT / "outputs" / "JOURNAL_SHA256SUMS.txt", ROOT
        ),
        "data_side_files": verify_manifest(
            ROOT / "outputs" / "data_side" / "SHA256SUMS.txt",
            ROOT / "outputs" / "data_side",
        ),
    }
    print(json.dumps({"status": "pass", **results}, indent=2))


if __name__ == "__main__":
    main()