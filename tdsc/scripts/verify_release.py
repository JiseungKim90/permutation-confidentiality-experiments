#!/usr/bin/env python3
"""Fail-closed audit for the public TDSC release tree and reference evidence."""

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASHES = {
    "reference/extraction/recovered.npz": "72a3a5d7f7fe3344f283d29fe65f9d12dabd87accb400b3c03f936d21854b7dd",
    "reference/extraction/recovered.json": "d49345c2820771ede65ccce55a8e1935d3bb8fbad996e522903880bf7034cbf6",
    "reference/completion/completed.npz": "f4fc25cb10a1bbe6d4987c7c008bb871faf220b669130f6017e2edbce15bfe4e",
    "reference/completion/completed.json": "d49345c2820771ede65ccce55a8e1935d3bb8fbad996e522903880bf7034cbf6",
}
REQUIRED = (
    "README.md",
    "SECURITY.md",
    "LICENSE",
    "MANIFEST.sha256",
    "CITATION.cff",
    "THIRD_PARTY_NOTICES.md",
    "requirements.txt",
    "requirements-recorded.txt",
    "scripts/run_extraction.py",
    "scripts/run_completion.py",
    "scripts/identifiability_exhaustive.py",
    "scripts/download_inputs.py",
    "scripts/verify_runs.py",
    "scripts/verify_finite_reference.py",
    "scripts/test_lta_cover_soundness.py",
    "scripts/test_attacker_boundary.py",
    "scripts/test_graph_gauge_bias_scope.py",
    "scripts/test_trusted_inputs.py",
    "lib/checkpoint.py",
    "lib/cifar10.py",
    "lib/fmap_partial.py",
    "lib/lta_run.py",
    "lib/trusted_io.py",
    "reference/extraction/attacker.json",
    "reference/extraction/verification.json",
    "reference/completion/verification.json",
    "reference/finite_reference.json",
)
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".cff", ".yml", ".yaml"}
FORBIDDEN_TEXT = {
    "Windows user path": re.compile(r"[A-Za-z]:[\\/]Users[\\/]", re.I),
    "Linux home path": re.compile("/" + "home/" + r"[^/\s]+/"),
    "private key": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{20,}"),
    "unsafe torch load": re.compile(r"weights_only\s*=\s*False"),
}
FORBIDDEN_HOST_LABELS = ("ubuntu" + "02", "lab" + "614")
FORBIDDEN_SUFFIXES = {".zip", ".pt", ".pth", ".pem", ".key"}
RUNTIME_PREFIXES = ("logs/", "results/", ".venv/")
TORCH_LOAD_CALL = "torch" + ".load("


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def read_json(relative):
    with (ROOT / relative).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_runtime_file(relative):
    return bool(
        relative.startswith(RUNTIME_PREFIXES)
        or (relative.startswith("data/") and relative != "data/README.md")
        or "/__pycache__/" in "/" + relative
    )


def main():
    failures = []
    publish_files = {}
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            failures.append("missing required file: %s" % relative)

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if is_runtime_file(relative):
            continue
        if relative != "MANIFEST.sha256":
            publish_files[relative] = path
        if path.suffix.lower() in FORBIDDEN_SUFFIXES and not relative.startswith("reference/"):
            failures.append("unintended binary or credential-like file: %s" % relative)
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8")
            for label, pattern in FORBIDDEN_TEXT.items():
                if pattern.search(text):
                    failures.append("%s in %s" % (label, relative))
            for host_label in FORBIDDEN_HOST_LABELS:
                if host_label in text:
                    failures.append("machine-specific host label in %s" % relative)
            if (
                path.suffix.lower() == ".py"
                and TORCH_LOAD_CALL in text
                and relative != "lib/checkpoint.py"
            ):
                failures.append("direct PyTorch checkpoint load outside trusted loader: %s" % relative)

    manifest_path = ROOT / "MANIFEST.sha256"
    manifest = {}
    if manifest_path.is_file():
        for line_number, line in enumerate(
                manifest_path.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split(None, 1)
            if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
                failures.append("malformed manifest line %d" % line_number)
                continue
            relative = parts[1].lstrip("*")
            if relative in manifest:
                failures.append("duplicate manifest entry: %s" % relative)
            manifest[relative] = parts[0]
        missing = sorted(set(publish_files) - set(manifest))
        extra = sorted(set(manifest) - set(publish_files))
        if missing:
            failures.append("files missing from manifest: %r" % missing)
        if extra:
            failures.append("manifest entries without files: %r" % extra)
        for relative in sorted(set(manifest).intersection(publish_files)):
            actual = sha256_file(publish_files[relative])
            if actual != manifest[relative]:
                failures.append("manifest SHA-256 mismatch for %s" % relative)

    for relative, expected in EXPECTED_HASHES.items():
        path = ROOT / relative
        if path.is_file():
            actual = sha256_file(path)
            if actual != expected:
                failures.append("SHA-256 mismatch for %s: %s" % (relative, actual))

    extraction = read_json("reference/extraction/verification.json")
    completion = read_json("reference/completion/verification.json")
    if not extraction.get("success"):
        failures.append("reference extraction is not successful")
    if extraction.get("extraction", {}).get("records_certified") != 29:
        failures.append("reference extraction does not certify 29 records")
    if extraction.get("evaluation", {}).get("full_identical_logit_vectors") != 10000:
        failures.append("reference extraction does not match all 10,000 logits")
    if not completion.get("success"):
        failures.append("reference completion is not successful")
    if completion.get("evaluation", {}).get("observable_maps_checked") != 29:
        failures.append("reference completion does not check 29 observable maps")
    if completion.get("evaluation", {}).get("full_identical_logit_vectors") != 10000:
        failures.append("reference completion does not match all 10,000 logits")

    report = {
        "success": not failures,
        "failures": failures,
        "reference_hashes_checked": len(EXPECTED_HASHES),
        "manifest_files_checked": len(manifest),
        "required_files_checked": len(REQUIRED),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
