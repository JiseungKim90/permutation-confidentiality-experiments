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
    "scripts/run_multicheckpoint.py",
    "scripts/promote_multicheckpoint_audit.py",
    "scripts/identifiability_exhaustive.py",
    "scripts/download_inputs.py",
    "scripts/verify_runs.py",
    "scripts/verify_finite_reference.py",
    "scripts/test_lta_cover_soundness.py",
    "scripts/test_fmap_assignment.py",
    "scripts/test_fmap_batch.py",
    "scripts/test_attacker_boundary.py",
    "scripts/test_graph_gauge_bias_scope.py",
    "scripts/test_trusted_inputs.py",
    "scripts/test_input_normalization.py",
    "scripts/test_finite_affine_laws.py",
    "scripts/test_affine_graph_semantics.py",
    "scripts/test_inference_evidence.py",
    "scripts/verify_inference_evidence.py",
    "scripts/verify_normal_inference.py",
    "scripts/run_validation.py",
    "scripts/reproduce_public_inference.py",
    "lib/input_normalization.py",
    "lib/checkpoint.py",
    "lib/cifar10.py",
    "lib/fmap_partial.py",
    "lib/lta_run.py",
    "lib/trusted_io.py",
    "reference/extraction/attacker.json",
    "reference/extraction/verification.json",
    "reference/completion/verification.json",
    "reference/finite_reference.json",
    "reference/multicheckpoint/audit.json",
    "reference/multicheckpoint/checkpoints.json",
    "reference/multicheckpoint/development_failures.json",
    "reference/public-reproduction-20261002/README.md",
    "reference/public-reproduction-20261002/workflow.json",
    "reference/public-reproduction-20261002/normal-inference.json",
    "reference/public-reproduction-20261002/validation.json",
    "reference/public-reproduction-20261002/requirements-frozen.txt",
    "reference/public-reproduction-20261002/inference.stdout.log",
    "training/legacy_qat/README.md",
    "training/legacy_qat/requirements-recorded.txt",
    "training/legacy_qat/lib_qat/checkpoint.py",
    "training/legacy_qat/scripts/qat_common.py",
    "training/legacy_qat/scripts/qat_data.py",
    "training/legacy_qat/scripts/qat_train.py",
    "training/legacy_qat/results/qat_train_qatf_r20_w5_s0.json",
    "training/legacy_qat/results/qat_train_qatf_r20_w5_s1.json",
    "training/legacy_qat/results/qat_train_qatf_r20_w6_s0.json",
    "training/legacy_qat/results/qat_train_qatf_r20_w6_s1.json",
    "training/legacy_qat/results/qat_train_qatf_r20_w7_s0.json",
    "training/legacy_qat/results/qat_train_qatf_r20_w7_s1.json",
    "training/legacy_qat/results/qat_train_qatf_r20_w8_s0.json",
    "training/legacy_qat/results/qat_train_qatf_r20_w8_s1.json",
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
TRUSTED_CHECKPOINT_LOADERS = {
    "lib/checkpoint.py",
    "training/legacy_qat/lib_qat/checkpoint.py",
}
EXPECTED_QAT_NAMES = {
    "qat-w%d-s%d" % (width, seed)
    for width in range(5, 9) for seed in range(2)
}
EXPECTED_QAT_PATHS = {
    "data/checkpoints/qat_qatf_r20_w%d_s%d.pt" % (width, seed)
    for width in range(5, 9) for seed in range(2)
}
OFFICIAL_CHECKPOINT = {
    "name": "official",
    "path": "data/cifar10_resnet20.pt",
    "sha256": "4118986f0df73003d572b0e397f0ac7b3f60af1f31aff3d2da164536e36f6ec8",
}
FINAL_AUDIT_COMMIT = "d3910df81a16c2a69bef299f24c30fcd0fe341fd"
EMPTY_GIT_DIFF_SHA256 = (
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
)


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


def is_sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def validate_multicheckpoint_audit(audit, checkpoint_items):
    failures = []
    expected = {
        item.get("name"): item
        for item in checkpoint_items
        if isinstance(item, dict)
    }
    if audit.get("schema") != "p050-public-multicheckpoint-audit-v2":
        failures.append("multi-checkpoint audit has the wrong schema")
    if audit.get("base_commit") != FINAL_AUDIT_COMMIT:
        failures.append("multi-checkpoint audit does not use the final code commit")

    summary = audit.get("result_summary", {})
    if (
        summary.get("attempted_checkpoints") != 9
        or summary.get("certified_checkpoints") != 9
        or summary.get("all_checkpoints_certified") is not True
        or summary.get("per_checkpoint_manual_tuning") is not False
    ):
        failures.append("multi-checkpoint audit summary is not a 9/9 fixed-configuration certificate")

    records = audit.get("records", [])
    if not isinstance(records, list) or len(records) != 9:
        failures.append("multi-checkpoint audit must contain exactly nine records")
        records = []
    names = [record.get("name") for record in records
             if isinstance(record, dict)]
    if len(names) != len(set(names)) or set(names) != set(expected):
        failures.append("multi-checkpoint audit record names do not match the input manifest")

    for record in records:
        if not isinstance(record, dict):
            failures.append("multi-checkpoint audit contains a non-object record")
            continue
        name = record.get("name")
        manifest_item = expected.get(name, {})
        if record.get("checkpoint_sha256") != manifest_item.get("sha256"):
            failures.append("multi-checkpoint digest mismatch for %s" % name)
        if (
            record.get("configuration_changed_for_checkpoint") is not False
            or record.get("full_certificate") is not True
            or record.get("outcome") != "certified"
        ):
            failures.append("multi-checkpoint record is not certified: %s" % name)

        extraction = record.get("extraction", {})
        if (
            extraction.get("success") is not True
            or extraction.get("returncode") != 0
            or extraction.get("client_certified_records") != 29
            or extraction.get("lta_invocations_certified") != 34
            or not isinstance(extraction.get("lta_passes_total"), int)
            or extraction.get("lta_passes_total") < 34
            or not isinstance(extraction.get("sessions"), int)
            or extraction.get("sessions") <= 0
            or not isinstance(extraction.get("affine_evaluations"), int)
            or extraction.get("affine_evaluations") <= 0
            or extraction.get("identical_logit_vectors") != 10000
            or extraction.get("post_calibration_identical_logit_vectors") != 9936
            or extraction.get("max_abs_logit_difference") != 0
            or extraction.get("parallel_batched_exact") is not True
        ):
            failures.append("incomplete extraction certificate for %s" % name)

        completion = record.get("completion", {})
        if (
            completion.get("success") is not True
            or completion.get("returncode") != 0
            or completion.get("observable_maps_checked") != 29
            or completion.get("observable_maps_successful") != 29
            or not isinstance(completion.get("sessions"), int)
            or completion.get("sessions") <= 0
            or not isinstance(completion.get("affine_evaluations"), int)
            or completion.get("affine_evaluations") <= 0
            or completion.get("identical_logit_vectors") != 10000
            or completion.get("post_calibration_identical_logit_vectors") != 9936
            or completion.get("max_abs_logit_difference") != 0
            or completion.get("parallel_batched_exact") is not True
        ):
            failures.append("incomplete completion certificate for %s" % name)

    provenance = audit.get("run_provenance", {})
    if provenance.get("git_commit") != FINAL_AUDIT_COMMIT:
        failures.append("multi-checkpoint provenance commit is inconsistent")
    if provenance.get("git_diff_sha256") != EMPTY_GIT_DIFF_SHA256:
        failures.append("multi-checkpoint audit was not launched from a clean tree")
    for field in ("source_set_sha256", "launch_sha256", "result_sha256"):
        if not is_sha256(provenance.get(field)):
            failures.append("missing or malformed multi-checkpoint %s" % field)
    environment = provenance.get("environment", {})
    if (
        environment.get("python") != "3.10.20"
        or environment.get("numpy") != "1.24.4"
        or environment.get("torch") != "2.10.0+cpu"
    ):
        failures.append("multi-checkpoint dependency environment is inconsistent")

    fixed = audit.get("fixed_configuration", {})
    expected_fixed = {
        "attack_seed": 20260923,
        "extraction_oracle_seed": 20260924,
        "completion_oracle_seed": 20260925,
        "weight_bits": 8,
        "t_steps": 3,
        "evaluation_images": 10000,
        "budget_sec": 1800,
        "evaluation_timeout_sec": 3600,
        "search_limit": 256,
        "minimum_marker_cover": 2,
        "shortcut_carrier_repeats": 2,
        "trace_arithmetic": False,
        "per_checkpoint_manual_tuning": False,
    }
    for field, value in expected_fixed.items():
        if fixed.get(field) != value:
            failures.append("unexpected fixed audit setting %s" % field)
    if fixed.get("control_values") != [1, 2, 4, 8, 16, 32, 64, 128, 255]:
        failures.append("unexpected fixed audit control values")
    return failures


def main():
    failures = []
    publish_files = {}
    checkpoint_manifest = read_json(
        "reference/multicheckpoint/checkpoints.json")
    allowed_data_files = {
        "data/README.md",
        OFFICIAL_CHECKPOINT["path"],
        "data/cifar-10-python.tar.gz",
    }.union(EXPECTED_QAT_PATHS)
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            failures.append("missing required file: %s" % relative)

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("data/") and relative not in allowed_data_files:
            failures.append("unexpected file in guarded data directory: %s" %
                            relative)
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
                and relative not in TRUSTED_CHECKPOINT_LOADERS
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
    multicheckpoint_audit = read_json("reference/multicheckpoint/audit.json")
    development_failures = read_json(
        "reference/multicheckpoint/development_failures.json")
    checkpoint_items = checkpoint_manifest.get("checkpoints", [])
    if not isinstance(checkpoint_items, list):
        failures.append("checkpoint manifest does not contain a list")
        checkpoint_items = []
    names = [item.get("name") for item in checkpoint_items
             if isinstance(item, dict)]
    paths = [item.get("path") for item in checkpoint_items
             if isinstance(item, dict)]
    if len(names) != len(set(names)):
        failures.append("checkpoint manifest contains duplicate names")
    if len(paths) != len(set(paths)):
        failures.append("checkpoint manifest contains duplicate paths")

    official = [item for item in checkpoint_items
                if isinstance(item, dict) and item.get("name") == "official"]
    if len(official) != 1:
        failures.append("checkpoint manifest must contain one official input")
    elif (
        official[0].get("distributed") is not False
        or official[0].get("path") != OFFICIAL_CHECKPOINT["path"]
        or official[0].get("sha256") != OFFICIAL_CHECKPOINT["sha256"]
    ):
        failures.append("official checkpoint manifest record is inconsistent")

    distributed_checkpoints = 0
    distributed_names = set()
    distributed_paths = set()
    checkpoint_root = (ROOT / "data" / "checkpoints").resolve()
    for item in checkpoint_items:
        if not isinstance(item, dict):
            failures.append("checkpoint manifest contains a non-object entry")
            continue
        distributed = item.get("distributed")
        if not isinstance(distributed, bool):
            failures.append("checkpoint distribution flag is missing: %s" %
                            item.get("name"))
            continue
        if not distributed:
            continue
        distributed_checkpoints += 1
        name = item.get("name")
        relative = item.get("path")
        if name not in EXPECTED_QAT_NAMES:
            failures.append("unexpected distributed checkpoint name: %s" % name)
        else:
            distributed_names.add(name)
        if relative not in EXPECTED_QAT_PATHS:
            failures.append("unexpected distributed checkpoint path: %s" %
                            relative)
            continue
        distributed_paths.add(relative)
        checkpoint_path = (ROOT / relative).resolve()
        if checkpoint_root not in checkpoint_path.parents:
            failures.append("distributed checkpoint escapes guarded directory: %s" %
                            relative)
            continue
        if not checkpoint_path.is_file():
            failures.append("distributed checkpoint is missing: %s" %
                            name)
        elif sha256_file(checkpoint_path) != item.get("sha256"):
            failures.append("distributed checkpoint digest mismatch: %s" %
                            name)
    if (distributed_checkpoints != 8
            or distributed_names != EXPECTED_QAT_NAMES
            or distributed_paths != EXPECTED_QAT_PATHS):
        failures.append("expected eight distributed QAT checkpoints, found %d" %
                        distributed_checkpoints)
    if checkpoint_manifest.get(
            "all_qat_checkpoints_distributed_in_repository") is not True:
        failures.append("QAT checkpoint distribution is not asserted")
    failures.extend(validate_multicheckpoint_audit(
        multicheckpoint_audit, checkpoint_items))
    if (
        development_failures.get("schema")
            != "p050-multicheckpoint-development-failures-v2"
        or development_failures.get("resolution", {}).get(
            "final_code_commit") != FINAL_AUDIT_COMMIT
        or development_failures.get("resolution_smoke", {}).get(
            "outcome") != "certified"
    ):
        failures.append("development failure record is incomplete or stale")
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
        "distributed_checkpoints_checked": distributed_checkpoints,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
