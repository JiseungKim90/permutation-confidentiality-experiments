"""Negative controls for the ordinary-inference evidence acceptance gate."""

import copy
import json
import tempfile
from pathlib import Path

import numpy as np

from verify_inference_evidence import (
    ROOT, SOURCE_FILES, check_raw_arrays, check_report, check_sources, digest,
)


def main():
    manifest = json.loads((ROOT / "reference/multicheckpoint/checkpoints.json").read_text())
    original = json.loads((ROOT / "reference/normal-inference-20261002/result.json").read_text())
    schedule = [item["name"] for item in manifest["checkpoints"]]
    check_report(original, manifest, schedule)
    mutations = [
        ("incomplete run", lambda r: r.update(status="running")),
        ("false success", lambda r: r.update(success=False)),
        ("promoted extraction claim", lambda r: r.update(new_extraction_experiments=1)),
        ("wrong dtype", lambda r: r["launch"].update(dtype="float32")),
        ("wrong dataset", lambda r: r.update(dataset_sha256="0" * 64)),
        ("missing checkpoint", lambda r: r["records"].pop()),
        ("duplicate checkpoint", lambda r: r["records"].append(copy.deepcopy(r["records"][0]))),
        ("short schedule", lambda r: r["launch"]["checkpoint_schedule"].pop()),
        ("short evaluation", lambda r: r["records"][0].update(n_test=9999)),
        ("noninteger planned count", lambda r: r["launch"].update(n_test=10000.0)),
        ("noninteger completed count", lambda r: r["records"][0].update(n_test=10000.0)),
        ("wrong checkpoint", lambda r: r["records"][0].update(checkpoint_sha256="0" * 64)),
        ("loader mismatch", lambda r: r["records"][0]["loader"].update(missing_keys=["conv1.weight"])),
        ("NaN error", lambda r: r["records"][0]["comparisons"]["repaired"].update(maximum_logit_absolute_difference=float("nan"))),
        ("negative error", lambda r: r["records"][0]["comparisons"]["repaired"].update(maximum_logit_absolute_difference=-1)),
        ("excess error", lambda r: r["records"][0]["comparisons"]["repaired"].update(maximum_logit_absolute_difference=1e-8)),
        ("missing vector", lambda r: r["records"][0]["comparisons"]["repaired"].update({"logit_vectors_within_1e-9": 9999})),
        ("prediction mismatch", lambda r: r["records"][0]["comparisons"]["repaired"].update(identical_predictions=9999)),
        ("inconsistent accuracy", lambda r: r["records"][0]["comparisons"]["repaired"].update(correct_predictions=-1)),
    ]
    rejected = []
    for name, mutate in mutations:
        record = copy.deepcopy(original)
        mutate(record)
        try:
            check_report(record, manifest, schedule)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError("invalid evidence accepted: " + name)
    for name, expected in (("empty expected schedule", []),
                           ("duplicate expected schedule", schedule + schedule[:1])):
        record = copy.deepcopy(original)
        if not expected:
            record["records"] = []
            record["launch"]["checkpoint_schedule"] = []
        try:
            check_report(record, manifest, expected)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError("invalid evidence accepted: " + name)
    launch = {"source_sha256": {name: digest(ROOT / name) for name in SOURCE_FILES}}
    assert check_sources(launch) == 9
    source_mutations = [
        ("empty source manifest", lambda s: s.clear()),
        ("incomplete source manifest", lambda s: s.pop("lib/models.py")),
        ("unexpected source dependency", lambda s: s.update({"README.md": digest(ROOT / "README.md")})),
        ("malformed source digest", lambda s: s.update({"lib/models.py": "invalid"})),
        ("changed source digest", lambda s: s.update({"lib/models.py": "0" * 64})),
    ]
    for name, mutate in source_mutations:
        candidate = copy.deepcopy(launch)
        mutate(candidate["source_sha256"])
        try:
            check_sources(candidate)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError("invalid evidence accepted: " + name)
    with tempfile.TemporaryDirectory(prefix="p050-inference-evidence-unit-") as directory:
        directory = Path(directory)
        logits = np.arange(30, dtype=np.float64).reshape(3, 10)
        labels = np.array([9, 0, 9], dtype=np.int64)
        path = directory / "fixture-ordinary-logits.npz"
        np.savez(path, labels=labels, reference=logits, repaired=logits,
                 uniform_bias_control=logits)
        comparison = {"maximum_logit_absolute_difference": 0.0,
                      "logit_vectors_within_1e-9": 3,
                      "identical_predictions": 3, "correct_predictions": 2}
        record = {"checkpoint": "fixture", "n_test": 3, "reference_correct": 2,
                  "raw_array_sha256": digest(path),
                  "comparisons": {"repaired": comparison,
                                  "uniform_bias_control": copy.deepcopy(comparison)}}
        check_raw_arrays([record], directory)
        record["comparisons"]["repaired"]["maximum_logit_absolute_difference"] = 1e-12
        try:
            check_raw_arrays([record], directory)
        except ValueError:
            rejected.append("summary disagrees with raw logits")
        else:
            raise AssertionError("incorrect raw-array summary accepted")
    print(json.dumps({"success": True, "negative_controls_rejected": rejected,
                      "negative_control_count": len(rejected)}, indent=2))


if __name__ == "__main__":
    main()
