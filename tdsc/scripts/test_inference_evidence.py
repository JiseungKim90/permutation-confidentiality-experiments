"""Negative controls for the ordinary-inference evidence acceptance gate."""

import copy
import json
import tempfile
from pathlib import Path

import numpy as np

from verify_inference_evidence import ROOT, check_raw_arrays, check_report, digest


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
