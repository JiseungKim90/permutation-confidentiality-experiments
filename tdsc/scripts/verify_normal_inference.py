"""Validate ordinary inference after an input-normalization repair.

No oracle service, chosen intermediate messages, or extraction is executed.
QAT-wrapper inputs are evaluated as the loader's floating-point networks,
not as native QAT or as the legacy integer recovery experiment.
"""
import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--n-test", type=int, default=10000)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--threads", type=int, default=4)
parser.add_argument(
    "--checkpoints", nargs="+",
    help="Manifest names; omit to evaluate all nine current checkpoints.")
args = parser.parse_args()
root = Path(args.root).resolve()
out = Path(args.out).resolve()
if args.n_test <= 0 or args.n_test > 10000 or args.batch_size <= 0 or args.threads <= 0:
    raise ValueError("invalid evaluation count, batch size, or thread count")
out.mkdir(parents=True, exist_ok=True)
if (out / "result.json").exists() or (out / "launch.json").exists():
    raise FileExistsError("use a fresh output directory")
sys.path.insert(0, str(root))
from lib import cifar10
from lib.input_normalization import NormalizedInputConv2d
from lib.resnet20 import load_resnet20_cifar10


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def uniform_bias_comparator(model, mean, std):
    # This intentionally retains the defective boundary rule as a control.
    clone = copy.deepcopy(model)
    conv = clone.conv1
    mean_t = torch.tensor(mean, dtype=conv.weight.dtype).reshape(1, -1, 1, 1)
    std_t = torch.tensor(std, dtype=conv.weight.dtype).reshape(1, -1, 1, 1)
    weight = conv.weight.detach().clone()
    bias = torch.zeros(conv.out_channels, dtype=weight.dtype)
    if conv.bias is not None:
        bias = conv.bias.detach().clone()
    with torch.no_grad():
        conv.weight.copy_(weight / std_t)
    conv.bias = nn.Parameter(bias - (weight * mean_t / std_t).sum(dim=(1, 2, 3)), requires_grad=False)
    return clone.eval()


torch.set_num_threads(args.threads)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
manifest = root / "reference/multicheckpoint/checkpoints.json"
configuration = json.loads(manifest.read_text())
if args.checkpoints:
    selected = set(args.checkpoints)
    known = {item["name"] for item in configuration["checkpoints"]}
    if selected - known:
        raise ValueError("unknown checkpoints: " + ", ".join(sorted(selected - known)))
    configuration["checkpoints"] = [item for item in configuration["checkpoints"] if item["name"] in selected]
files = ["lib/input_normalization.py", "scripts/verify_normal_inference.py",
         "scripts/test_input_normalization.py",
         "lib/resnet20.py", "lib/models.py", "lib/cifar10.py",
         "lib/checkpoint.py", "lib/trusted_io.py",
         "reference/multicheckpoint/checkpoints.json"]
launch = {
    "schema": "p050-normal-inference-validation-v1",
    "utc": datetime.now(timezone.utc).isoformat(),
    "working_directory": str(root),
    "command": [sys.executable] + sys.argv,
    "source_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "source_sha256": {f: digest(root / f) for f in files},
    "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
    "dtype": "float64", "threads": args.threads, "batch_size": args.batch_size,
    "n_test": args.n_test, "seeds": "None; existing authenticated checkpoints and fixed test-set order.",
    "checkpoint_schedule": [c["name"] for c in configuration["checkpoints"]],
    "success_criteria": "Every input logit differs by at most 1e-9; every prediction matches; all planned checkpoints and samples are evaluated.",
    "failure_criteria": "Any criterion violation, missing input, loader mismatch, or incomplete run.",
    "scope": "Ordinary floating-point inference only; no new extraction, native-QAT, or encrypted-backend result.",
    "stdout_log": "stdout.log", "stderr_log": "stderr.log", "summary": "result.json",
}
write_json(out / "launch.json", launch)
report = {"schema": launch["schema"], "launch": launch, "status": "running",
          "records": [], "new_extraction_experiments": 0}
started = time.monotonic()
try:
    data_file = root / "data/cifar-10-python.tar.gz"
    data = cifar10.load(str(data_file))
    X, y = data["test_batch"]
    if len(X) != 10000 or len(y) != 10000:
        raise AssertionError("unexpected test dataset length")
    X, y = X[:args.n_test], y[:args.n_test]
    report["dataset_sha256"] = digest(data_file)
    for item in configuration["checkpoints"]:
        model, metadata = load_resnet20_cifar10(str(root / item["path"]), item["sha256"])
        if metadata["missing_keys"] or metadata["unexpected_keys"]:
            raise AssertionError("checkpoint loader did not match the architecture")
        model = model.double().eval()
        mean = metadata.get("normalise_mean") or cifar10.MEAN.tolist()
        std = metadata.get("normalise_std") or cifar10.STD.tolist()
        repaired = copy.deepcopy(model)
        repaired.conv1 = NormalizedInputConv2d(model.conv1, mean, std, (32, 32))
        repaired.eval()
        legacy = uniform_bias_comparator(model, mean, std)
        mean_t = torch.tensor(mean, dtype=torch.float64).reshape(1, -1, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float64).reshape(1, -1, 1, 1)
        buffers = {"reference": [], "repaired": [], "uniform_bias_control": []}
        with torch.inference_mode():
            for begin in range(0, len(X), args.batch_size):
                raw = torch.from_numpy(X[begin:begin + args.batch_size])
                buffers["reference"].append(model((raw - mean_t) / std_t).numpy())
                buffers["repaired"].append(repaired(raw).numpy())
                buffers["uniform_bias_control"].append(legacy(raw).numpy())
                if begin % (8 * args.batch_size) == 0:
                    print(json.dumps({"checkpoint": item["name"], "completed": min(begin + args.batch_size, len(X)), "planned": len(X)}), flush=True)
        arrays = {name: np.concatenate(values) for name, values in buffers.items()}
        if any(a.shape != (args.n_test, 10) or not np.isfinite(a).all() for a in arrays.values()):
            raise AssertionError("missing or nonfinite logits")
        reference = arrays["reference"]
        prediction = reference.argmax(axis=1)
        record = {"checkpoint": item["name"], "checkpoint_sha256": item["sha256"],
                  "loader": metadata, "n_test": args.n_test,
                  "reference_correct": int((prediction == y).sum()),
                  "comparisons": {}}
        for name in ("repaired", "uniform_bias_control"):
            value = arrays[name]
            difference = np.abs(value - reference)
            record["comparisons"][name] = {
                "maximum_logit_absolute_difference": float(difference.max()),
                "logit_vectors_within_1e-9": int((difference.max(axis=1) <= 1e-9).sum()),
                "identical_predictions": int((value.argmax(axis=1) == prediction).sum()),
                "correct_predictions": int((value.argmax(axis=1) == y).sum()),
            }
        fixed = record["comparisons"]["repaired"]
        record["success"] = fixed["logit_vectors_within_1e-9"] == args.n_test and fixed["identical_predictions"] == args.n_test
        array_path = out / (item["name"] + "-ordinary-logits.npz")
        np.savez_compressed(str(array_path), labels=y, **arrays)
        record["raw_array_sha256"] = digest(array_path)
        report["records"].append(record)
        write_json(out / "result.json", report)
        print(json.dumps({"checkpoint_complete": record}), flush=True)
    report["success"] = len(report["records"]) == len(configuration["checkpoints"]) and all(r["success"] for r in report["records"])
    report["status"] = "complete" if report["success"] else "failed"
except Exception as exc:
    report["success"] = False
    report["status"] = "failed"
    report["exception"] = repr(exc)
    raise
finally:
    report["elapsed_seconds"] = time.monotonic() - started
    report["ended_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(out / "result.json", report)
print(json.dumps({"status": report["status"], "success": report["success"], "checkpoints": len(report["records"])}), flush=True)
raise SystemExit(0 if report["success"] else 1)
