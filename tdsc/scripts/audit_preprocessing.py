"""Check an ordinary stem-convolution rewrite; no extraction is performed."""

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import conv2d


root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from lib.cifar10 import MEAN, STD
from lib.resnet20 import (
    CIFAR10_RESNET20_SHA256,
    fold_input_normalisation,
    load_resnet20_cifar10,
    resnet20_float_params,
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


torch.set_num_threads(1)
checkpoint = root / "data/cifar10_resnet20.pt"
model, metadata = load_resnet20_cifar10(
    str(checkpoint), CIFAR10_RESNET20_SHA256)
fp = resnet20_float_params(model)
mean = np.asarray(MEAN, dtype=np.float64)
std = np.asarray(STD, dtype=np.float64)
original_w = torch.from_numpy(fp["stem"]["W"].copy().reshape(-1, 3, 3, 3))
original_b = torch.from_numpy(fp["stem"]["b"].copy())
folded = fold_input_normalisation(fp, mean, std)["stem"]
folded_w = torch.from_numpy(folded["W"].reshape(-1, 3, 3, 3))
folded_b = torch.from_numpy(folded["b"])
mean_t = torch.from_numpy(mean).reshape(1, 3, 1, 1)
std_t = torch.from_numpy(std).reshape(1, 3, 1, 1)
inputs = {
    "zero": torch.zeros((1, 3, 32, 32), dtype=torch.float64),
    "half": torch.full((1, 3, 32, 32), 0.5, dtype=torch.float64),
    "ramp": torch.arange(3 * 32 * 32, dtype=torch.float64).reshape(1, 3, 32, 32) / (3 * 32 * 32 - 1),
}
boundary = torch.ones((32, 32), dtype=torch.bool)
boundary[1:-1, 1:-1] = False
checks = []
for name, x in inputs.items():
    conventional = conv2d((x - mean_t) / std_t, original_w, original_b, padding=1)
    current_fold = conv2d(x, folded_w, folded_b, padding=1)
    error = (conventional - current_fold).abs()
    checks.append({
        "input": name,
        "interior_max_absolute_difference": float(error[:, :, 1:-1, 1:-1].max()),
        "boundary_max_absolute_difference": float(error[:, :, boundary].max()),
        "boundary_values_different_at_1e-10": int((error[:, :, boundary] > 1e-10).sum()),
        "boundary_values_checked": int(error[:, :, boundary].numel()),
        "output_channels_with_boundary_difference": int((error[:, :, boundary].max(dim=-1).values > 1e-10).sum()),
    })
confirmed = all(
    r["interior_max_absolute_difference"] < 1e-10
    and r["boundary_max_absolute_difference"] > 1e-10 for r in checks)
report = {
    "schema": "p050-ordinary-forward-preprocessing-audit-v1",
    "utc": datetime.now(timezone.utc).isoformat(),
    "python": platform.python_version(),
    "numpy": np.__version__,
    "torch": torch.__version__,
    "checkpoint_sha256": sha256(checkpoint),
    "script_sha256": sha256(Path(__file__)),
    "normalization_source_sha256": sha256(root / "lib/resnet20.py"),
    "seed_schedule": "None; three deterministic synthetic ordinary inputs.",
    "normalise_mean": mean.tolist(),
    "normalise_std": std.tolist(),
    "checks": checks,
    "boundary_mismatch_confirmed": confirmed,
    "scope": "Float64 stem outputs before activation and quantization; this does not measure prediction changes.",
    "new_extraction_experiments": 0,
}
print(json.dumps(report, indent=2, sort_keys=True))
raise SystemExit(0 if confirmed else 1)
