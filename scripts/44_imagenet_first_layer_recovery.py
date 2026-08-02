"""Row-aligned recovery on first Conv+BN layers of pretrained ImageNet CNNs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import models

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.unlabeled_recovery import (
    AmbiguousRecoveryError,
    FreshPermutationOracle,
    InconsistentTranscriptError,
    recover_bias_anchored,
    recover_bias_free,
    row_canonicalize,
)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--precisions", type=int, nargs="+", default=[256, 4096, 65536])
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs" / "imagenet_first_layer_row_recovery.json"
    )
    return parser.parse_args()


def architectures():
    return {
        "resnet50_v1": (
            lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
            lambda model: (model.conv1, model.bn1),
        ),
        "densenet121_v1": (
            lambda: models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1),
            lambda model: (model.features.conv0, model.features.norm0),
        ),
        "mobilenet_v2_v1": (
            lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1),
            lambda model: (model.features[0][0], model.features[0][1]),
        ),
        "efficientnet_b0_v1": (
            lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
            lambda model: (model.features[0][0], model.features[0][1]),
        ),
    }


def fold_conv_bn(conv, bn, precision):
    weight = conv.weight.detach().cpu().numpy().reshape(conv.out_channels, -1)
    if conv.bias is None:
        bias = np.zeros(conv.out_channels, dtype=np.float64)
    else:
        bias = conv.bias.detach().cpu().numpy().astype(np.float64)
    scale = (
        bn.weight.detach().cpu().numpy()
        / np.sqrt(bn.running_var.detach().cpu().numpy() + bn.eps)
    )
    folded_weight = weight * scale[:, None]
    folded_bias = (
        bn.bias.detach().cpu().numpy()
        + scale * (bias - bn.running_mean.detach().cpu().numpy())
    )
    return (
        np.rint(folded_weight * precision).astype(np.int64),
        np.rint(folded_bias * precision).astype(np.int64),
    )


def verify(recovered, true_weight, anchor):
    return bool(np.array_equal(recovered.weight, row_canonicalize(true_weight, anchor)))


def attempt_biased(weight, bias, seed):
    if np.unique(bias).size != bias.size:
        return {"status": "not_applicable", "reason": "quantized bias collision"}
    oracle = FreshPermutationOracle(weight, bias, seed=seed, noise_bound=0.499)
    try:
        recovered = recover_bias_anchored(
            oracle.query, output_dim=weight.shape[0], input_dim=weight.shape[1]
        )
        return {
            "status": "exact" if verify(recovered, weight, bias) else "wrong",
            "queries": oracle.query_count,
        }
    except (AmbiguousRecoveryError, InconsistentTranscriptError) as exc:
        return {"status": type(exc).__name__, "reason": str(exc),
                "queries": oracle.query_count}


def attempt_bias_free(weight, seed):
    m, d = weight.shape
    anchor = next((j for j in range(d) if np.unique(weight[:, j]).size == m), None)
    if anchor is None:
        return {"status": "not_applicable",
                "reason": "no pairwise-distinct quantized weight column"}
    oracle = FreshPermutationOracle(weight, seed=seed, noise_bound=0.499)
    try:
        recovered = recover_bias_free(
            oracle.query, output_dim=m, input_dim=d, anchor_coordinate=anchor
        )
        return {
            "status": "exact" if verify(recovered, weight, weight[:, anchor]) else "wrong",
            "queries": oracle.query_count,
            "anchor_coordinate": anchor,
        }
    except (AmbiguousRecoveryError, InconsistentTranscriptError) as exc:
        return {"status": type(exc).__name__, "reason": str(exc),
                "queries": oracle.query_count, "anchor_coordinate": anchor}


def main():
    args = parse_args()
    started = time.time()
    rows = []
    for model_index, (name, (builder, first_layer)) in enumerate(architectures().items()):
        model = builder().eval()
        conv, bn = first_layer(model)
        for precision_index, precision in enumerate(args.precisions):
            weight, bias = fold_conv_bn(conv, bn, precision)
            seed = args.seed + model_index * 1000 + precision_index * 100
            row = {
                "model": name,
                "precision": precision,
                "shape": [int(v) for v in weight.shape],
                "distinct_biases": int(np.unique(bias).size),
                "bias_anchored": attempt_biased(weight, bias, seed),
                "bias_free": attempt_bias_free(weight, seed + 1),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
        del model

    result = {
        "experiment": "pretrained ImageNet first-layer row recovery",
        "coefficients": [1, 2, 3],
        "noise_bound_lattice_units": 0.499,
        "seed": args.seed,
        "rows": rows,
        "summary": {
            "settings": len(rows),
            "bias_anchored_exact": sum(
                row["bias_anchored"]["status"] == "exact" for row in rows
            ),
            "bias_free_exact": sum(
                row["bias_free"]["status"] == "exact" for row in rows
            ),
            "wrong": sum(
                row[mode]["status"] == "wrong"
                for row in rows for mode in ("bias_anchored", "bias_free")
            ),
        },
        "wall_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result["summary"], indent=2))
    print(f"[done] wrote {args.output} in {result['wall_seconds']:.2f}s")


if __name__ == "__main__":
    main()
