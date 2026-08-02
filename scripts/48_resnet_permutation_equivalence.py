"""Functional equivalence of hidden channel permutations on trained ResNet-20.

The script constructs a nontrivially permuted parameterization of every hidden
channel tensor while respecting residual additions.  Convolution input/output
channels, batch-normalization parameters, shortcut projections, and the final
classifier are transformed consistently.  The resulting state dict differs
coordinate-wise from the original but computes the same logits.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.models import ResNet20


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=root / "models" / "resnet20_seed0.pt"
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=root / "data"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs" / "resnet_permutation_equivalence.json"
    )
    return parser.parse_args()


def load_model(path):
    model = ResNet20()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    return model.eval(), checkpoint


def as_index(permutation, device):
    return torch.as_tensor(permutation, dtype=torch.long, device=device)


def permute_batchnorm(batchnorm, output_permutation):
    index = as_index(output_permutation, batchnorm.weight.device)
    with torch.no_grad():
        batchnorm.weight.copy_(batchnorm.weight.detach().clone()[index])
        batchnorm.bias.copy_(batchnorm.bias.detach().clone()[index])
        batchnorm.running_mean.copy_(
            batchnorm.running_mean.detach().clone()[index]
        )
        batchnorm.running_var.copy_(
            batchnorm.running_var.detach().clone()[index]
        )


def transform_conv_bn(conv, batchnorm, input_permutation, output_permutation):
    input_index = as_index(input_permutation, conv.weight.device)
    output_index = as_index(output_permutation, conv.weight.device)
    with torch.no_grad():
        weight = conv.weight.detach().clone()
        conv.weight.copy_(weight[output_index][:, input_index])
        if conv.bias is not None:
            conv.bias.copy_(conv.bias.detach().clone()[output_index])
    permute_batchnorm(batchnorm, output_permutation)


def permutation_digest(permutation):
    payload = np.asarray(permutation, dtype=np.int64).tobytes()
    return hashlib.sha256(payload).hexdigest()[:16]


def build_equivalent_model(original, seed):
    model = copy.deepcopy(original).eval()
    rng = np.random.default_rng(seed)
    records = []

    input_permutation = np.arange(3, dtype=np.int64)
    stem_output = rng.permutation(16)
    transform_conv_bn(
        model.conv1, model.bn1, input_permutation, stem_output
    )
    records.append({
        "site": "stem_output",
        "width": 16,
        "digest": permutation_digest(stem_output),
    })
    current = stem_output

    for stage_name in ("layer1", "layer2", "layer3"):
        stage = getattr(model, stage_name)
        for block_index, block in enumerate(stage):
            input_channels = block.conv1.in_channels
            output_channels = block.conv1.out_channels
            if len(current) != input_channels:
                raise RuntimeError("permutation width does not match block input")

            middle = rng.permutation(output_channels)
            has_projection = len(block.shortcut) != 0
            if has_projection:
                output = rng.permutation(output_channels)
            else:
                output = np.asarray(current, dtype=np.int64).copy()

            transform_conv_bn(
                block.conv1, block.bn1, current, middle
            )
            transform_conv_bn(
                block.conv2, block.bn2, middle, output
            )
            if has_projection:
                transform_conv_bn(
                    block.shortcut[0], block.shortcut[1], current, output
                )

            records.extend([
                {
                    "site": f"{stage_name}.{block_index}.middle",
                    "width": int(output_channels),
                    "digest": permutation_digest(middle),
                },
                {
                    "site": f"{stage_name}.{block_index}.output",
                    "width": int(output_channels),
                    "digest": permutation_digest(output),
                    "constrained_by_identity_shortcut": not has_projection,
                },
            ])
            current = output

    final_index = as_index(current, model.fc.weight.device)
    with torch.no_grad():
        model.fc.weight.copy_(
            model.fc.weight.detach().clone()[:, final_index]
        )
    return model.eval(), records


def parameter_difference(first, second):
    changed = 0
    total = 0
    max_difference = 0.0
    for (name_a, tensor_a), (name_b, tensor_b) in zip(
        first.state_dict().items(), second.state_dict().items()
    ):
        if name_a != name_b:
            raise RuntimeError("state dict order mismatch")
        if tensor_a.dtype.is_floating_point:
            difference = torch.abs(tensor_a - tensor_b)
            changed += int(torch.count_nonzero(difference).item())
            total += tensor_a.numel()
            if difference.numel():
                max_difference = max(
                    max_difference, float(difference.max().item())
                )
    return {
        "changed_coordinates": changed,
        "floating_coordinates": total,
        "changed_fraction": changed / total,
        "max_parameter_difference": max_difference,
    }


def test_loader(data_dir, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )


def evaluate(original, equivalent, loader):
    original_correct = 0
    equivalent_correct = 0
    agreement = 0
    total = 0
    max_logit_error = 0.0
    sum_logit_error = 0.0
    logit_count = 0
    with torch.no_grad():
        for inputs, targets in loader:
            first = original(inputs)
            second = equivalent(inputs)
            difference = torch.abs(first - second)
            max_logit_error = max(
                max_logit_error, float(difference.max().item())
            )
            sum_logit_error += float(difference.sum().item())
            logit_count += difference.numel()
            first_prediction = first.argmax(dim=1)
            second_prediction = second.argmax(dim=1)
            original_correct += int(
                first_prediction.eq(targets).sum().item()
            )
            equivalent_correct += int(
                second_prediction.eq(targets).sum().item()
            )
            agreement += int(
                first_prediction.eq(second_prediction).sum().item()
            )
            total += targets.size(0)
    return {
        "examples": total,
        "original_accuracy_percent": 100.0 * original_correct / total,
        "equivalent_accuracy_percent": 100.0 * equivalent_correct / total,
        "prediction_agreement_percent": 100.0 * agreement / total,
        "max_absolute_logit_error": max_logit_error,
        "mean_absolute_logit_error": sum_logit_error / logit_count,
    }


def main():
    args = parse_args()
    started = time.time()
    original, checkpoint = load_model(args.checkpoint)
    equivalent, sites = build_equivalent_model(original, args.seed)
    differences = parameter_difference(original, equivalent)
    evaluation = evaluate(
        original, equivalent, test_loader(args.data_dir, args.batch_size)
    )

    independent_sites = [
        row for row in sites
        if not row.get("constrained_by_identity_shortcut", False)
    ]
    log2_parameterizations = sum(
        math.lgamma(row["width"] + 1) / math.log(2)
        for row in independent_sites
    )
    output = {
        "experiment": "trained ResNet-20 permutation-equivalent clone",
        "checkpoint": str(args.checkpoint),
        "checkpoint_test_accuracy": (
            checkpoint.get("test_accuracy")
            if isinstance(checkpoint, dict) else None
        ),
        "seed": args.seed,
        "permutation_sites": sites,
        "independent_permutation_sites": len(independent_sites),
        "log2_consistent_parameterizations": log2_parameterizations,
        "parameter_difference": differences,
        "functional_equivalence": evaluation,
        "interpretation": (
            "The hidden channel ordering selects a parameterization, not a "
            "different network function.  Guessing that ordering is unnecessary "
            "when the security goal excludes functionally equivalent extraction."
        ),
        "wall_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
