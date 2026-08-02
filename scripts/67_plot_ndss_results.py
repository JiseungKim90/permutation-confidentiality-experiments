#!/usr/bin/env python3
"""Create deterministic TSV inputs and a compact result summary for P064.

This helper intentionally uses only the Python standard library so that result
aggregation does not depend on the server's NumPy/Matplotlib ABI.  The paper
renders the emitted coordinates with PGFPlots.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import mean


def load(path: str | Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(header)]
    lines.extend("\t".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "min": min(values), "max": max(values)}


def public_model_summary(result_dir: Path) -> list[dict]:
    paths = sorted(
        glob.glob(str(result_dir / "stip_final_full_prompt_public*128x32_3trial*.json"))
    )
    if len(paths) < 2:
        raise RuntimeError(f"expected at least two public-model runs, found {len(paths)}")
    output = []
    for path in paths:
        payload = load(path)
        output.append(
            {
                "model": payload["model"],
                "prompt_count": payload["prompt_count"],
                "prompt_length": payload["prompt_length"],
                "trials": payload["trials"],
                "elapsed_seconds": payload["elapsed_seconds"],
                "aggregate": payload["aggregate"],
            }
        )
    return output


def query_budget_summary(result_dir: Path, figure_dir: Path) -> dict:
    paths = sorted(
        glob.glob(
            str(result_dir / "stip_chosen_codebook_hybrid_private_seed*_4096q_128x32.json")
        )
    )
    if len(paths) != 3:
        raise RuntimeError(f"expected three private codebook runs, found {len(paths)}")
    payloads = [load(path) for path in paths]
    budgets = payloads[0]["known_budgets"]
    token = [
        [item["results"][index]["hybrid_token_top1"] for item in payloads]
        for index in range(len(budgets))
    ]
    prompt = [
        [item["results"][index]["hybrid_full_prompt_exact"] for item in payloads]
        for index in range(len(budgets))
    ]
    rows = []
    for budget, token_values, prompt_values in zip(budgets, token, prompt):
        token_stats = aggregate(token_values)
        prompt_stats = aggregate(prompt_values)
        rows.append(
            [
                budget,
                token_stats["mean"],
                token_stats["min"],
                token_stats["max"],
                prompt_stats["mean"],
                prompt_stats["min"],
                prompt_stats["max"],
            ]
        )
    save_tsv(
        figure_dir / "private_codebook_budget.tsv",
        [
            "queries",
            "token_mean",
            "token_min",
            "token_max",
            "prompt_mean",
            "prompt_min",
            "prompt_max",
        ],
        rows,
    )
    return {
        "runs": len(paths),
        "budgets": budgets,
        "hybrid_token": [aggregate(values) for values in token],
        "hybrid_full_prompt": [aggregate(values) for values in prompt],
        "fallback_token_mean": mean(
            item["results"][0]["fallback_token_top1"] for item in payloads
        ),
        "fallback_full_prompt_mean": mean(
            item["results"][0]["fallback_full_prompt_exact"] for item in payloads
        ),
    }


def defense_summary(result_dir: Path, figure_dir: Path) -> dict:
    payload = load(result_dir / "stip_noise_defense_32x16_3trial_20260802.json")
    utility = {row["sigma"]: row for row in payload["utility"]["aggregate"]}
    attack = {
        (row["sigma"], row["repeat_queries"]): row
        for row in payload["attack"]["aggregate"]
    }
    rows = []
    for sigma in sorted(utility):
        rows.append(
            [
                sigma,
                utility[sigma]["clean_top1_agreement_mean"],
                attack[(sigma, 1)]["token_top1_mean"],
                attack[(sigma, 16)]["token_top1_mean"],
                attack[(sigma, 1)]["full_prompt_exact_mean"],
                attack[(sigma, 16)]["full_prompt_exact_mean"],
            ]
        )
    save_tsv(
        figure_dir / "noise_utility_tradeoff.tsv",
        [
            "sigma",
            "clean_top1_agreement",
            "token_repeat_1",
            "token_repeat_16",
            "prompt_repeat_1",
            "prompt_repeat_16",
        ],
        rows,
    )
    return {
        "utility": payload["utility"]["aggregate"],
        "attack_repeat_1": [attack[(sigma, 1)] for sigma in sorted(utility)],
        "attack_repeat_16": [attack[(sigma, 16)] for sigma in sorted(utility)],
        "sigma_to_embedding_std": payload["sigma_to_embedding_std"],
    }


def tokenizer_summary(result_dir: Path) -> dict:
    output = {}
    patterns = {
        "extension_256": "tokenizer_extension_256_seed*_20260802.json",
        "bpe_4096": "tokenizer_bpe4096_seed*_20260802.json",
    }
    for name, pattern in patterns.items():
        payloads = [load(path) for path in sorted(glob.glob(str(result_dir / pattern)))]
        if len(payloads) != 3:
            raise RuntimeError(f"expected three {name} runs, found {len(payloads)}")
        rows = []
        for index, reference in enumerate(payloads[0]["attack"]["coverage"]):
            row = {"private_vocabulary_coverage": reference["private_vocabulary_coverage"]}
            for field in ["token_top1", "full_prompt_exact", "known_token_top1"]:
                row[field] = aggregate(
                    [item["attack"]["coverage"][index][field] for item in payloads]
                )
            rows.append(row)
        output[name] = {
            "runs": len(payloads),
            "pseudotoken_linkage_repeat_stable": all(
                item["attack"]["pseudotoken_linkage_exact_within_tolerance"]
                for item in payloads
            ),
            "collision_rows_at_1e-6": [
                item["attack"]["private_signature_collision_rows_at_1e-6"]
                for item in payloads
            ],
            "coverage": rows,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--figure-dir", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()
    result_dir = Path(args.result_dir)
    figure_dir = Path(args.figure_dir)
    summary = {
        "public_models": public_model_summary(result_dir),
        "private_codebook": query_budget_summary(result_dir, figure_dir),
        "noise_defense": defense_summary(result_dir, figure_dir),
        "private_tokenizer": tokenizer_summary(result_dir),
    }
    path = Path(args.summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()