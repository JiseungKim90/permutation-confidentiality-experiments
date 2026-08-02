#!/usr/bin/env python3
"""Aggregate independently trained STIP embedding runs without pooling seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def metric_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.input]
    if len(records) < 2:
        parser.error("at least two independent runs are required")
    keys = ("model", "selection", "train_fraction_of_active", "steps", "learning_rate")
    reference = {key: records[0][key] for key in keys}
    for record in records:
        if not record.get("not_synthetic_row_replacement"):
            raise ValueError("input is not an independently trained run")
        if record["changed_unselected_rows"] != 0:
            raise ValueError("an input changed an unselected row")
        if {key: record[key] for key in keys} != reference:
            raise ValueError("training configurations differ")
        if record["orbit_recovery"] is None:
            raise ValueError("an input omitted orbit recovery")

    seed_rows = []
    for record in sorted(records, key=lambda item: item["seed"]):
        orbit = record["orbit_recovery"]
        seed_rows.append(
            {
                "seed": record["seed"],
                "active_rows": record["active_rows"],
                "selected_rows": record["selected_rows"],
                "changed_rows": record["changed_rows"],
                "changed_unselected_rows": record["changed_unselected_rows"],
                "validation_loss_before": record["validation_loss_before"],
                "validation_loss_after": record["validation_loss_after"],
                "selected_top1": orbit["top1"]["selected"]["rate"],
                "selected_top5": orbit["top5"]["selected"]["rate"],
                "selected_certificate": orbit["half_margin_certificate"]["selected"]["rate"],
                "selected_top1_successes": orbit["top1"]["selected"]["successes"],
                "selected_top5_successes": orbit["top5"]["selected"]["successes"],
                "selected_certificate_successes": orbit["half_margin_certificate"]["selected"]["successes"],
                "checkpoint_sha256": record["checkpoint_sha256"],
            }
        )

    selected_counts = [row["selected_rows"] for row in seed_rows]
    output = {
        "experiment": "aggregate independently trained partial GPT-2 embedding rows",
        "runs": len(seed_rows),
        **reference,
        "seeds": [row["seed"] for row in seed_rows],
        "seed_results": seed_rows,
        "selected_rows": {
            "mean": float(np.mean(selected_counts)),
            "min": int(min(selected_counts)),
            "max": int(max(selected_counts)),
        },
        "selected_top1": metric_summary([row["selected_top1"] for row in seed_rows]),
        "selected_top5": metric_summary([row["selected_top5"] for row in seed_rows]),
        "selected_certificate": metric_summary(
            [row["selected_certificate"] for row in seed_rows]
        ),
        "validation_loss_before": metric_summary(
            [row["validation_loss_before"] for row in seed_rows]
        ),
        "validation_loss_after": metric_summary(
            [row["validation_loss_after"] for row in seed_rows]
        ),
        "pooled_changed_rows": int(sum(selected_counts)),
        "pooled_selected_top1": float(
            sum(row["selected_top1_successes"] for row in seed_rows)
            / sum(selected_counts)
        ),
        "pooled_selected_top5": float(
            sum(row["selected_top5_successes"] for row in seed_rows)
            / sum(selected_counts)
        ),
        "pooled_selected_certificate": float(
            sum(row["selected_certificate_successes"] for row in seed_rows)
            / sum(selected_counts)
        ),
        "interpretation": (
            "Mean, standard deviation, minimum, and maximum are reported across seeds. "
            "Pooled rates are supplementary because selected-row counts vary with the "
            "seeded corpus order."
        ),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(
        (json.dumps(output, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()