"""Aggregate a one-seed STIP trained-row fraction sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = [
        json.loads(Path(name).read_text(encoding="utf-8"))
        for name in args.input
    ]
    if len(records) < 2:
        raise ValueError("the fraction sweep requires at least two runs")

    identity = {
        (row["model"], row["seed"], row["selection"], row["steps"])
        for row in records
    }
    if len(identity) != 1:
        raise ValueError(f"incompatible sweep identities: {identity}")

    rows: list[dict] = []
    for record in sorted(records, key=lambda row: row["train_fraction_of_active"]):
        orbit = record["orbit_recovery"]
        if record["changed_rows"] != record["selected_rows"]:
            raise ValueError("a selected row did not change")
        if record["changed_unselected_rows"] != 0:
            raise ValueError("an unselected row changed")
        rows.append(
            {
                "fraction_of_active": record["train_fraction_of_active"],
                "active_rows": record["active_rows"],
                "selected_rows": record["selected_rows"],
                "changed_rows": record["changed_rows"],
                "changed_unselected_rows": record["changed_unselected_rows"],
                "selected_top1": orbit["top1"]["selected"]["rate"],
                "selected_top5": orbit["top5"]["selected"]["rate"],
                "selected_certificate": orbit["half_margin_certificate"][
                    "selected"
                ]["rate"],
                "validation_loss_before": record["validation_loss_before"],
                "validation_loss_after": record["validation_loss_after"],
                "checkpoint_sha256": record["checkpoint_sha256"],
            }
        )

    fractions = [row["fraction_of_active"] for row in rows]
    if len(set(fractions)) != len(fractions):
        raise ValueError("duplicate fraction in sweep")
    top1 = [row["selected_top1"] for row in rows]
    result = {
        "experiment": "independently trained GPT-2 embedding fraction sweep",
        "interpretation": (
            "One seed locates a possible training-fraction boundary. "
            "The separate 25%-fraction three-seed campaign remains the "
            "primary repeated result."
        ),
        "model": records[0]["model"],
        "seed": records[0]["seed"],
        "selection": records[0]["selection"],
        "steps": records[0]["steps"],
        "runs": len(rows),
        "fractions": fractions,
        "results": rows,
        "selected_top1_min": min(top1),
        "selected_top1_max": max(top1),
        "selected_top1_range": max(top1) - min(top1),
        "all_selected_rows_changed": all(
            row["changed_rows"] == row["selected_rows"] for row in rows
        ),
        "all_unselected_rows_unchanged": all(
            row["changed_unselected_rows"] == 0 for row in rows
        ),
        "no_failure_transition_above_95_percent": min(top1) >= 0.95,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2) + "\n"
    destination.write_bytes(payload.encode("utf-8"))
    print(payload, end="")


if __name__ == "__main__":
    main()