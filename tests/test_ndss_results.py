"""Regression tests for every quantitative claim promoted to the NDSS draft."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


_ARTIFACT = Path(__file__).resolve().parents[1]
EVIDENCE = _ARTIFACT / "evidence" / "ndss-2027" / "results"
if not EVIDENCE.exists():
    EVIDENCE = _ARTIFACT.parent / "results" / "raw"


def load(name: str):
    return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))


def test_public_main_claim():
    data = load("stip_final_full_prompt_public_128x32_3trial_20260802.json")
    aggregate = data["aggregate"]
    assert aggregate["token_top1"]["mean"] == 1.0
    assert aggregate["full_prompt_exact"]["mean"] == 1.0
    assert aggregate["raw_permuted_token_top1_control"]["mean"] == pytest.approx(
        8.138020833333333e-05
    )
    assert aggregate["all_invariance_checks_pass"]
    assert aggregate["all_replays_exact"]


def test_private_codebook_claims():
    names = [
        f"stip_chosen_codebook_hybrid_private_seed{seed}_4096q_128x32.json"
        for seed in (20260802, 20260803, 20260804)
    ]
    rows = [load(name)["results"][-1] for name in names]
    assert [row["known_prompt_queries"] for row in rows] == [4096] * 3
    assert sum(row["hybrid_token_top1"] for row in rows) / 3 == pytest.approx(
        0.9921061197916666
    )
    assert sum(row["hybrid_full_prompt_exact"] for row in rows) / 3 == pytest.approx(
        0.7708333333333334
    )
    assert sum(row["fallback_full_prompt_exact"] for row in rows) / 3 == pytest.approx(
        0.049479166666666664
    )
    assert all(row["conditional_accuracy_given_coverage"] == 1.0 for row in rows)
    assert all(row["membership_coverage_agreement"] == 1.0 for row in rows)


def test_private_tokenizer_boundary_claim():
    rows = [
        load(f"tokenizer_bpe4096_seed{seed}_20260802.json")
        for seed in (20260802, 20260803, 20260804)
    ]
    collisions = [
        row["attack"]["private_signature_collision_rows_at_1e-6"] for row in rows
    ]
    assert collisions == [112, 119, 120]
    unique_fraction = sum((4096 - value) / 4096 for value in collisions) / 3
    assert unique_fraction == pytest.approx(0.971435546875)
    assert all(
        row["attack"]["coverage"][0]["token_top1"] == 0.0
        and row["attack"]["coverage"][0]["full_prompt_exact"] == 0.0
        for row in rows
    )


def test_noise_tradeoff_claim():
    data = load("stip_noise_defense_32x16_3trial_20260802.json")
    utility = next(row for row in data["utility"]["aggregate"] if row["sigma"] == 0.05)
    attack_one = next(
        row
        for row in data["attack"]["aggregate"]
        if row["sigma"] == 0.05 and row["repeat_queries"] == 1
    )
    attack_sixteen = next(
        row
        for row in data["attack"]["aggregate"]
        if row["sigma"] == 0.05 and row["repeat_queries"] == 16
    )
    assert utility["clean_top1_agreement_mean"] == pytest.approx(0.8881944219271342)
    assert attack_one["token_top1_mean"] == pytest.approx(0.076171875)
    assert attack_sixteen["token_top1_mean"] == pytest.approx(0.10286458333333333)
