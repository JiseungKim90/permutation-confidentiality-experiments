import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "invariant_obfuscation"
OFFICIAL = "786668f20936ae794d4e39483f401492e82d257e"
CODE_SHA = "a4c9b7b2c6d8b6b2963c8138935a1839b93025e2fdd6ffb89ca9ac9361537853"

CASES = {
    "gpt2": (
        "gelo_official_dataset_gpt2_512x32_10trial_20260802.json",
        {4: (10, 1.0, 7, 0.925, 10, 1.0),
         8: (10, 1.0, 10, 1.0, 10, 1.0),
         12: (10, 1.0, 10, 1.0, 10, 1.0)},
        768,
    ),
    "gpt2-medium": (
        "gelo_official_dataset_gpt2medium_256x32_10trial_20260802.json",
        {8: (10, 1.0, 10, 1.0, 10, 1.0),
         16: (10, 1.0, 10, 1.0, 10, 1.0),
         24: (7, 0.925, 9, 0.975, 10, 1.0)},
        1024,
    ),
    "codellama/CodeLlama-7b-hf": (
        "gelo_official_dataset_codellama7b_128x16_10trial_20260802.json",
        {10: (10, 1.0, 2, 0.75, 8, 0.94)},
        4096,
    ),
}


def select(results, *, layer, shield, precision, mixing):
    rows = [
        row for row in results
        if row["layer"] == layer
        and row["shield_kind"] == shield
        and row["precision"] == precision
        and row["mixing"] == mixing
    ]
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize("model", CASES)
def test_paper_table_and_provenance(model):
    filename, expected, hidden_dim = CASES[model]
    data = json.loads((OUT / filename).read_text())
    assert data["model"] == model
    assert data["official_commit"] == OFFICIAL
    assert data["official_batch_code_sha256"] == CODE_SHA
    assert all(row["exact_sets"] == 10 for row in data["results"]
               if row["precision"] == "float32")
    assert max(row["mean_control_recall"] for row in data["results"]) <= 0.075
    assert max(row["row_max"] for row in data["results"]) < hidden_dim

    for layer, values in expected.items():
        gauss_exact, gauss_recall, man_exact, man_recall, union_exact, union_recall = values
        gauss = select(data["results"], layer=layer, shield="gaussian",
                       precision="bfloat16", mixing="non_orthogonal")
        manifold = select(data["results"], layer=layer, shield="manifold",
                          precision="bfloat16", mixing="non_orthogonal")
        assert (gauss["exact_sets"], gauss["mean_recall"]) == pytest.approx(
            (gauss_exact, gauss_recall)
        )
        assert (manifold["exact_sets"], manifold["mean_recall"]) == pytest.approx(
            (man_exact, man_recall)
        )
        assert (
            manifold["exact_candidate_unions"],
            manifold["mean_candidate_union_recall"],
        ) == pytest.approx((union_exact, union_recall))

