# Invariant-obfuscation experiments

This artifact extends the STIP-only branch with a common maximal-invariant
analysis and an official-code GELO candidate-set attack.

## Pinned external code

GELO is not vendored. Clone https://github.com/noskill/gelo beside this artifact
and checkout commit 786668f20936ae794d4e39483f401492e82d257e. The imported batch
implementation is gelo/train_learned_attack.py:MixedBatchDataset, whose SHA-256
is a4c9b7b2c6d8b6b2963c8138935a1839b93025e2fdd6ffb89ca9ac9361537853.

## Main runner

scripts/68_gelo_official_dataset_attack.py extracts public candidate hidden
states, asks the official GELO dataset to sample and mix rows, and ranks
candidates by residual to the observed rowspace. It does not learn or invert
the mixing matrix.

Retained headline outputs:

- outputs/invariant_obfuscation/gelo_official_dataset_gpt2_512x32_10trial_20260802.json
- outputs/invariant_obfuscation/gelo_official_dataset_gpt2medium_256x32_10trial_20260802.json
- outputs/invariant_obfuscation/gelo_official_dataset_codellama7b_128x16_10trial_20260802.json

Every JSON records the Hugging Face model revision, GELO commit, imported-code
hash, candidate-text hashes, seed, condition, singular values, attack metrics,
and same-rank random-subspace control.

## Verification

Run from the artifact root:

    PYTHONPATH=. python3 -m pytest -q \
      tests/test_stip_orbit.py \
      tests/test_ndss_results.py \
      tests/test_maximal_invariants.py

The retained 2026-08-02 log reports 10 passed tests. The attack and tests are
CPU-compatible. A GPU only accelerates public hidden-state extraction.

