# Journal extension artifact map

## Environment

- Experiment server: `lab614`
- Canonical server root: `/home/user/research-vault/projects/P050/experiments`
- Checkpoint: `/home/user/research-vault/projects/CS615/models/resnet20_seed0.pt`
- Concrete-TFHE: Concrete Python 2.11.0
- Concrete environment: `PYTHONPATH=/home/user/venvs/p050-concrete211/site`
- Main seed: 20260801
- Data-side server root: `/home/user/p050_journal_targets`
- Data-side model cache: `/home/user/p050_journal_targets/models`
- Data-side seed: 20260802

All model, cryptographic, lifecycle, equivalence, audit-bypass, and figure computations were executed on the server. Scripts 55 and 56 are document/artifact audits and run in the journal workspace.

## Script map

| Script | Purpose | Canonical output |
|---|---|---|
| 40 | low-query row recovery | `outputs/unlabeled_row_recovery.json` |
| 40-orbit | exhaustive private-embedding orbit recovery and PEFT inspection | `outputs/data_side/private_embedding_orbits_exact_fullvocab_20260802.json` |
| 43-STIP | exact public-margin certificate, partial-row-update, vocabulary-overlap controls | `outputs/data_side/stip_margin_phase_exact_rows_20260802.json` |
| 41 | coefficient collision stress | `outputs/mixed_query_collision_stress.json` |
| 41-GELO | formula-level row-space candidate-presence recovery | `outputs/data_side/gelo_rowspace_*.json` |
| 42-GELO | official-wrapper remote-input replay | `outputs/data_side/gelo_official_transcript_gpt2_20260802.json` |
| 42 | activation-mask finite sanity check | `outputs/activation_mask_classification.json` |
| 43 | original extension figure | `fig_journal_extension.pdf/.png` |
| 44 | ImageNet first-layer anchors | `outputs/imagenet_first_layer_row_recovery.json` |
| 45 | first-layer Concrete-TFHE replay | `outputs/logs/45_tfhe_mixed_row_recovery.json` |
| 46 | deterministic interpolation | `outputs/interpolation_recovery.json` |
| 47 | permutation session lifecycle | `outputs/session_lifecycle_recovery.json` |
| 48 | permutation-equivalent ResNet-20 | `outputs/resnet_permutation_equivalence.json` |
| 49 | fixed-session later-layer TFHE | `outputs/logs/49_tfhe_fixed_session_later_layer.json` |
| 50 | dense offset audit bypass | `outputs/offset_query_audit.json` |
| 51 | final journal figure | `fig_journal_extension_v2.pdf/.png` |
| 54 | protected first-activation boundary | `outputs/logs/54_tfhe_protected_activation_input255.json`, `...input31.json` |
| 55 | preliminary conference overlap audit | `outputs/conference_overlap_audit.json` |
| 56 | manuscript-to-JSON consistency audit | `outputs/artifact_consistency_audit.json` |
| 57 | fresh-session layer-orbit recovery at 4/5/8 bits | `outputs/fresh_session_orbit_recovery_input15.json`, `...input31.json`, `...input255.json` |
| 58 | state-machine-compliant fresh-session later-layer Concrete-TFHE | `outputs/logs/58_tfhe_state_machine_fresh_session_orbit_recovery.json` |
| 59 | verify journal and data-side SHA-256 manifests | `outputs/JOURNAL_SHA256SUMS.txt`, `outputs/data_side/SHA256SUMS.txt` |
| 60 | actual LM-loss adaptation of selected tied GPT-2 embedding rows | `outputs/data_side/stip_independent_embedding_frequency_f025_seed*.json` |
| 60b | sequential server grid runner with per-run stdout logs | `outputs/data_side/logs/stip_independent_embedding_grid_*.log` |
| 61 | aggregate seed statistics for actually changed rows | `outputs/data_side/stip_independent_embedding_frequency_f025_3seed_20260802.json` |

## Server commands for the data-side orbit suite

From `/home/user/p050_journal_targets/experiments`:

```bash
python3 40_orbit_private_embeddings.py --cache-dir /home/user/p050_journal_targets/models --samples 100000 --seed 20260802 --index flat --output /home/user/p050_journal_targets/results/private_embedding_orbits_exact_fullvocab_20260802.json
python3 43_stip_margin_phase_diagram.py --cache-dir /home/user/p050_journal_targets/models --output /home/user/p050_journal_targets/results/stip_margin_phase_exact_rows_20260802.json --detail-dir /home/user/p050_journal_targets/results/stip_margin_detail_exact_rows_20260802 --variants all --batch-size 512 --threads 16 --seed 20260802
python3 41_gelo_rowspace_presence.py --cache-dir /home/user/p050_journal_targets/models --model gpt2 --candidates 128 --seq-len 16 --source-count 4 --trials 20 --seed 20260802 --layers 2 4 8 12 --output /home/user/p050_journal_targets/results/gelo_rowspace_gpt2_20260802.json
python3 41_gelo_rowspace_presence.py --cache-dir /home/user/p050_journal_targets/models --model gpt2 --candidates 512 --seq-len 32 --source-count 4 --trials 20 --seed 20260802 --layers 2 4 8 12 --output /home/user/p050_journal_targets/results/gelo_rowspace_gpt2_512x32_20260802.json
python3 41_gelo_rowspace_presence.py --cache-dir /home/user/p050_journal_targets/models --model gpt2-medium --candidates 128 --seq-len 16 --source-count 4 --trials 20 --seed 20260802 --layers 4 8 16 24 --output /home/user/p050_journal_targets/results/gelo_rowspace_gpt2medium_20260802.json
python3 42_gelo_official_transcript.py --official-repo /home/user/p050_journal_targets/gelo-official --model gpt2 --cache-dir /home/user/p050_journal_targets/models --candidates 128 --seq-len 16 --source-count 4 --trials 20 --seed 20260802 --layers 2 4 8 12 --output /home/user/p050_journal_targets/results/gelo_official_transcript_gpt2_20260802.json
SEEDS="20260802 20260803 20260804" FRACTIONS="0.25" bash 60b_run_stip_independent_grid.sh
python3 61_aggregate_stip_independent_training.py --input /home/user/p050_journal_targets/results/stip_independent_embedding_freq_f025_seed20260802.json --input /home/user/p050_journal_targets/results/stip_independent_embedding_frequency_f025_seed20260803.json --input /home/user/p050_journal_targets/results/stip_independent_embedding_frequency_f025_seed20260804.json --output /home/user/p050_journal_targets/results/stip_independent_embedding_frequency_f025_3seed_20260802.json
```

`40_orbit_private_embeddings.py` also inspects the pinned LoRA artifacts recorded in its JSON.  Every model repository and revision, dataset, seed, condition, and per-setting result is stored in the corresponding output.

## Server commands for the new P0 suite

From the canonical server root:

```bash
PYTHONPATH=/home/user/venvs/p050-concrete211/site python3 scripts/46_interpolation_recovery.py
PYTHONPATH=/home/user/venvs/p050-concrete211/site python3 scripts/47_session_lifecycle_recovery.py
PYTHONPATH=/home/user/venvs/p050-concrete211/site python3 scripts/48_resnet_permutation_equivalence.py
PYTHONPATH=/home/user/venvs/p050-concrete211/site python3 scripts/49_tfhe_fixed_session_later_layer.py
PYTHONPATH=/home/user/venvs/p050-concrete211/site python3 scripts/50_offset_query_audit.py
PYTHONPATH=/home/user/venvs/p050-concrete211/site python3 scripts/51_plot_journal_extension_v2.py
PYTHONPATH=/home/user/venvs/p050-concrete211/site python3 scripts/54_tfhe_protected_activation_boundary.py --checkpoint /home/user/research-vault/projects/CS615/models/resnet20_seed0.pt --input-max 255
python3 scripts/57_fresh_session_orbit_recovery.py --checkpoint /home/user/research-vault/projects/CS615/models/resnet20_seed0.pt --input-max 15 --output outputs/fresh_session_orbit_recovery_input15.json
python3 scripts/57_fresh_session_orbit_recovery.py --checkpoint /home/user/research-vault/projects/CS615/models/resnet20_seed0.pt --input-max 31 --output outputs/fresh_session_orbit_recovery_input31.json
python3 scripts/57_fresh_session_orbit_recovery.py --checkpoint /home/user/research-vault/projects/CS615/models/resnet20_seed0.pt --input-max 255 --output outputs/fresh_session_orbit_recovery_input255.json
PYTHONPATH=/home/user/venvs/p050-concrete211/site python3 scripts/58_tfhe_fresh_session_orbit_recovery.py --checkpoint /home/user/research-vault/projects/CS615/models/resnet20_seed0.pt --layer layer2.0.shortcut.0 --input-max 255 --target-round 5 --seed 20260801 --output outputs/logs/58_tfhe_state_machine_fresh_session_orbit_recovery.json
```

## Fresh-session interpretation

Script 57 fixes the attacker choice to the largest admissible constant input; it does not search model parameters for a favorable anchor. All 21 layers are attempted in every trial. Exact layer counts are 3/21 at 4 bits, 5/21 at 5 bits, and 15/21 at 8 bits, with 9/63, 15/63, and 45/63 exact trials respectively. These counts refer to local convolution kernel operators. The complete 10x64 final classifier is additionally evaluated: it fails at 4 and 5 bits but is gap-certified and exact in all three 8-bit trials, using 1,152 queries. Script 58 executes the selected 32x16 affine layer in Concrete-TFHE behind an executable session wrapper. Every one of the 267 encrypted queries receives a unique session ID, advances monotonically to target round 5, and uses that pair once; 267 deliberate duplicate submissions are rejected before evaluation. Fresh input/output permutations are derived per accepted session/round pair, and the attack still recovers all 16 columns plus the bias. Earlier rounds are state transitions, not full encrypted-network execution. The file `outputs/logs/58_environment_numpy2_incompatibility.log` records the incompatible global NumPy 2.2.6 environment; the successful run uses the isolated NumPy 1.26.4 path above.

## Protected-activation interpretation

The folded first affine baseline is exact for 8-bit inputs. The direct affine-plus-ReLU circuit is intentionally recorded as `compile_unsupported`: its 18-bit accumulator feeds a table lookup, while Concrete 2.11 supports at most 16-bit table lookups. This is a backend/parameter boundary, not a theorem that protected activation is impossible. Rescaling, decomposition, another secure-nonlinearity protocol, or a different backend requires a separate end-to-end cost evaluation.

For the reduced `[0,31]` input range, the protected circuit is exact but takes 58.77 seconds per vector versus 18.8 milliseconds for the affine baseline, a 3,122x online ratio. This five-bit analogue is not an accuracy-equivalent replacement for the eight-bit deployment.

## Data-side interpretation

The exact FlatIP run searches the entire public vocabulary for every deployed token row.  GPT-2 IMDB recovers 96.4323% top-1 from word rows and 100% from the position-zero representation over 50,257 tokens.  BERT SST-2 recovers 99.9148% and 100%, respectively, over 30,522 tokens.  The two inspected GPT-2 LoRA artifacts contain no embedding state, saved embedding module, or trainable token indices; the exact theorem applies to those artifacts, not to every PEFT configuration.

Script 43 computes every public signature's exact nearest-neighbor separation and applies the sufficient half-margin certificate.  Certified rates are 73.5539% (GPT-2 word), 97.4969% (GPT-2 position zero), 99.9115% (BERT word), and 99.9869% (BERT position zero).  The token-ID reindexing control changes no rate, and all four private signature dictionaries have zero bitwise row collisions.  Its older partial-row campaign is a controlled substitution study, not a trained checkpoint; the manuscript now uses the independently trained evidence below.  The vocabulary-overlap control counts unaligned rows as semantic failures and preserves only pseudotoken linkage.  Full per-token arrays remain server-side in `/home/user/p050_journal_targets/results/stip_margin_detail_exact_rows_20260802` (3.1 MiB); the compact JSON and log are copied locally.

Scripts 60--61 train three separate GPT-2 partial-embedding checkpoints on 500,000 SciFact/NFCorpus tokens.  The transformer is frozen; only the most frequent 25% of active tied embedding rows receive gradients for 200 steps.  Across seeds 20260802--20260804, all 13,285 selected rows change and zero unselected rows change.  Search over the actually changed rows gives mean/minimum Top-1 96.8537%/96.7503%, mean Top-5 97.1852%, and mean sufficient-certificate rate 32.2079%; mean validation loss falls from 10.2809 to 9.1568.  The three checkpoints remain server-side under `/home/user/p050_journal_targets/checkpoints`; their SHA-256 values, run parameters, per-seed results, JSONs, and stdout logs are retained without committing model weights.

The formula-level GELO suite contains 24 settings per experiment family: four layers, six mixing/precision/shield conditions, and 20 trials.  All 1,440 source sets are recovered exactly.  The official-wrapper replay pins commit `786668f20936ae794d4e39483f401492e82d257e` and captures the exact tensor sent to `GeloObfuscatedLinear._project`; all 240 source sets are exact.  The combined 1,680 trials establish candidate presence under complete associated batch rows.  They do not claim deployed-service exploitation, stream association, or free-form prompt reconstruction.

Data-side JSON and logs are under `outputs/data_side`.  `outputs/data_side/SHA256SUMS.txt` binds the copied claim artifacts.

## Log retention and integrity

The `extended-version` branch retains every compact JSON and stdout log used by a journal claim, plus the failed NumPy 2 environment log that explains the isolated Concrete runtime. Large per-token arrays and third-party model caches remain on the server at the paths recorded above; they are not silently treated as Git artifacts. Exploratory data-side runs are retained under `outputs/data_side`, while the exact full-vocabulary, margin, three-seed trained-embedding, GELO formula-level, and official-wrapper files named in the manuscript are the canonical claim artifacts.

`outputs/JOURNAL_SHA256SUMS.txt` binds all journal-extension JSON, logs, and rendered summary figures copied to the branch. `outputs/data_side/SHA256SUMS.txt` independently binds the full data-side directory copied from `lab614`. Regenerate a result under a new filename rather than overwriting a canonical file; update the appropriate checksum and the artifact-consistency audit together.
## Integrity checks

From a standalone checkout of the `extended-version` branch:

```bash
python scripts/59_verify_journal_checksums.py
```

For document-to-artifact checks, run from `Journal-TDSC`:

```powershell
python .\experiments\scripts\55_conference_overlap_audit.py
python .\experiments\scripts\56_artifact_consistency_audit.py
pdflatex -interaction=nonstopmode -halt-on-error tdsc_main.tex
pdflatex -interaction=nonstopmode -halt-on-error tdsc_main.tex
```

The overlap audit is only a normalized 8-word shingle diagnostic and does not replace Crossref Similarity Check.
