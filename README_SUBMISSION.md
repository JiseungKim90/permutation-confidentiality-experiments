# Submission Artifact: Shuffling is Not Enough

This directory is the **canonical reviewer-facing artifact** for
“Shuffling is Not Enough: An Impossibility Result for Permutation-Based
Model Confidentiality”.

It contains only the files that support the final paper claims:
the core scripts, the strict KD outputs, the updated lineage and
tradeoff logs, and the stronger TFHE transcript evidence on trained
ResNet-20 layers. Exploratory material such as legacy KD runs and the
negative CKKS transcript prototype is intentionally excluded.

## Entry points

To reproduce the core paper experiments:

```bash
bash run_all.sh
```

If `concrete-python>=2.11` is installed, `run_all.sh` also reproduces
the appendix TFHE checks, including the trained-layer transcript check
in `scripts/25_tfhe_resnet_transcript.py`.

To reproduce the main-text KD table:

```bash
bash run_kd_submission.sh
```

To reproduce the full canonical submission artifact in one command:

```bash
bash run_submission_artifact.sh
```

## Canonical KD setting

- Teacher checkpoint: `models/resnet20_seed0.pt`
- Teacher logits: quantized checkpoint forward pass (`p = 256`)
- Student: same-architecture `ResNet20`
- Query subsets: fixed random subsets with `subset_seed = 20260409`
- Query budgets: `2,000`, `5,000`
- Distillation epochs: `10`
- Seeds: `10`
- Optimizer: SGD (`lr = 0.05`, `momentum = 0.9`, `weight_decay = 1e-4`)
- Scheduler: CosineAnnealingLR with `T_max = 10`
- Temperature: `4.0`
- Spectral prior weight: `5.0`
- Spectral scope: all `21` convolutional layers

The machine-readable KD summaries are:

- `outputs/cifar_kd_q2000_e10_s10_quantized_random.json`
- `outputs/cifar_kd_q5000_e10_s10_quantized_random.json`

Their corresponding logs are:

- `outputs/logs/cifar_kd_q2000_e10_s10_quantized_random.log`
- `outputs/logs/cifar_kd_q5000_e10_s10_quantized_random.log`

## Validation

To verify that this bundle contains the canonical files and no
machine-local absolute paths, run:

```bash
bash validate_submission_artifact.sh
```

## Notes

- `models/` and `data/` are placeholder directories in the clean bundle.
  Populate them before reproduction, or copy the trained checkpoints and
  datasets into place.
- This bundle intentionally contains only canonical submission files.
