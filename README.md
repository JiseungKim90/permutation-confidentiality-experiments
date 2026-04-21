# Artifact: Shuffling is Not Enough

Reproduces all experiments in
"Shuffling is Not Enough: An Impossibility Result for Permutation-Based
Model Confidentiality" (ESORICS 2026).

## Requirements

Python 3.9+ with:

```
pip install -r requirements.txt
```

Optional: `concrete-python>=2.11` is required only for
`24_tfhe_real.py` and `25_tfhe_resnet_transcript.py`
(the real TFHE appendix checks). All other scripts run without it.

Tested with PyTorch 2.7.1, torchvision 0.22.1, NumPy 1.26.4,
TenSEAL 0.3.16, concrete-python 2.11.0 on Linux (x86-64, CPU).

## Data Setup

Datasets are **not** included in this repository. Before running experiments,
download them into the `data/` directory:

**CIFAR-10** (required by scripts 00, 08, 11, 14, 15, 16, 18):
```bash
mkdir -p data && cd data
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar xzf cifar-10-python.tar.gz
rm cifar-10-python.tar.gz
cd ..
```

**ImageNette** (required by scripts 33):
```bash
mkdir -p data && cd data
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar xzf imagenette2-320.tgz
rm imagenette2-320.tgz
cd ..
```

After setup, your `data/` directory should look like:
```
data/
  cifar-10-batches-py/   # CIFAR-10
  imagenette2-320/       # ImageNette (10-class ImageNet subset)
```

ImageNet-scale scripts (30, 31, 32) use **pretrained torchvision weights**
downloaded automatically at runtime; no manual data download is needed.

## Reproduction

```
cd experiments
bash run_all.sh
```

Expected total runtime: ~7 hours on a single CPU (dominated by
`00_train_cifar.py` ~5h and `14_lineage_aggressive.py` ~1.5h).
Models from `00_train_cifar.py` are cached; re-runs of other scripts
complete in under 15 minutes each.

For a one-command reproduction of the full submission artifact
(including the KD table used in the main text), run:

```
cd experiments
bash run_submission_artifact.sh
```

To stage a clean reviewer-facing bundle containing only the canonical
submission files, run:

```
cd experiments
bash prepare_submission_artifact.sh
```

The canonical file set is described in
`ARTIFACT_MANIFEST.md`.

## Submission KD Table

The submission additionally reports the CIFAR-10 / ResNet-20
logit-distillation experiment with spectral priors
(`2,000` and `5,000` queries; `10` distillation epochs; `10` seeds).
After generating the CIFAR checkpoints via `run_all.sh`, reproduce
that table with:

```
cd experiments
bash run_kd_submission.sh
```

This produces:

```
outputs/cifar_kd_q2000_e10_s10_quantized_random.json
outputs/cifar_kd_q5000_e10_s10_quantized_random.json
outputs/logs/cifar_kd_q2000_e10_s10_quantized_random.log
outputs/logs/cifar_kd_q5000_e10_s10_quantized_random.log
```

The exact KD hyperparameters used in the submission are:

| Parameter | Value |
|---|---|
| Teacher | `models/resnet20_seed0.pt` |
| Teacher logits | quantized checkpoint forward (`p = 256`) |
| Student | same-architecture `ResNet20` |
| Query budgets | `2,000`, `5,000` |
| Query subset | fixed random subset (`subset_seed = 0`) |
| Distillation epochs | `10` |
| Seeds | `10` |
| Optimizer | SGD |
| Learning rate | `0.05` |
| Momentum | `0.9` |
| Weight decay | `1e-4` |
| Scheduler | CosineAnnealingLR (`T_max=10`) |
| Temperature | `4.0` |
| Spectral prior weight | `5.0` |
| Spectral scope | all `21` convolutional layers |

The JSON outputs store this configuration together with
per-seed histories and epoch-wise summaries. Each history entry
records `lr_train` (the learning rate used during that epoch),
`lr_next` (the learning rate after the scheduler step),
`kd_loss`, `reg_loss`, and `test_acc`.

## Script-to-Paper Mapping

| Script | Section / Table | Description |
|---|---|---|
| `00_train_cifar.py` | Appendix A | Train 20 ResNet-20 models on CIFAR-10 |
| `02_dp_analysis.py` | Sec 3.3, Tab 1 | Local DP parameter ε₀ (Gaussian mechanism formula) |
| `08_cifar_trained.py` | Sec 4, Tab 2, Tab A.1, Tab A.2 | Exact recovery on trained models (all precisions, all layers) |
| `11_tradeoff_cifar.py` | Sec 5.4, Tab 4 | Privacy-utility tradeoff (attack error, FP acc, pred. agr., test acc.) |
| `14_lineage_aggressive.py` | Sec 5.3 | Lineage detection under 3 fine-tuning regimes |
| `15_bias_recovery.py` | Appendix A | Bias-having MLP attack |
| `16_noise_fingerprinting.py` | Sec 5.4, Tab 4 (FP acc. column) | Fingerprinting accuracy under increasing noise |
| `18_kd_spectral_priors_cifar.py` | Sec 5 (impact), Tab. `tab:kd` | Logit distillation with and without recovered spectral priors |
| `22_tfhe_validate.py` | Appendix A | CKKS check, concrete-python identity proxy, and simulated TFHE noise validation |
| `24_tfhe_real.py` | Appendix A | Small synthetic Concrete sanity check plus larger simulated quantized layer |
| `25_tfhe_resnet_transcript.py` | Appendix A | Trained-layer TFHE single-round transcript check under fixed/fresh permutations |

## Pre-computed Logs

`outputs/logs/` contains the stdout logs from the canonical server run
that produced the paper's numbers. Each log corresponds to one script:

| Log | Key outputs |
|---|---|
| `02_dp_analysis.log` | Tab. 1: ε₀ values (10,491 / 166,316 / 2,727,577) |
| `08_cifar_trained.log` | Exact recovery on 20 models × 3 precisions |
| `11_tradeoff_cifar.log` | Tab. 4: attack error, pred. agr., test acc. (N=100 trials) |
| `14_lineage_aggressive.log` | Sec. 5.3: lineage accuracy, sep. ratios (3 regimes) |
| `15_bias_recovery.log` | Appendix A: bias-having MLP exact recovery |
| `16_noise_fingerprinting.log` | Tab. 4 FP acc. column: fingerprinting under noise |
| `cifar_kd_q2000_e10_s10_quantized_random.log` | KD table: 2,000-query run (`10` epochs, `10` seeds; quantized logits, random subset) |
| `cifar_kd_q5000_e10_s10_quantized_random.log` | KD table: 5,000-query run (`10` epochs, `10` seeds; quantized logits, random subset) |
| `22_tfhe_validate.log` | Appendix A: CKKS check, TFHE identity proxy, and simulated post-bootstrapping noise validation |
| `24_tfhe_real.log` | Appendix A: small synthetic Concrete sanity check plus larger simulated quantized layer |
| `25_tfhe_resnet_transcript.log` | Appendix A: trained-layer TFHE single-round transcript check |
| `25_tfhe_resnet_transcript_doublecheck.log` | Appendix A: independent second-checkpoint TFHE transcript double-check |

The corresponding machine-readable summaries are included in:

| JSON | Key outputs |
|---|---|
| `cifar_kd_q2000_e10_s10_quantized_random.json` | Config, query indices, per-seed histories, and epoch-wise summaries for the 2,000-query KD run |
| `cifar_kd_q5000_e10_s10_quantized_random.json` | Config, query indices, per-seed histories, and epoch-wise summaries for the 5,000-query KD run |

Older exploratory KD runs are retained under `outputs/legacy/` for record
keeping, but they are not part of the submission artifact.
The negative CKKS transcript prototype (`26_ckks_resnet_transcript.py`
and its logs) is likewise not part of the submission artifact.

## Directory Structure

```
experiments/
  ARTIFACT_MANIFEST.md
  lib/             Shared library (attack.py, models.py)
  scripts/         Numbered experiment scripts
  prepare_submission_artifact.sh
  run_kd_submission.sh
  run_submission_artifact.sh
  models/          Trained checkpoints (generated by 00_train_cifar.py)
  data/            CIFAR-10 / MNIST (auto-downloaded)
  outputs/
    *.json         Submission KD summaries
    legacy/        Older exploratory KD outputs (not part of submission)
    logs/          Pre-computed logs (included) and re-run outputs
```

## Notes

- `11_tradeoff_cifar.py` reports pred. agr. and test acc. as the mean
  of 100 independent noise draws (seeds 0-99) for reproducibility.
  Attack error uses np.random.seed(42).
- The 20 trained models (seeds 0-19, mean accuracy 82.8%) are required
  by scripts 08, 11, 14, 16. Run `00_train_cifar.py` first or place
  pre-trained checkpoints in `models/`.
- `18_kd_spectral_priors_cifar.py` requires only
  `models/resnet20_seed0.pt`; `run_kd_submission.sh` checks this
  prerequisite explicitly.
- The submission KD sweep uses quantized teacher logits and a fixed
  random query subset; both choices are recorded in the output JSON.
- `14_lineage_aggressive.py` fine-tunes 15 child models (5 bases x 3
  children) under 3 regimes; the aggressive regime takes ~1.5 hours.
- The KD submission sweep is intentionally separated from `run_all.sh`
  because it is much slower than the original artifact path
  (`10` distillation epochs, `10` seeds, two query budgets).
- `run_submission_artifact.sh` is the top-level entrypoint for
  full-paper reproduction; it runs `run_all.sh` first and then
  `run_kd_submission.sh`.
