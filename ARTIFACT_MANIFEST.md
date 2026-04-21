# Submission Artifact Manifest

This manifest describes the **canonical submission artifact** for
“Shuffling is Not Enough: An Impossibility Result for Permutation-Based
Model Confidentiality”.

## Canonical entrypoints

- `run_all.sh`
  Reproduces the core theorem-supporting experiments and appendix checks.
- `run_kd_submission.sh`
  Reproduces the main-text KD table under the canonical submission setting
  (quantized teacher logits, fixed random subset, 10 epochs, 10 seeds).
- `run_submission_artifact.sh`
  Runs the two scripts above in sequence.
- `prepare_submission_artifact.sh`
  Stages a clean artifact directory that contains only the canonical files.

## Canonical outputs

- `outputs/logs/00_train_cifar.log`
- `outputs/logs/02_dp_analysis.log`
- `outputs/logs/08_cifar_trained.log`
- `outputs/logs/11_tradeoff_cifar.log`
- `outputs/logs/14_lineage_aggressive.log`
- `outputs/logs/15_bias_recovery.log`
- `outputs/logs/16_noise_fingerprinting.log`
- `outputs/logs/22_tfhe_validate.log`
- `outputs/logs/24_tfhe_real.log`
- `outputs/logs/25_tfhe_resnet_transcript.log`
- `outputs/logs/25_tfhe_resnet_transcript.json`
- `outputs/logs/25_tfhe_resnet_transcript_doublecheck.log`
- `outputs/logs/25_tfhe_resnet_transcript_doublecheck.json`
- `outputs/logs/25_tfhe_all21.json` (full 21-layer transcript; 11,424 total queries reported in paper)
- `outputs/logs/30_imagenet_pretrained.log`
- `outputs/logs/31_imagenet_architectures.log`
- `outputs/logs/32_imagenet_lineage.log`
- `outputs/logs/33_imagenet_tradeoff.log`
- `outputs/logs/11b_per_layer_resnet56_jota.log` (R56 per-layer tradeoff; canonical source for tab:tradeoff R56 rows)
- `outputs/logs/cifar_kd_q2000_e10_s10_quantized_random.log`
- `outputs/logs/cifar_kd_q5000_e10_s10_quantized_random.log`
- `outputs/cifar_kd_q2000_e10_s10_quantized_random.json`
- `outputs/cifar_kd_q5000_e10_s10_quantized_random.json`
- `outputs/figures/fig_dp_vacuousness.png`

## Canonical included scripts

- `scripts/00_train_cifar.py`
- `scripts/02_dp_analysis.py`
- `scripts/08_cifar_trained.py`
- `scripts/11_tradeoff_cifar.py`
- `scripts/14_lineage_aggressive.py`
- `scripts/15_bias_recovery.py`
- `scripts/16_noise_fingerprinting.py`
- `scripts/18_kd_spectral_priors_cifar.py`
- `scripts/22_tfhe_validate.py`
- `scripts/24_tfhe_real.py`
- `scripts/25_tfhe_resnet_transcript.py`
- `scripts/30_imagenet_pretrained.py`
- `scripts/31_imagenet_architectures.py`
- `scripts/32_imagenet_lineage.py`
- `scripts/33_imagenet_tradeoff.py`

## Canonical KD setting

- Teacher checkpoint: `models/resnet20_seed0.pt`
- Teacher logits: quantized checkpoint forward pass (`p = 256`)
- Query subsets: fixed random subsets with `subset_seed = 20260409` (ResNet-20 canonical KD)
- ResNet-56 KD ablation (Appendix) uses `random.seed(0)` per `cifar_kd_r56_*` JSONs; matches paper §A description
- Query budgets: `2,000` and `5,000`
- Student architecture: `ResNet20`
- Distillation epochs: `10`
- Seeds: `10`
- Optimizer: SGD (`lr = 0.05`, `momentum = 0.9`, `weight_decay = 1e-4`)
- Scheduler: CosineAnnealingLR with `T_max = 10`
- Temperature: `4.0`
- Spectral prior weight: `5.0`
- Spectral scope: all `21` convolutional layers

## Notes

- Absolute machine-local paths are intentionally excluded from the canonical
  output files included in the submission artifact.
- This manifest describes only the reviewer-facing bundle produced by
  `prepare_submission_artifact.sh`.
- The reviewer-facing bundle intentionally excludes exploratory material such as
  `outputs/legacy/`, `scripts/17_kd_spectral_priors_mnist.py`, and the negative
  CKKS transcript prototype `scripts/26_ckks_resnet_transcript.py`.
