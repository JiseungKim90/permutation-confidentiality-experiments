# Submission Artifact Manifest

This manifest describes the **canonical submission artifact** for
“Shuffling is Not Enough: An Impossibility Result for Permutation-Based
Model Confidentiality”.

## Canonical entrypoints

- `run_all.sh`
  Reproduces the core theorem-supporting experiments and appendix checks.
- `run_gpu_submission.sh`
  GPU entrypoint for ImageNet experiments and the full KD sweep.
- `run_kd_submission.sh`
  Reproduces the main-text KD table under the canonical submission setting
  (quantized teacher logits, fixed random subset, 100 epochs, 10 seeds).
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
- `outputs/logs/30_imagenet_pretrained.log`
- `outputs/logs/31_imagenet_architectures.log`
- `outputs/logs/32_imagenet_lineage.log`
- `outputs/logs/33_imagenet_tradeoff.log`
- `outputs/logs/kd_r20_q250_e100_s10.log`
- `outputs/logs/kd_r20_q500_e100_s10.log`
- `outputs/logs/kd_r20_q1000_e100_s10.log`
- `outputs/logs/kd_r20_q2000_e100_s10.log`
- `outputs/logs/kd_r20_q5000_e100_s10.log`
- `outputs/logs/kd_r56_q2000_e100_s10.log`
- `outputs/logs/kd_r56_q5000_e100_s10.log`
- `outputs/kd_r20_q250_e100_s10.json`
- `outputs/kd_r20_q500_e100_s10.json`
- `outputs/kd_r20_q1000_e100_s10.json`
- `outputs/kd_r20_q2000_e100_s10.json`
- `outputs/kd_r20_q5000_e100_s10.json`
- `outputs/kd_r56_q2000_e100_s10.json`
- `outputs/kd_r56_q5000_e100_s10.json`
- `outputs/kd_scrambled_r20_q2000_e100_s10.json`
- `outputs/kd_scrambled_r20_q5000_e100_s10.json`
- `outputs/imagenet_lineage_r50.json`
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
- Query subsets: fixed random subsets with `subset_seed = 20260409`
- Query budgets: `250`, `500`, `1,000`, `2,000`, `5,000` (R20); `2,000`, `5,000` (R56)
- Student architectures: `ResNet20` (R20), `ResNet56` (R56)
- Distillation epochs: `100`
- Seeds: `10`
- Optimizer: SGD (`lr = 0.05`, `momentum = 0.9`, `weight_decay = 1e-4`)
- Scheduler: CosineAnnealingLR with `T_max = 100`
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
