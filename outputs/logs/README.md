# Experiment Log Index

This directory contains the stdout / JSON traces backing every numerical
claim in the P050 paper (main.tex). Each paper table, figure, and inline
number should be traceable to exactly one log or JSON below. Logs marked
`DEPRECATED` are retained for historical reference only and are **not**
cited by the current paper.

## Paper → log mapping

### Table 2: Sorted-spectrum recovery on 20 trained ResNet-20 (CIFAR)
- `00_train_cifar.log` — training run (20 seeds, 10 epochs)
- `08_cifar_trained.log` — attack execution on every conv layer × 4/8/12-bit; max-error = 0 for all 20 models
- `lib/attack.py::attack_layer` path

### Table 3: Exact recovery on 8 ImageNet-pretrained architectures
- `30_imagenet_pretrained.log` — torchvision model loading
- `31_imagenet_architectures.log` — attack run; max-error = 0 across 600 conv layers, 58M-param max

### Table 4: Distillation ablation (KD: baseline/first/colnorm/full)
- ResNet-20: `cifar_kd_q2000_e10_s10_quantized_random.log`,
  `cifar_kd_q5000_e10_s10_quantized_random.log` (10 seeds each)
- ResNet-56: `cifar_kd_r56_q2000_q5000_e10_s5_quantized_random_jota.log`
  (JOTA run, 5 seeds, output-019d8ef7)

### Table 5: Confidentiality–utility tradeoff
- **ResNet-20 Proxy column**: `11_tradeoff_cifar.log` (N=100 proxy trials)
- **ResNet-20 Per-layer column**: `11b_tradeoff_per_layer_lab614.log`
  (N=20 per-layer trials, lab614 CPU)
- **ResNet-56 Per-layer column**: `11b_per_layer_resnet56_jota.log`
  (N=20 per-layer trials, JOTA GPU, output-019d8f06)
- **Pred. agr. column** (both CIFAR blocks): derived from the per-layer
  traces above (not the proxy traces).
- Note: the ResNet-50 Imagenette block was removed from Table 5 after the
  code audit flagged the old `33_imagenet_tradeoff.log` as using fabricated
  ground truth (mode-of-predictions heuristic instead of real WNID→ImageNet
  labels). The fixed experiment is shipped as
  `jota_artifacts/jota_r50_imagenette_per_layer.zip` and will be run on
  JOTA; partial validation is in `r50_imagenette_smoke_test_lab614.log`.

### Table 6: Scheme-fit coverage
- No log; derived from reading BCD+25, YZL24, LCZ+25, UMS+25, ZCHZ24.

### §4.3 TFHE transcript replay on Concrete v2.11
- `22_tfhe_validate.log`, `22_tfhe_validate.json` — backend sanity check
- `24_tfhe_real.log` — end-to-end run
- `25_tfhe_all21.json` — all 21 conv layers, both permutation regimes
- `25_tfhe_resnet_transcript.log`, `.json` — primary 11,424-query replay
- `25_tfhe_resnet_transcript_doublecheck.log`, `.json` — independent rerun

### §5.1 Fingerprinting and lineage (CIFAR)
- `16_noise_fingerprinting.log` — 190 pair separation on 20 CIFAR checkpoints
- `14_lineage_aggressive.log` — 15/15 CIFAR lineage (mild/moderate/aggressive),
  real SGD fine-tune bases

### §5.1 ResNet-50 V1/V2 fingerprint distance 24.7 (ImageNet)
- `32_imagenet_lineage_DEPRECATED_synthetic_bases.log` —
  the two-model V1/V2 distance is legit (both real pretrained weights);
  the paper no longer cites the lineage count from this log because
  the "perturbed bases" used synthetic Gaussian perturbation rather than
  real fine-tuning.

### §App Bias recovery (MNIST MLP)
- `15_bias_recovery.log` — two-layer MLP with explicit biases; attack
  recovers sort(W+b) exactly. Script was refactored to call
  `attack_layer` and `simulate_query` directly per code audit.

### §3.3 Shuffle amplification context
- `02_dp_analysis.log` — DP parameter sanity check (not cited numerically
  but referenced as background)

## Smoke tests and provenance

- `r50_imagenette_smoke_test_lab614.log` — pre-upload validation of
  `jota_r50_imagenette_per_layer.zip` on lab614 CPU: fp16→fp32 weight
  load OK, 500-image subset traversal OK, 82.20% clean top-1 on real
  labels, conv1 attack error 9.43e-02 at mult=100.
- `imagenette_per_layer_r50_n20_jota.log` + sibling JSON
  `../imagenette_per_layer_r50_n20_jota.json` — JOTA GPU run of the
  same package (wall 71.3s, 019d8f9e). Reproduces the smoke-test clean
  top-1 (82.20%) and attack-error numbers. **Open investigation:**
  mult=100 onward collapses to exactly 0.00 ± 0.00 for both pred
  agreement and top-1 accuracy across all 20 trials. This is sharper
  than the R56 CIFAR collapse (50% at mult=1,000, 10% at mult=5,000)
  and warrants an intermediate-multiplier rerun before it can be
  quoted as a paper-ready phase-transition datapoint.

## JOTA artifact run ids

| Local file prefix | JOTA job id | Input zip |
|---|---|---|
| `cifar_kd_r56_q2000_q5000_e10_s5_quantized_random_jota.log` | 019d8ef7 | `jota_resnet56_kd.zip` |
| `11b_per_layer_resnet56_jota.log` | 019d8f06 | `jota_resnet56_per_layer.zip` |
| `imagenette_per_layer_r50_n20_jota.log` | 019d8f9e | `jota_r50_imagenette_per_layer.zip` |
| `jota_r56_comprehensive.log` | 019d8f1d | `jota_r56_comprehensive.zip` |

## Deprecated logs (do not cite)

- `32_imagenet_lineage_DEPRECATED_synthetic_bases.log` — synthetic
  Gaussian-perturbed R50 bases, not real fine-tuning. The
  only surviving claim from this log is the V1-vs-V2 distance, which
  uses both real pretrained checkpoints and is documented under §5.1
  above.
- `33_imagenet_tradeoff_DEPRECATED_fake_labels.log` — uses
  `folder_to_imagenet[i] = mode(clean_preds_on_folder_i)`, a circular
  ground-truth that measures model self-consistency rather than true
  top-1 accuracy. Replaced by `scripts/33_imagenet_tradeoff.py` (fixed
  to use `IMAGENETTE_TO_IMAGENET = [0, 217, 482, 491, 497, 566, 569,
  571, 574, 701]`). The fixed run is pending on JOTA; see the smoke
  test log above for provisional validation.
