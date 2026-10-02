# QAT checkpoint provenance

This directory publishes the training code and run records for the eight
ResNet-20 checkpoints in `../../data/checkpoints/`.  They are LSQ-style
quantization-aware fine-tunes of the same public CIFAR-10 ResNet-20
initialization; they are not A2Q+ checkpoints and are not independent
from-scratch models.

## Contents

- `scripts/qat_train.py`, `qat_common.py`, and `qat_data.py` are the training
  entry point and its QAT/data implementation.
- `lib_qat/` is the fixed dependency snapshot used by those scripts.
- `results/` contains the complete JSON record emitted by each of the eight
  training runs.

The source was copied from the historical experiment repository at commit
`4d8f7765e5474f9da890931084c4bea42e7a23d5`.  That code identifies `lib_qat`
as a snapshot of its earlier commit `22b2843`.  For anonymous publication,
only the literal host name in `lib_qat/models.py` and the host/absolute-path
metadata in the JSON records were replaced; the training algorithm, options,
measurements, and checkpoint bytes were not changed.

## Recorded runs

| bits (weight/activation/accumulator) | seed | steps | epochs | CIFAR-10 top-1 |
|---|---:|---:|---:|---:|
| 5/5/16 | 0 | 1,178 | 3.0157 | 89.37% |
| 5/5/16 | 1 | 1,172 | 3.0003 | 89.69% |
| 6/6/18 | 0 | 1,166 | 2.9850 | 90.40% |
| 6/6/18 | 1 | 1,166 | 2.9850 | 90.48% |
| 7/7/20 | 0 | 1,165 | 2.9824 | 91.68% |
| 7/7/20 | 1 | 1,169 | 2.9926 | 91.47% |
| 8/8/22 | 0 | 1,387 | 3.5507 | 92.00% |
| 8/8/22 | 1 | 1,393 | 3.5661 | 91.52% |

Each run used a 25-minute wall-clock cosine schedule.  The JSON files retain
the full optimizer options, layer occupancy records, software versions, and
stopping condition.

## Reproduction

Use Python 3.10 with the recorded NumPy 1.26.4 and PyTorch 2.12.0 environment.
Place the authenticated public initialization at
`data/cifar10_resnet20.pt` under this directory and extract the authenticated
CIFAR-10 archive to `data/cifar10_extract/cifar-10-batches-py/`.  From this
directory, one recorded configuration is:

```bash
OMP_NUM_THREADS=1 python3 scripts/qat_train.py \
  --arch resnet20 --mode qat --seed 0 --minutes 25 \
  --w-bits 8 --a-bits 8 --acc-bits 22 \
  --tag qatf_r20_w8_s0
```

Change the seed and the three bit widths according to the table.  A run writes
`data/qat_<tag>.pt` and `results/qat_train_<tag>.json`.  Verify the checkpoint
against `../../reference/multicheckpoint/checkpoints.json`; the public audit
rejects any byte sequence with a different SHA-256 digest.

The journal robustness audit deliberately converts every checkpoint through
the same post-training 8-bit simulator.  Thus the 5--7-bit entries test weight
set variation under a fixed protocol configuration; they are not native-QAT
backend executions.
