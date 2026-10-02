# TDSC identifiability and recovery experiments

This directory is the public reproduction package for **When Do Fresh
Permutations Hide a Model? Identifiability of Multi-Round Hybrid-FHE
Transcripts**. It adds the journal extension's code to the ESORICS artifact,
including multi-round identifiability checks, process-isolated ResNet-20
extraction, and the class-aware completion experiment.

This GitHub package is the public artifact referenced by URL. The article does
not depend on a separate supplementary-material submission.

The package contains the source and compact records for the journal's
simulator case study and finite enumeration. The historical QAT inputs are
not distributed, and the conference measurements are archived evidence, not
new journal replications. Bulky raw traces are not in this publication tree.
The limitations below are part of the evidence record.

## Contents

- `scripts/run_extraction.py`: three-process, fail-closed whole-network attack.
- `scripts/run_completion.py`: recovery of the five observable coefficients
  omitted by the initial clone, followed by independent evaluation.
- `scripts/identifiability_exhaustive.py`: finite-domain transcript-fibre and
  frame-control enumeration.
- `scripts/test_*.py`: regressions for line-tracking attack (LTA) uniqueness,
  the attacker/oracle boundary, and the observed-port bias condition.
- `lib/`: the complete dependency tree for the public entry points, plus the
  earlier single-layer and graph experiment modules used to reach them.
- `reference/`: compact canonical outputs, the full finite-enumeration report,
  and recovered arrays.
- `scripts/download_inputs.py`: authenticated-by-digest input downloader.

## Environment

The original reference runs recorded Python 3.8.10, NumPy 1.17.4, and PyTorch
2.4.1+cpu. `requirements-recorded.txt` preserves that record. The October 2
checkpoint audit used NumPy 1.24.4; it is a separate environment record.
PyTorch releases
through 2.9.1 have published `weights_only=True` deserialization
vulnerabilities, so the installable public environment is tested on Python
3.10 and pins NumPy 1.24.4 and PyTorch 2.10.0:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip check
```

`requirements.txt` selects PyTorch's official CPU wheel; the experiments do
not require CUDA.

Set numerical libraries to one thread for the whole-network runs:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## Inputs

No model checkpoint or dataset archive is committed. Fetch both files from the
upstream release and the official CIFAR-10 site, then verify their full SHA-256
digests automatically:

```bash
python3 scripts/download_inputs.py
```

Expected files are:

```text
4118986f0df73003d572b0e397f0ac7b3f60af1f31aff3d2da164536e36f6ec8  data/cifar10_resnet20.pt
6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce  data/cifar-10-python.tar.gz
```

The loaders authenticate the complete in-memory bytes before parsing them.
The checkpoint then uses `weights_only=True`; the CIFAR loader reads only the
named batch directly from the authenticated archive and does not extract it.
Do not change the fixed digests to load an untrusted file. See `SECURITY.md`.

## Fast validation

From this directory, run:

```bash
python3 scripts/run_checks.py
```

This compiles all Python files, audits the release tree and reference hashes,
runs the four regressions, and performs a reduced finite-enumeration smoke
test. It does not download inputs or run the whole-network attack.

For the complete finite enumeration used by the paper:

```bash
python3 scripts/identifiability_exhaustive.py \
  --max-control-width 6 \
  --output results/identifiability/result.json
```

Success requires zero stabilizer/controller disagreements, zero constructive
failures, singleton fibres on the rich grids, and one gauge orbit in every
rich-domain separating two-round transcript class.

## Whole-network extraction

After downloading the inputs, run:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TDSC_RPC_RUN=tdsc-extraction \
  TDSC_RPC_NTEST=10000 TDSC_RPC_BUDGET=1800 \
  TDSC_RPC_EVAL_TIMEOUT=2400 \
  TDSC_RPC_ATTACK_SEED=20260923 TDSC_RPC_ORACLE_SEED=20260924 \
  TDSC_RPC_WBITS=8 TDSC_RPC_T=3 TDSC_RPC_TRACE_ARITHMETIC=1 \
  TDSC_RPC_EXPECT_MODEL_SHA256=b3b291c77186dfbccf406cd3bb361e9dc257788e1886174bc2f3ea6966babcda \
  PYTHONPATH=scripts:. python3 scripts/run_extraction.py
```

The run is successful only if all three processes exit zero; every attacker
boundary, transcript-integrity, admissibility, rounding, and LTA uniqueness
check passes; 29/29 records and 34/34 LTA invocations are certified; and an
independent evaluator obtains 10,000/10,000 identical logit vectors. A clean
canonical run used 3,676 sessions and 73,520 affine evaluations.

## Class-aware completion

The completion consumes `results/provenance/tdsc-extraction/recovered.npz` and
`recovered.json`. Either run extraction first or seed that directory with the
two files in `reference/extraction/`. Then run:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  TDSC_CLASS_RUN=tdsc-completion TDSC_CLASS_SOURCE=tdsc-extraction \
  TDSC_CLASS_ATTACK_SEED=20260923 TDSC_CLASS_ORACLE_SEED=20260925 \
  TDSC_CLASS_WBITS=8 TDSC_CLASS_NTEST=10000 \
  TDSC_CLASS_SEARCH=256 TDSC_CLASS_COVER=2 TDSC_CLASS_SREPEATS=2 \
  TDSC_CLASS_VALUES=1,2,4,8,16,32,64,128,255 \
  TDSC_CLASS_TRACE=1 TDSC_CLASS_SKIP_EVAL=0 \
  PYTHONPATH=scripts:. python3 scripts/run_completion.py
```

Success requires 830 singleton first-convolution slope checks, 2,048 singleton
shortcut-carrier checks, exact verification of all rounded solves, closure of
all four known distinguishers, 29/29 observable affine maps in one coherent
gauge, and 10,000/10,000 identical logit vectors. The completion changes three
first-convolution and two shortcut coefficients in 97 sessions and 234 affine
evaluations.

After both runs complete, verify every paper-facing condition and canonical
output digest with one command:

```bash
python3 scripts/verify_runs.py \
  --extraction-dir results/provenance/tdsc-extraction \
  --completion-dir results/provenance/tdsc-completion
```

## Fixed-configuration multi-checkpoint audit

The `tdsc-multicheckpoint-audit-20261002` branch contains a robustness audit
whose failures are reported in the revised article. The
driver authenticates each checkpoint by SHA-256 and runs the same attack
seed, oracle seeds, quantization, probe count, search limit, control values,
and completion settings for every selected checkpoint. It never retries or
tunes parameters after seeing an outcome:

```bash
python3 scripts/run_multicheckpoint.py \
  --manifest reference/multicheckpoint/checkpoints.json \
  --run-name audit-run \
  --workers 1 \
  --n-test 10000 \
  --budget-sec 1800 \
  --eval-timeout-sec 2400
```

The original release checkpoint passed the full certificate: 29/29 extraction
records, 34/34 line-tracking invocations, 29/29 observable maps after
completion, and 10,000/10,000 identical logit vectors. Two eight-bit QAT
fine-tunes, converted through the simulator's post-training quantizer rather
than executed with their native QAT operators, failed closed under the same
configuration. Seed 0 stopped at
`layer3.1.conv2` with 36 uncertified columns and six unresolved rows. Seed 1
certified all 34 line-tracking invocations but produced only 28/29 accepted
records: at `layer3.2.conv2`, 63/64 channels were resolved, ten rows remained
ambiguous, and the placement probe was still unverified after three repair
attempts. Completion and evaluation were therefore not run for either QAT
checkpoint.

A development-only follow-up changed only the probe count from `T=3` to
`T=5` for QAT seed 0. It again failed closed at `layer3.1.conv2`, with 53
uncertified columns and four unresolved rows. This follow-up does not satisfy
the fixed-configuration gate and is reported only to rule out the simple
explanation that two more probe steps solve the failure.

The compact, path-sanitized record is
`reference/multicheckpoint/audit.json`. The QAT checkpoints share the same
official float initialization and are not independent-from-scratch models.
Their bytes are not distributed in this repository; their digests and expected
relative paths are listed in `reference/multicheckpoint/checkpoints.json`.
Consequently, this branch publishes all audit code and the complete compact
outcome record, but it is not a self-contained distribution of those
historical QAT inputs. The paper therefore retains the original experiment as
a one-checkpoint case study and makes no cross-checkpoint success-rate claim.

## Scope of the evidence

The finite enumeration checks small instances; it does not replace the proofs.
The whole-network runs use an exact-integer R3 simulator and are not a deployed
Safhire/A2Q+ exploit. The four completion probes close four known
distinguishers; the 29-map certificate is the structural check for this
simulator. Arithmetic traces report completed affine outputs, not internal
multiply-accumulate widths.

The October 2 checkpoint regression disabled arithmetic tracing. It reproduces
the recorded recovery outcomes but does not independently recheck the older
trace-derived range bounds. The range numbers in the article come from the
archived verification records, not from new traces in that regression.

The simulator's uniform stem-bias fold is not an exact rewrite of input
normalization followed by zero-padded convolution. At an image boundary it
subtracts contributions for kernel positions outside the image. This is a
confirmed preprocessing discrepancy, not evidence that the original trained
network was faithfully reproduced. Equality between the simulator and a clone
of that same simulator does not resolve the discrepancy. The code is left
unchanged to preserve the meaning of the recorded experiment; a corrected
model would need a new, separately validated evidence record.

The theorems concern decoded transcripts with the specified frame law, not
raw ciphertexts or decryption residuals. The open-domain characterization
does not guarantee efficient recovery on a finite eight-bit message domain.
The published Safhire description supports the chosen-message threat model,
but a complete residual session implementation, native A2Q+ response map,
sampling law, and full abort behavior have not been matched to this simulator.
No deployed-system extraction is established by this package.

## October 2 evidence and preprocessing review

`reference/review-20261002/evidence.json` records the read-only provenance
audit, the freshly repeated finite enumeration, and the ordinary-forward
preprocessing check. Record hashes and the current source hashes match the
published checkpoint audit. This is an integrity check of existing runs,
not a new execution of the extraction experiment.

The finite enumeration again has zero condition disagreements and zero
controller failures, with 5,460 response/target pairs and 197,956 constructive
permutation checks. These finite checks supplement, rather than prove, the
symbolic theorems.

The ordinary-forward check uses the authenticated baseline checkpoint and
three deterministic inputs (zero, constant one-half, and a ramp). All 16 stem
channels have boundary differences; interior differences are below 1e-10.
It checks float64 outputs before activation and quantization and does not
measure any change in prediction accuracy. To repeat this benign check from
this directory:

```bash
python3 scripts/audit_preprocessing.py .
```

The review did not change the model constructor or recovery implementation.
The boundary discrepancy, failed additional checkpoints, and missing
deployment correspondence remain unresolved scientific limitations.

## Upstream inputs and citation

The ResNet-20 checkpoint is the official `cifar10_resnet20-4118986f.pt` asset
from [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models),
which uses the BSD 3-Clause license. The CIFAR-10 archive is downloaded from
the [official dataset page](https://www.cs.toronto.edu/~kriz/cifar.html).
See `THIRD_PARTY_NOTICES.md` and `CITATION.cff`.

The code in this directory is released under the BSD 3-Clause License; see
`LICENSE`.
