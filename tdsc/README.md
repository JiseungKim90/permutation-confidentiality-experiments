# TDSC identifiability and recovery experiments

This directory is the public reproduction package for **When Do Fresh
Permutations Hide a Model? Identifiability of Multi-Round Hybrid-FHE
Transcripts**. It supplements the ESORICS artifact with the journal extension's
multi-round identifiability checks, process-isolated ResNet-20 extraction, and
class-aware completion experiment.

The package contains the complete source required to reproduce every
code-backed result retained in the journal article. Obsolete exploratory
entry points and bulky raw traces are not part of the publication tree.
Compact reference outputs retain the checks needed to audit every numerical
statement reported by the paper.

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

The claim-producing runs used Python 3.8.10, NumPy 1.17.4, and PyTorch
2.4.1+cpu. `requirements-recorded.txt` preserves that record. PyTorch releases
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

## Scope of the evidence

The finite enumeration checks small instances; it does not replace the proofs.
The whole-network runs use an exact-integer R3 simulator and are not a deployed
Safhire/A2Q+ exploit. The four completion probes close four known
distinguishers; the 29-map certificate is the structural check for this
simulator. Arithmetic traces report completed affine outputs, not internal
multiply-accumulate widths.

## Upstream inputs and citation

The ResNet-20 checkpoint is the official `cifar10_resnet20-4118986f.pt` asset
from [chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models),
which uses the BSD 3-Clause license. The CIFAR-10 archive is downloaded from
the [official dataset page](https://www.cs.toronto.edu/~kriz/cifar.html).
See `THIRD_PARTY_NOTICES.md` and `CITATION.cff`.

The code in this directory is released under the BSD 3-Clause License; see
`LICENSE`.
