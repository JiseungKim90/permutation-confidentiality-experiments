# GELO MS MARCO 100K scale and open-set study

## Status and canonical location

This run is executed on `ubuntu02` under
`/home/ubuntu/research-vault/projects/P050/ndss-2027/artifact`.
The canonical retained run is
`outputs/gelo_msmarco_100k_dedup_20260802`.
The non-deduplicated sibling directory is exploratory evidence only: it exposed
an item-identity error caused by two MS MARCO passages with identical 32-token
GPT-2 prefixes.

## Pinned inputs

- MS MARCO archive SHA-256:
  `70667529e474322327d6441c8f7b621f2a805a7f6220897d9f80e5a8294bd62e`.
- GPT-2 snapshot:
  `openai-community/gpt2@607a30d783dfa663caf39e06633721c8d4cfcd7e`.
- GELO commit:
  `786668f20936ae794d4e39483f401492e82d257e`.
- Artifact branch:
  `ndss-2027-invariant-obfuscation`.

## Data design

The preparation script scans all 8,841,823 MS MARCO passages and retains the
lowest keyed BLAKE2b-64 priorities. A passage is usable only when GPT-2 yields
at least 32 tokens. Exact 32-token prefixes are deduplicated before splitting.
The retained 120K prefixes are partitioned as follows:

- indices [0,100K): public candidate bank;
- indices [100K,110K): threshold-calibration open set;
- indices [110K,120K): held-out evaluation open set.

Closed-set sources are sampled from the first 10K candidates. This makes the
10K, 50K, and 100K bank results paired: only distractors are added. Each split
contains 50 deterministic four-source trials. Partial trials contain two
in-bank and two held-out sources.

## Model conditions

The attacker always builds the candidate bank from public GPT-2 hidden states.
The public victim uses the same checkpoint. The private victim fine-tunes all
parameters in transformer blocks [0,8) for 100 steps at learning rate 1e-5 on
indices [90K,98K), with held-out validation on [98K,100K). These ranges are
disjoint from source, calibration, and open-set pools. The private manifest
records parameter drift and held-out perplexity.

Hidden states at layers 4, 8, and 12 are retained as resumable FP16 memmaps.
Scoring converts one candidate chunk at a time and uses float64 residuals.
The production evaluator uses the PyTorch/MKL CPU backend after a regression
test establishes numerical agreement with the NumPy reference to `1e-12`.
The candidate cache remains public in both victim conditions.

## Observation conditions

The evaluator includes ideal orthogonal mixing, GELO Gaussian shielding,
condition-50 non-orthogonal mixing, BF16 quantized stress conditions, and
manifold shielding. Orthogonal and non-orthogonal mixing matrices are sampled
by functions imported from the pinned official GELO source. The evaluator
records that source file's SHA-256.

## Metrics and calibration

For every bank size, trial, layer, victim model, and condition, the artifact
records recall at the source count, average precision, exact positive ranks,
the top 20 candidates, candidate and semantic-union metrics, calibrated
threshold precision/recall, candidate false-positive rate, trial-wise false
positive, runtime, row-space rank, condition number, and peak RSS.

A threshold is calibrated independently for every layer, victim model,
condition, and bank size from 20 disjoint calibration trials. The threshold is
the conservative empirical 5% family-wise false-positive quantile of the
minimum candidate score. It is then frozen for closed, partial, and held-out
open-set evaluation.

## Campaigns

The core campaign evaluates layers 4/8/12, public/private victims, and
`ideal`, `gelo_nonorth`, and `manifold_stress`. The robustness campaign
evaluates layer 8, both victims, and `gelo_gaussian`,
`quantized_gaussian`, `quantized_nonorth`, and `manifold`. Together they
cover every declared condition while reserving the full layer sweep for the
three conditions that define the central boundary.

## Completion and derived artifacts

The detached `p050_gelo_full` tmux session runs the core and robustness
campaigns. A separate `p050_gelo_finalize` session waits for it to terminate
and invokes `scripts/74_finalize_gelo_msmarco_campaign.py`. The finalizer only
emits `final/campaign_complete.json` when the core contains exactly 3,060
unique records and robustness contains exactly 1,360 unique records, with the
declared 20 calibration and 50 evaluation trials in every group.

The final directory contains JSON and CSV versions of the 100K performance
table and rank-based precision--recall points. Positive ranks are sufficient
to reconstruct the exact per-trial PR curve at every recall event; open-set
trials have no positive class and are instead summarized by candidate- and
trial-level false-positive rates. The completion manifest binds raw inputs,
summaries, provenance manifests, and derived tables by SHA-256.

## Private-drift and strict open-set follow-up

The `p050_gelo_followup` session waits for the full matrix above and then runs
a paired public-to-private boundary study. It trains checkpoints at steps
0/100/400/1600/3200 along one deterministic early-prefix fine-tuning
trajectory. At every checkpoint it records relative parameter drift,
row-level hidden-state cosine and L2 drift, bidirectional rowspace residuals,
and held-out MS MARCO loss/perplexity on candidate, calibration-open, and
evaluation-open pools.

Every checkpoint is attacked with the same layer-8 trials and public 100K
candidate bank. The threshold uses 100 disjoint calibration trials and the
first order statistic, corresponding to a nominal 1% trial-wise false-positive
level under exchangeability. Evaluation uses 100 closed, 50 partial, and 300
held-out open-set trials. The finalizer requires exactly 550 unique records
per checkpoint and emits Wilson intervals for trial-wise false positives.

This experiment isolates public-bank mismatch; it does not repeat transforms
that the rowspace theorem predicts to be irrelevant. Its conclusion is scoped
to the measured fine-tuning trajectory and is not generalized to arbitrary
proprietary model prefixes.

## Interpretation guardrails

Candidate-ID misses caused by identical 32-token inputs are evaluation errors,
not attack failures; exact-prefix deduplication removes them. Private-prefix
degradation is evidence about public-bank mismatch, not an impossibility
result for an attacker who knows the private checkpoint. Open-set performance
must be reported with calibrated trial-wise and candidate-level false-positive
rates, not recall alone. Manifold shield candidates are reported both as
semantic-source false positives and as members of the full observed candidate
union.

