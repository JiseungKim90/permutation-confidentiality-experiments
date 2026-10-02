# Public-input ordinary-inference reproduction

This run started from a fresh HTTPS clone of the public repository at
`21465733af0f55ac565f0d093f87e736e6601a8e`, with a clean working tree and a
new Python 3.10.20 environment installed from `tdsc/requirements.txt`.
Both the official release checkpoint and CIFAR-10 archive were downloaded
from the documented public URLs. No historical QAT file was copied into
the checkout. Dependency wheels may use the package manager's cache.

From `tdsc/`, after installing the declared environment:

```bash
python3 scripts/reproduce_public_inference.py --out results/public-only-new --threads 2
```

The six stages (dependencies, release, four-suite validation, download,
ordinary inference, independent verification) all completed successfully.
All 10,000 corrected logit vectors differ from the explicitly normalized
reference by at most 3.197442310920451e-14; all predictions agree.
The uniform-bias control changes 113 predictions. Raw logit-array SHA-256
`b8568b81191d9f8f6034ce0737860ba05cfd24091d6c9026398bb925f8aeddfc`
matches the earlier public-checkpoint result.

The independent verifier recomputed every count and maximum error from the
raw arrays and checked all nine required ordinary-forward source files.
The evidence suite rejected all 27 negative controls, including empty or
incomplete source manifests and empty checkpoint schedules.

- `workflow.json`: launch record, stage commands and exit codes, download digests, independent verification.
- `normal-inference.json`: full numerical result and source/input hashes.
- `validation.json`: four benign regression results and their source hashes.
- `inference.stdout.log`: actual captured ordinary-forward output.
- `requirements-frozen.txt`: installed dependency versions.

Only execution-root paths were sanitized in the published JSON files.
Each retains the SHA-256 of its original report; the independent verifier's
report digest refers to that original, not the path-sanitized publication.
Raw arrays, input bytes, full stage logs, and the environment remain in the
server run tree. The original inference stdout SHA-256 is
`26c7f8fd90f8aa6cd6ba0bd1e74fb68f3ebf68a9e6f331895250824e02c2d492`.

This record supports ordinary float64 inference for the public checkpoint.
It does not run or certify extraction, native QAT, or an encrypted backend.
The historical failed recovery attempts and missing QAT distribution remain
separate evidence; this successful run does not supersede them.
