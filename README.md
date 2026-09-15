# TDSC journal extension: transcript identifiability

This release branch contains the complete public code and compact reference
evidence for the journal extension of the
[ESORICS artifact](https://github.com/JiseungKim90/permutation-confidentiality-experiments/tree/main).

**Article:** *When Do Fresh Permutations Hide a Model? Identifiability of
Multi-Round Hybrid-FHE Transcripts*

The conference result recovered independently sorted column spectra. The
journal extension asks when complete multi-round transcripts determine one
observable affine model, develops the corresponding frame-consistent recovery,
and separates transcript agreement from prediction agreement.

## Journal reproduction package

Start with [tdsc/README.md](tdsc/README.md). The `tdsc/` directory includes:

- the complete current extraction and class-aware completion implementations;
- the full library dependency tree and four fail-closed/security regressions;
- the finite-domain transcript-fibre and controller enumeration;
- verified reference arrays and compact paper-facing result records;
- input download and digest verification;
- one-command offline and post-run validators; and
- BSD 3-Clause licensing and citation metadata for the journal code.

Check the publication tree without downloading the model or dataset:

```bash
python3 -m pip install -r tdsc/requirements.txt
python3 -m pip check
cd tdsc
python3 scripts/run_checks.py
```

The release manifest covers every file in `tdsc/`, and the GitHub workflow runs
the same offline checks on changes to that directory.

## Validation status

The historical claim records used Python 3.8.10, NumPy 1.17.4, and PyTorch
2.4.1+cpu. The complete public candidate was independently re-executed on
Linux with Python 3.10.21, NumPy 1.24.4, and PyTorch 2.10.0+cpu; its canonical
arrays and arithmetic traces were byte-identical to the reference outputs.

- Full finite enumeration: 9 one-layer cases, 3 two-round cases, 5,460
  response/target pairs, 197,956 controller checks, 0 condition mismatches, and
  0 controller failures.
- Initial extraction: 29/29 records and 34/34 LTA invocations certified; 3,676
  sessions; 73,520 affine evaluations; 10,000/10,000 identical logits.
- Class-aware completion: 830 first-convolution and 2,048 shortcut singleton
  checks; 5 observable coefficients recovered in 97 sessions and 234 affine
  evaluations; all four known distinguishers closed; 29/29 observable affine
  maps verified; 10,000/10,000 identical logits.
- Public-boundary and release audits: no private state in the attacker view, no
  forbidden data access, and no manifest or reference-hash failures. A clean
  `pip install -r tdsc/requirements.txt` selected the CPU-only wheel and
  passed `pip check`.

The exact commands, success criteria, limitations, and input hashes are in the
journal package README. The digest-locked input boundary and dependency
rationale are documented in [tdsc/SECURITY.md](tdsc/SECURITY.md).

## Scope

The network experiment uses an exact-integer R3 ResNet-20 simulator. It is not a
claim of extraction from a deployed Safhire/A2Q+ service. The finite enumeration
checks small instances and does not replace the article's proofs. Arithmetic
traces describe completed affine outputs rather than deployed-backend internal
multiply-accumulate widths.

The conference instructions and code remain on
[`main`](https://github.com/JiseungKim90/permutation-confidentiality-experiments/tree/main).
