# Ordinary-inference and algebra validation

These records concern ordinary float64 inference and finite algebra tests.
They do not certify extraction of the trained network, native QAT execution,
or correspondence with an encrypted backend.

- `normal-inference.json`: all three authenticated weight sets, 10,000 test
  images each, in the public Python 3.10.20 / NumPy 1.24.4 / PyTorch 2.10.0+cpu
  environment. All numerical records and raw-array digests match the earlier
  three-checkpoint ordinary-inference run.
- `independent-check.json`: separate recomputation of every reported error and
  count from the raw arrays, plus verification of nine source hashes. Its
  `report_sha256` identifies the raw execution-host result; that same digest
  is retained in the published result's provenance.
- `semantic-validation.json`: four successful suites, including synthetic
  three-coordinate graph actions and eighteen evidence-rejection controls.
- `semantic-failed-attempt.json`: the preceding validation attempt. Three
  suites passed, but a syntax error in a dictionary-key mutation prevented
  the fourth suite from starting. It was corrected before the complete rerun.
  The failed attempt is not counted as successful validation.

Only execution-host paths were normalized for publication. Raw arrays and
complete console logs remain on the execution host. The launch records fix
commands, source digests, seeds, environment, output locations and acceptance
criteria. Source digests, not the parent commit alone, identify new files that
were tested before being committed.

The manuscript uses the same normal-inference numbers as before and adds the
independent three-coordinate algebra check. Existing failed recovery runs
remain in `../multicheckpoint/audit.json`; none is superseded by these checks.
