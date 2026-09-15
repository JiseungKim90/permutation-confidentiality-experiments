# TDSC journal extension: transcript identifiability

This branch separates the current journal extension from the [ESORICS version](https://github.com/JiseungKim90/permutation-confidentiality-experiments/tree/main).

**Article:** *When Do Fresh Permutations Hide a Model? Identifiability of Multi-Round Hybrid-FHE Transcripts*

The journal studies when a complete sequence of shuffled replies identifies the observable affine model. The conference result recovered independently sorted column spectra; it did not by itself resolve correspondence between columns or across protocol rounds.

## Included material

- [Finite-model enumeration](tdsc/identifiability_exhaustive.py): the original standalone standard-library program for the nine one-layer families, three two-round families and exact hidden-frame control checks. It operates entirely on explicitly enumerated small affine models, without loading a checkpoint or contacting a service.
- [Observed-port bias regression](tdsc/test_graph_gauge_bias_scope.py): a fixed mathematical example checking that every observed port's bias matters, including ports with no incoming edge.
- [Result checker](tdsc/verify_finite_results.py): compares every deterministic numerical field, witness and success condition with [the reference output](tdsc/finite_reference.json). Missing fields, changed results and nonboolean success values are rejected; machine-specific runtime metadata is not compared.
- [Article result summaries](tdsc/results_summary.json): the finite-model results and the separate ResNet-20 simulator summaries.

The first two scripts are unchanged copies of the retained research sources. The reference contains all deterministic fields of the archived enumeration output; hostnames, process IDs, local paths and runtime timestamps are not duplicated.

## Running the mathematical checks

The enumeration and JSON checker use only the Python standard library. The bias regression also uses NumPy. Publication-time syntax and unit checks passed with Python 3.12.14 and NumPy 2.3.5; the latter is pinned in [tdsc/requirements.txt](tdsc/requirements.txt).

From the repository root:

```bash
python3 -m pip install -r tdsc/requirements.txt
python3 tdsc/identifiability_exhaustive.py --max-control-width 6 --output finite-results.json
python3 tdsc/verify_finite_results.py --report finite-results.json --reference tdsc/finite_reference.json
python3 tdsc/test_graph_gauge_bias_scope.py
```

The historical full enumeration used Python 3.8.10 on Linux. It covered nine one-layer families, three two-round families, 5,460 binary response/target pairs and 197,956 constructive control checks. The publication update checked the retained source, performed bounded syntax/unit checks and compared its archived result with the public article summary. It did not rerun the full enumeration: the experiment host's SSH execution environment was unavailable in this session.

The static result checker accepts the retained reference and rejects altered counts, a nonboolean success field and a missing required section. Passing this checker means a supplied report matches the reference; it is not itself a fresh execution of the enumeration.

## Scope and code availability

The article contains its proofs, experimental settings, tables, limitations and interpretation. There is no separate supplementary paper.

The mathematical verification programs above are public. The journal's automated model-extraction implementation, completion implementation and orchestration scripts are not included in this release. The files inherited from `main` are the earlier conference code and must not be treated as a reproduction of the current TDSC network results. This branch is therefore **not a complete executable reproduction package for the entire journal article**.

The network summaries concern one exact-integer ResNet-20 simulator instance, not extraction from a deployed hybrid-FHE service. Full-set accuracy is descriptive: 64 of the 10,000 images were used for integer-range calibration. The 9,936-image comparison excludes that subset.

The conference instructions remain in [the README on main](https://github.com/JiseungKim90/permutation-confidentiality-experiments/blob/main/README.md). The earlier `extended-version` branch describes a different journal draft and is not the source of the results summarized here.
