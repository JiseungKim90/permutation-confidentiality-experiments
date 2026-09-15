# TDSC journal extension: transcript identifiability

This branch separates the current journal extension from the [ESORICS version](https://github.com/JiseungKim90/permutation-confidentiality-experiments/tree/main).

**Article:** *When Do Fresh Permutations Hide a Model? Identifiability of Multi-Round Hybrid-FHE Transcripts*

The journal article studies when a complete sequence of shuffled replies identifies the observable affine model. The conference result recovered independently sorted column spectra; it did not by itself resolve correspondence between columns or across protocol rounds.

## Material on this branch

[tdsc/results_summary.json](tdsc/results_summary.json) records the numerical summaries retained in the article: complete finite-model enumeration, the exact hidden-frame control check, and the distinction between ordinary prediction agreement and agreement of all observable affine maps.

The article contains its proofs, experimental settings, tables, limitations, and interpretation. There is no separate supplementary paper. Hash lists, execution diaries, and local-machine paths are not part of the manuscript.

The summaries are transcribed from the existing verified research record. This documentation update did not rerun the experiments. The network result concerns one exact-integer ResNet-20 simulator instance, not extraction from a deployed hybrid-FHE service. Accuracy on the full 10,000-image set is descriptive: 64 of those images were used for integer-range calibration. The 9,936-image comparison excludes that subset.

## Code availability

The files inherited from `main` are the earlier conference code; they must not be treated as a reproduction of the current TDSC results. This branch currently adds non-executable summaries and explanatory documentation, not the journal's automated model-extraction implementation or its launch scripts. It is therefore not a complete executable TDSC reproduction package.

The conference instructions remain available in [the README on main](https://github.com/JiseungKim90/permutation-confidentiality-experiments/blob/main/README.md). The earlier `extended-version` branch describes a different journal draft and is not the source of the results summarized here.
