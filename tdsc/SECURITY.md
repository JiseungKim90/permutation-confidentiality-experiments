# Security notes

Do not load an arbitrary model checkpoint or CIFAR pickle with this artifact.
The public entry points accept only the two upstream files whose SHA-256
digests are fixed in the source and README.

The loaders read each input once, compare the complete in-memory byte string
with the trusted digest, and pass only those same authenticated bytes to the
parser. Changing an expected digest to match an untrusted file defeats this
boundary.

The historical claim-producing environment used PyTorch 2.4.1. That version is
retained only as provenance. The installable public environment uses PyTorch
2.10.0 because earlier `weights_only=True` loaders are affected by published
deserialization vulnerabilities:

- <https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6>
- <https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p>

Downloaded checkpoints and datasets are ignored by Git. The release audit also
rejects private keys, token-like strings, personal absolute paths, direct
`torch.load` calls outside the digest-locked loader, and unexpected checkpoint
files in the publication tree.
