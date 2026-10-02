# Third-party inputs

## PyTorch CIFAR Models

The downloader obtains the pretrained `cifar10_resnet20-4118986f.pt` checkpoint
from <https://github.com/chenyaofo/pytorch-cifar-models>. The checkpoint is not
stored in this repository.

That upstream project is distributed under the BSD 3-Clause License, copyright
(c) 2021, chenyaofo. The license text is available at
<https://github.com/chenyaofo/pytorch-cifar-models/blob/master/LICENSE>.

## CIFAR-10

The downloader obtains `cifar-10-python.tar.gz` from the official CIFAR site at
<https://www.cs.toronto.edu/~kriz/cifar.html>. The archive is not stored in this
repository.

## Published QAT checkpoints

The eight files under `data/checkpoints/` are author-produced fine-tunes of the
PyTorch CIFAR Models ResNet-20 checkpoint on CIFAR-10. They are distributed as
research outputs under this repository's license. Their dependency on the
upstream initialization and dataset, training procedure, run metadata, and
complete SHA-256 digests are recorded in `training/legacy_qat/` and
`reference/multicheckpoint/checkpoints.json`.
