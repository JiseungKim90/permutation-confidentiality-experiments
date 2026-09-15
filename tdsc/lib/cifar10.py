"""CIFAR-10 loader for the python pickle archive (no torchvision)."""
import io
import pickle
import tarfile

import numpy as np

from .trusted_io import read_verified_bytes


MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float64)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float64)
CIFAR10_SHA256 = "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce"


def load(tar_path, extract_dir=None, want=("test_batch",),
         expected_sha256=CIFAR10_SHA256):
    """Return named batches after authenticating and reading the archive in memory."""
    del extract_dir
    payload = read_verified_bytes(tar_path, expected_sha256)
    out = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for name in want:
            member_name = "cifar-10-batches-py/" + name
            member = archive.getmember(member_name)
            if not member.isfile() or member.name != member_name:
                raise ValueError("unexpected CIFAR archive member: %r" % member.name)
            fh = archive.extractfile(member)
            if fh is None:
                raise ValueError("cannot read CIFAR archive member: %r" % member.name)
            d = pickle.load(fh, encoding="bytes")
            X = d[b"data"].astype(np.float64).reshape(-1, 3, 32, 32) / 255.0
            y = np.asarray(d[b"labels"], dtype=np.int64)
            out[name] = (X, y)
    return out
