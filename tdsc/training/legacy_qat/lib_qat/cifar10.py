"""CIFAR-10 loader for the python pickle archive (no torchvision)."""
import os
import pickle
import tarfile

import numpy as np

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float64)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float64)


def load(tar_path, extract_dir=None, want=("test_batch",)):
    """Returns {name: (images (N,3,32,32) float in [0,1], labels (N,))}."""
    if extract_dir is None:
        extract_dir = os.path.join(os.path.dirname(tar_path), "cifar10_extract")
    root = os.path.join(extract_dir, "cifar-10-batches-py")
    if not os.path.isdir(root):
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(extract_dir)
    out = {}
    for name in want:
        with open(os.path.join(root, name), "rb") as fh:
            d = pickle.load(fh, encoding="bytes")
        X = d[b"data"].astype(np.float64).reshape(-1, 3, 32, 32) / 255.0
        y = np.asarray(d[b"labels"], dtype=np.int64)
        out[name] = (X, y)
    return out
