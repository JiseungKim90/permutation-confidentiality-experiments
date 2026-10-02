"""Data loading for the QAT item: CIFAR-10 from the pickle archive and
Tiny-ImageNet from the numpy arrays built by scripts/qat_prep_tin.py.

No torchvision and no PIL are available on the host, so the Tiny-ImageNet JPEGs
are decoded once with ImageMagick (`convert`) into uint8 numpy arrays; see
scripts/qat_prep_tin.py.
"""
import os
import pickle

import numpy as np

from qat_common import DATA

CIFAR_DIR = os.path.join(DATA, "cifar10_extract", "cifar-10-batches-py")
TIN_TRAIN_X = os.path.join(DATA, "qat_tin_train_x.npy")
TIN_TRAIN_Y = os.path.join(DATA, "qat_tin_train_y.npy")
TIN_VAL_X = os.path.join(DATA, "qat_tin_val_x.npy")
TIN_VAL_Y = os.path.join(DATA, "qat_tin_val_y.npy")


def load_cifar10():
    """(Xtr uint8 (50000,3,32,32), ytr, Xte, yte)."""
    Xs, ys = [], []
    for i in range(1, 6):
        with open(os.path.join(CIFAR_DIR, "data_batch_%d" % i), "rb") as fh:
            d = pickle.load(fh, encoding="bytes")
        Xs.append(d[b"data"].reshape(-1, 3, 32, 32).astype(np.uint8))
        ys.append(np.asarray(d[b"labels"], dtype=np.int64))
    with open(os.path.join(CIFAR_DIR, "test_batch"), "rb") as fh:
        d = pickle.load(fh, encoding="bytes")
    Xte = d[b"data"].reshape(-1, 3, 32, 32).astype(np.uint8)
    yte = np.asarray(d[b"labels"], dtype=np.int64)
    return np.concatenate(Xs), np.concatenate(ys), Xte, yte


def load_tin(mmap=True):
    mode = "r" if mmap else None
    return (np.load(TIN_TRAIN_X, mmap_mode=mode), np.load(TIN_TRAIN_Y),
            np.load(TIN_VAL_X, mmap_mode=mode), np.load(TIN_VAL_Y))


def augment(batch_u8, pad, rng):
    """Random crop with `pad` pixels of zero padding, plus a horizontal flip.
    batch_u8: (N,3,H,W) uint8.  Returns float32 in [0,1]."""
    n, c, h, w = batch_u8.shape
    out = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=np.uint8)
    out[:, :, pad:pad + h, pad:pad + w] = batch_u8
    dx = rng.integers(0, 2 * pad + 1, size=n)
    dy = rng.integers(0, 2 * pad + 1, size=n)
    res = np.empty_like(batch_u8)
    for i in range(n):
        res[i] = out[i, :, dy[i]:dy[i] + h, dx[i]:dx[i] + w]
    flip = rng.random(n) < 0.5
    res[flip] = res[flip][:, :, :, ::-1]
    return res.astype(np.float32) / 255.0


def plain(batch_u8):
    return np.ascontiguousarray(batch_u8).astype(np.float32) / 255.0
