"""Raw MNIST loader (no torchvision).  Downloads the four idx files if allowed."""
import gzip
import os
import time
import urllib.request

import numpy as np

BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]


def download(dest, budget_sec=120.0):
    """Try to fetch the four files within the budget.  Returns (ok, info)."""
    os.makedirs(dest, exist_ok=True)
    t0 = time.time()
    info = {"files": {}, "budget_sec": budget_sec}
    for f in FILES:
        path = os.path.join(dest, f)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            info["files"][f] = "already present (%d bytes)" % os.path.getsize(path)
            continue
        left = budget_sec - (time.time() - t0)
        if left <= 1:
            info["files"][f] = "skipped: download budget exhausted"
            info["ok"] = False
            info["elapsed_sec"] = round(time.time() - t0, 2)
            return False, info
        try:
            req = urllib.request.Request(BASE + f, headers={"User-Agent": "python"})
            with urllib.request.urlopen(req, timeout=max(5.0, left)) as r:
                data = r.read()
            with open(path, "wb") as fh:
                fh.write(data)
            info["files"][f] = "downloaded (%d bytes)" % len(data)
        except Exception as exc:
            info["files"][f] = "failed: %s: %s" % (type(exc).__name__, exc)
            info["ok"] = False
            info["elapsed_sec"] = round(time.time() - t0, 2)
            return False, info
    info["ok"] = True
    info["elapsed_sec"] = round(time.time() - t0, 2)
    return True, info


def _read_idx(path):
    with gzip.open(path, "rb") as fh:
        data = fh.read()
    magic = int.from_bytes(data[0:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(ndim)]
    arr = np.frombuffer(data[4 + 4 * ndim:], dtype=np.uint8)
    return arr.reshape(dims)


def load(dest):
    xtr = _read_idx(os.path.join(dest, FILES[0])).astype(np.float32) / 255.0
    ytr = _read_idx(os.path.join(dest, FILES[1])).astype(np.int64)
    xte = _read_idx(os.path.join(dest, FILES[2])).astype(np.float32) / 255.0
    yte = _read_idx(os.path.join(dest, FILES[3])).astype(np.int64)
    return (xtr.reshape(len(xtr), -1), ytr, xte.reshape(len(xte), -1), yte)
