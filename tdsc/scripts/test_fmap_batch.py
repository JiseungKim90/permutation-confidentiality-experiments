#!/usr/bin/env python3
"""Exactness regressions for the batched integer feature-map evaluator."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT))

from lib.fmap import (conv_int, conv_int_batch, forward_fmap,
                      forward_fmap_batched)  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(20261002)
    for n, channels, height, width, outputs, stride, kernel in [
            (3, 2, 5, 7, 4, 1, 3),
            (2, 3, 6, 5, 5, 2, 3),
            (4, 4, 3, 4, 3, 1, 1)]:
        x = rng.integers(-7, 8, size=(n, channels, height, width),
                         dtype=np.int64)
        W = rng.integers(
            -5, 6, size=(outputs, channels * kernel * kernel),
            dtype=np.int64)
        b = rng.integers(-11, 12, size=outputs, dtype=np.int64)
        reference = np.stack(
            [conv_int(row, W, b, stride, kernel) for row in x])
        batched = conv_int_batch(x, W, b, stride, kernel)
        assert np.array_equal(reference, batched)

    A = 255
    C = 4
    stem = {
        "W": rng.integers(-3, 4, size=(C, 3 * 9), dtype=np.int64),
        "b": rng.integers(-20, 21, size=C, dtype=np.int64),
        "eta": 7.0,
    }
    block = {
        "name": "layer1.0",
        "W1": rng.integers(-3, 4, size=(C, C * 9), dtype=np.int64),
        "b1": rng.integers(-20, 21, size=C, dtype=np.int64),
        "eta1": 11.0,
        "W2": rng.integers(-3, 4, size=(C, C * 9), dtype=np.int64),
        "b2": rng.integers(-20, 21, size=C, dtype=np.int64),
        "S": rng.integers(-2, 3, size=(C, C), dtype=np.int64),
        "bs": rng.integers(-10, 11, size=C, dtype=np.int64),
        "eta": 13.0,
        "stride": 1,
    }
    fc = {
        "W": rng.integers(-4, 5, size=(3, C), dtype=np.int64),
        "b": rng.integers(-10, 11, size=3, dtype=np.int64),
    }
    net = SimpleNamespace(A=A, stem=stem, blocks=[block], fc=fc)
    images = rng.random((7, 3, 8, 8))
    reference = forward_fmap(net, images)
    for batch in (1, 2, 4, 16):
        assert np.array_equal(
            reference, forward_fmap_batched(net, images, batch=batch))

    print({"convolution_cases": 3, "forward_images": len(images),
           "batch_sizes": [1, 2, 4, 16], "success": True})


if __name__ == "__main__":
    main()
