#!/usr/bin/env python3
"""Regression tests for digest-locked input and checkpoint loading."""

import hashlib
import sys
import tempfile
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.checkpoint import load_verified_checkpoint  # noqa: E402
from lib.trusted_io import read_verified_bytes  # noqa: E402


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def rejected(callable_):
    try:
        callable_()
    except ValueError:
        return True
    return False


def main():
    with tempfile.TemporaryDirectory(prefix="p050-trusted-input-") as directory:
        root = Path(directory)
        raw = root / "input.bin"
        raw.write_bytes(b"authenticated input")
        expected = digest(raw.read_bytes())
        assert read_verified_bytes(raw, expected) == b"authenticated input"
        assert rejected(lambda: read_verified_bytes(raw, "0" * 64))

        checkpoint = root / "checkpoint.pt"
        torch.save({"weight": torch.tensor([1, 2, 3])}, str(checkpoint))
        checkpoint_hash = digest(checkpoint.read_bytes())
        loaded = load_verified_checkpoint(checkpoint, checkpoint_hash)
        assert torch.equal(loaded["weight"], torch.tensor([1, 2, 3]))
        assert rejected(lambda: load_verified_checkpoint(checkpoint, "f" * 64))

        checkpoint.write_bytes(checkpoint.read_bytes() + b"tampered")
        assert rejected(lambda: load_verified_checkpoint(checkpoint, checkpoint_hash))

    print({
        "verified_bytes_accepted": True,
        "wrong_digest_rejected": True,
        "verified_checkpoint_accepted": True,
        "tampered_checkpoint_rejected_before_deserialisation": True,
    })


if __name__ == "__main__":
    main()
