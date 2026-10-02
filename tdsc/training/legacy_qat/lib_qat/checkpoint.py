"""Digest-locked, data-only loading for the archived QAT training code."""

import hashlib
import io
import re
from pathlib import Path

import torch


def load_verified_checkpoint(path, expected_sha256, map_location="cpu"):
    """Authenticate checkpoint bytes before restricted PyTorch deserialisation."""
    expected = str(expected_sha256).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("expected_sha256 must be a 64-character lowercase digest")
    payload = Path(path).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(
            "checkpoint SHA-256 is %s, expected %s" % (actual, expected))
    return torch.load(
        io.BytesIO(payload), map_location=map_location, weights_only=True)
