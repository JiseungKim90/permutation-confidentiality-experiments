"""Checkpoint loading restricted to bytes authenticated by a caller-supplied digest."""

import io

import torch

from .trusted_io import read_verified_bytes


def load_verified_checkpoint(path, expected_sha256, map_location="cpu"):
    """Hash-lock a checkpoint before passing the same in-memory bytes to PyTorch."""
    payload = read_verified_bytes(path, expected_sha256)
    return torch.load(
        io.BytesIO(payload),
        map_location=map_location,
        weights_only=True,
    )

