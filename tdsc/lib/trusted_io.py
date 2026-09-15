"""Digest-locked readers for public experiment inputs."""

import hashlib
import hmac
from pathlib import Path


def _validated_digest(expected_sha256):
    value = str(expected_sha256).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
    return value


def read_verified_bytes(path, expected_sha256):
    """Read once, verify once, and return exactly the authenticated bytes."""
    expected = _validated_digest(expected_sha256)
    payload = Path(path).read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(actual, expected):
        raise ValueError(
            "SHA-256 mismatch for %s: got %s, expected %s"
            % (path, actual, expected)
        )
    return payload

