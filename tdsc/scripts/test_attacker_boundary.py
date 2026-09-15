#!/usr/bin/env python3
"""Regression tests for the public attacker/private oracle boundary."""

from __future__ import annotations

import gzip
import multiprocessing as mp
import os
import sys
import tempfile
from pathlib import Path


REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT / "scripts"))
sys.path.insert(0, str(REPRO_ROOT))

from run_extraction import (  # noqa: E402
    audit_attacker_view,
    remove_private_environment_for_spawn,
    restore_private_environment,
    write_canonical_gzip_ndjson,
)


def _spawn_probe(connection, public_config):
    try:
        connection.send({
            "ok": True,
            "audit": audit_attacker_view(public_config, globals()),
        })
    except Exception as exc:
        connection.send({"ok": False, "error": "%s: %s" % (
            type(exc).__name__, exc)})
    finally:
        connection.close()


def main() -> None:
    public_config = {
        "attack_seed": 7,
        "T": 3,
        "public_manifest_sha256": "public-only",
    }
    try:
        audit_attacker_view(
            dict(public_config, oracle_seed=11, expected_model_sha256="truth"),
            globals(),
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("private config keys were not rejected")

    injected = {
        "TDSC_RPC_ORACLE_SEED": "11",
        "TDSC_RPC_EXPECT_MODEL_SHA256": "truth",
        "TDSC_CLASS_ORACLE_SEED": "13",
    }
    old = {key: os.environ.get(key) for key in injected}
    os.environ.update(injected)
    parent, child = mp.get_context("spawn").Pipe(duplex=False)
    process = mp.get_context("spawn").Process(
        target=_spawn_probe, args=(child, public_config))
    saved = remove_private_environment_for_spawn()
    try:
        process.start()
    finally:
        restore_private_environment(saved)
    child.close()
    if not parent.poll(30):
        raise AssertionError("spawned attacker-view probe timed out")
    result = parent.recv()
    process.join(30)
    assert process.exitcode == 0
    assert result["ok"], result
    assert result["audit"]["private_config_keys_present"] == []
    assert result["audit"]["private_environment_keys_present"] == []
    assert result["audit"]["private_imported_globals_present"] == []
    assert all(os.environ.get(key) == value for key, value in injected.items())
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value

    rows = [{"round": 0, "value": 1}, {"round": 1, "value": -2}]
    with tempfile.TemporaryDirectory(prefix="p050-boundary-test-") as directory:
        first = os.path.join(directory, "first.ndjson.gz")
        second = os.path.join(directory, "second.ndjson.gz")
        digest1 = write_canonical_gzip_ndjson(first, rows)
        digest2 = write_canonical_gzip_ndjson(second, rows)
        assert digest1 == digest2
        with open(first, "rb") as left, open(second, "rb") as right:
            assert left.read() == right.read()
        with gzip.open(first, "rt", encoding="utf-8") as handle:
            assert handle.read() == (
                '{"round":0,"value":1}\n'
                '{"round":1,"value":-2}\n')

    print({
        "spawned_attacker_view": result["audit"],
        "private_environment_restored_in_parent": True,
        "canonical_gzip_file_sha256": digest1[0],
        "canonical_content_sha256": digest1[1],
    })


if __name__ == "__main__":
    main()
