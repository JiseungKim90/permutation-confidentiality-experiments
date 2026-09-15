#!/usr/bin/env python3
"""Run the public release's offline syntax, security, and regression checks."""

import compileall
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(relative, *arguments):
    command = [sys.executable, str(ROOT / relative)] + list(arguments)
    completed = subprocess.run(command, cwd=str(ROOT), check=False)
    if completed.returncode:
        raise RuntimeError("command failed (%d): %r" % (completed.returncode, command))
    return command


def main():
    if not compileall.compile_dir(str(ROOT / "lib"), quiet=1):
        raise RuntimeError("library compilation failed")
    if not compileall.compile_dir(str(ROOT / "scripts"), quiet=1):
        raise RuntimeError("script compilation failed")

    commands = [
        run("scripts/verify_release.py"),
        run("scripts/verify_finite_reference.py"),
        run("scripts/test_trusted_inputs.py"),
        run("scripts/test_lta_cover_soundness.py"),
        run("scripts/test_attacker_boundary.py"),
        run("scripts/test_graph_gauge_bias_scope.py"),
    ]
    with tempfile.TemporaryDirectory(prefix="p050-finite-smoke-") as directory:
        output = Path(directory) / "result.json"
        commands.append(run(
            "scripts/identifiability_exhaustive.py",
            "--max-control-width", "2",
            "--output", str(output),
        ))
        with output.open("r", encoding="utf-8") as handle:
            finite = json.load(handle)
        if not finite.get("success"):
            raise RuntimeError("finite smoke result did not meet its success criteria")

    print(json.dumps({
        "success": True,
        "commands": commands,
        "finite_smoke_max_control_width": 2,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
