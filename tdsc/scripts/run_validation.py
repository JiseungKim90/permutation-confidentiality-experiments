"""Logged validation of ordinary inference and abstract affine mathematics.

This entry point never invokes an oracle, attacker, or recovery experiment.
Use a new output directory; incomplete or failed results are preserved.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TESTS = ("test_input_normalization.py", "test_finite_affine_laws.py",
         "test_affine_graph_semantics.py", "test_inference_evidence.py")


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    files = ["scripts/" + name for name in TESTS]
    files += ["scripts/run_validation.py", "scripts/verify_inference_evidence.py",
              "lib/input_normalization.py", "reference/normal-inference-20261002/result.json",
              "reference/multicheckpoint/checkpoints.json"]
    launch = {
        "schema": "p050-benign-validation-v1", "utc": datetime.now(timezone.utc).isoformat(),
        "working_directory": str(ROOT), "command": [sys.executable] + sys.argv,
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
        "python": platform.python_version(), "seed_schedule": {"numpy": 20261002, "python_random": 20261002},
        "tests": list(TESTS), "success_criteria": "All four tests exit zero and report success=true.",
        "failure_criteria": "Any failed assertion, nonzero exit, invalid JSON, or missing test output.",
        "scope": "Normal inference and finite mathematical regressions only."
    }
    save(out / "launch.json", launch)
    report = {"schema": launch["schema"], "launch": launch, "status": "running", "results": []}
    save(out / "result.json", report)
    try:
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        for name in TESTS:
            command = [sys.executable, str(ROOT / "scripts" / name)]
            run = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
            (out / (name + ".stdout.log")).write_text(run.stdout, encoding="utf-8")
            (out / (name + ".stderr.log")).write_text(run.stderr, encoding="utf-8")
            if run.returncode:
                raise RuntimeError("test failed: " + name + "\n" + run.stderr)
            result = json.loads(run.stdout)
            if result.get("success") is not True:
                raise RuntimeError("test did not meet acceptance conditions: " + name)
            report["results"].append({"test": name, "result": result})
            save(out / "result.json", report)
        report.update(status="complete", success=True)
    except Exception as exc:
        report.update(status="failed", success=False, exception=repr(exc))
        raise
    finally:
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save(out / "result.json", report)
    print(json.dumps({"success": True, "tests_passed": len(report["results"]),
                      "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
