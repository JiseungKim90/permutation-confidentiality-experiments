"""Reproduce ordinary inference using only publicly downloadable inputs.

This workflow never starts an oracle, attacker, or model-recovery procedure.
Use the environment in requirements.txt and a new output directory.
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


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    files = ["requirements.txt", "scripts/reproduce_public_inference.py",
             "scripts/download_inputs.py", "scripts/run_validation.py",
             "scripts/verify_normal_inference.py", "scripts/verify_inference_evidence.py",
             "scripts/verify_release.py"]
    launch = {
        "schema": "p050-public-ordinary-inference-v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "working_directory": str(ROOT), "command": [sys.executable] + sys.argv,
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_status": subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
        "python": platform.python_version(), "checkpoint_schedule": ["official"],
        "n_test": 10000, "threads": args.threads, "batch_size": 128,
        "seeds": "None; authenticated release weights and fixed CIFAR-10 test order.",
        "success_criteria": "All stages exit zero; all four benign suites pass; both public inputs pass fixed digests; all 10000 corrected logit vectors and predictions agree; independent raw-array and complete-source checks pass.",
        "failure_criteria": "A failed or missing stage, input, sample, source, or acceptance condition.",
        "scope": "Ordinary float64 inference only; no extraction, native QAT, or encrypted backend.",
    }
    save(out / "launch.json", launch)
    report = {"schema": launch["schema"], "launch": launch, "status": "running", "stages": []}
    save(out / "result.json", report)
    environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
                       OPENBLAS_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))

    def stage(name, arguments, expect_json=False):
        command = [sys.executable] + arguments
        item = {"name": name, "command": command, "status": "running",
                "stdout": name + ".stdout.log", "stderr": name + ".stderr.log"}
        report["stages"].append(item)
        save(out / "result.json", report)
        print(json.dumps({"stage": name, "status": "running"}), flush=True)
        with (out / item["stdout"]).open("w", encoding="utf-8") as stdout, \
                (out / item["stderr"]).open("w", encoding="utf-8") as stderr:
            process = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
        item["returncode"] = process.returncode
        item["status"] = "complete" if process.returncode == 0 else "failed"
        if process.returncode:
            raise RuntimeError("stage failed: " + name + "; inspect captured stdout and stderr")
        if expect_json:
            item["result"] = json.loads((out / item["stdout"]).read_text(encoding="utf-8"))
            if item["result"].get("success") is not True:
                item["status"] = "failed"
                raise RuntimeError("stage did not report success: " + name)
        save(out / "result.json", report)
        print(json.dumps({"stage": name, "status": "complete"}), flush=True)

    try:
        stage("dependencies", ["-m", "pip", "check"])
        stage("release", ["scripts/verify_release.py"], True)
        stage("validation", ["scripts/run_validation.py", "--out", str(out / "validation")], True)
        stage("download", ["scripts/download_inputs.py"], True)
        normal = out / "normal-inference"
        stage("inference", ["scripts/verify_normal_inference.py", "--root", str(ROOT),
                            "--out", str(normal), "--checkpoints", "official",
                            "--n-test", "10000", "--batch-size", "128", "--threads", str(args.threads)])
        stage("independent-verification", ["scripts/verify_inference_evidence.py",
              "--report", str(normal / "result.json"), "--raw-dir", str(normal),
              "--check-source", "--checkpoints", "official"], True)
        report.update(status="complete", success=True)
    except Exception as exc:
        report.update(status="failed", success=False, exception=repr(exc))
        raise
    finally:
        report["ended_utc"] = datetime.now(timezone.utc).isoformat()
        save(out / "result.json", report)
    print(json.dumps({"success": True, "stages_completed": len(report["stages"]),
                      "report": str(out / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
