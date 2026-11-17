#!/usr/bin/env python3
"""Run the exported source's selected CPU checks without loading a checkpoint."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "snapshot"
PINS = {
    "qwenputer": "bec1a0c9654ac8151b6f36ed0154a19c53141384",
    "qwenGPU": "e2695f6a0a818ebc68e2e14b6b990eddf04c51b3",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def portable_path(path: Path, external_label: str) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return f"<external-checkout:{external_label}>"


def dependency_info(name: str, path: Path) -> dict:
    if not path.is_dir():
        raise ValueError(f"Missing {name} checkout; clone the pinned dependency first")
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    revision = result.stdout.strip()
    if revision != PINS[name]:
        raise ValueError(f"{name} revision {revision} does not match required {PINS[name]}")
    dirty = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if dirty:
        raise ValueError(f"{name} has tracked changes; use an unmodified pinned checkout")
    return {"path": portable_path(path, name), "revision": revision, "tracked_files_clean": True}


def counts(output: str, framework: str) -> tuple[int | None, int]:
    if framework == "unittest":
        matches = re.findall(r"Ran (\d+) tests? in", output)
        return (int(matches[-1]) if matches else None), 0
    matches = re.findall(r"(?<!\w)(\d+) passed(?=,| in|\s*$)", output, re.MULTILINE)
    subtests = re.findall(r"(\d+) subtests passed", output)
    return (int(matches[-1]) if matches else None), (int(subtests[-1]) if subtests else 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python with Torch, Transformers and pytest installed")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh or empty directory for exact logs and report.json")
    parser.add_argument("--qwenputer", type=Path, default=SNAPSHOT / "references" / "qwenputer")
    parser.add_argument("--qwengpu", type=Path, default=SNAPSHOT / "references" / "qwenGPU")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error("--output-dir must be a fresh or empty directory; previous evidence is not overwritten")
    out.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONPATH": os.pathsep.join(map(str, [SNAPSHOT, args.qwenputer.resolve(), args.qwengpu.resolve()])),
    })
    report = {
        "format": "qwen-life-cpu-checks-v1",
        "started_at_utc": utc_now(),
        "status": "running",
        "scope": "Selected standalone-rule, symbolic-circuit, merge-plan, runtime-unit, and UI/bridge-unit tests",
        "checkpoint_loaded": False,
        "gpu_enabled": False,
        "full_model_reproduction": False,
        "historical_reports_modified": False,
        "environment_controls": {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "groups": [],
    }
    target = out / "report.json"
    write_report(target, report)
    try:
        report["dependencies"] = {
            "qwenputer": dependency_info("qwenputer", args.qwenputer.resolve()),
            "qwenGPU": dependency_info("qwenGPU", args.qwengpu.resolve()),
        }
        # Metadata lookup does not import Torch, initialize CUDA, or load models.
        probe = subprocess.run(
            [args.python, "-B", "-c", "import importlib.metadata as m,json,platform,sys; print(json.dumps({'python':sys.version.split()[0],'platform':platform.system(),'architecture':platform.machine(),'packages':{name:m.version(name) for name in ['torch','transformers','pytest','numpy','safetensors','tokenizers']}}))"],
            env=env, capture_output=True, text=True, check=True,
        )
        report["environment"] = json.loads(probe.stdout)
        groups = [
            ("reference", SNAPSHOT / "reference", "unittest", ["-m", "unittest", "-v"], 8),
            ("circuit-runtime", SNAPSHOT, "pytest", ["-m", "pytest", "-q", "-p", "no:cacheprovider", "prototype/test_internal_circuit.py", "tests/test_internal_merge_plan.py", "tests/test_internal_runtime.py"], 31),
            ("ui-bridge", SNAPSHOT / "ui", "unittest", ["-m", "unittest", "-v"], 13),
        ]
        for name, cwd, framework, command, expected in groups:
            started = time.monotonic()
            completed = subprocess.run(
                [args.python, "-B", *command], cwd=cwd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            # Preserve combined output bytes exactly; counts use a decoded view only.
            log_name = f"{name}.log"
            (out / log_name).write_bytes(completed.stdout)
            passed, subtests = counts(completed.stdout.decode("utf-8", errors="replace"), framework)
            ok = completed.returncode == 0 and passed == expected
            group = {
                "name": name, "cwd": cwd.relative_to(ROOT).as_posix(),
                "command": ["<python>", "-B", *command], "log": log_name,
                "exit_code": completed.returncode, "status": "passed" if ok else "failed",
                "expected_tests": expected, "passed_tests": passed,
                "reported_passed_subtests": subtests,
                "seconds": round(time.monotonic() - started, 3),
            }
            report["groups"].append(group)
            write_report(target, report)
            print(f"{name}: {group['status']} ({passed}/{expected} tests)", flush=True)
        report["passed_tests"] = sum(group["passed_tests"] or 0 for group in report["groups"])
        report["reported_passed_subtests"] = sum(group["reported_passed_subtests"] for group in report["groups"])
        report["status"] = "passed" if all(group["status"] == "passed" for group in report["groups"]) else "failed"
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        report["status"] = "failed"
        # Avoid embedding absolute local paths, command strings, or environment data.
        report["setup_error"] = type(error).__name__
        print(f"CPU check setup failed ({type(error).__name__}); verify the Python environment and pinned dependency checkouts.", file=sys.stderr)
    finally:
        report["finished_at_utc"] = utc_now()
        write_report(target, report)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
