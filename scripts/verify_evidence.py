#!/usr/bin/env python3
"""Verify snapshot integrity and recorded headline counts; never run a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root: Path) -> dict:
    snapshot = root / "snapshot"
    manifest = read_json(snapshot / "MANIFEST.json")
    seen: set[str] = set()
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"Invalid manifest path: {relative}")
        require(str(relative) not in seen, f"Duplicate manifest path: {relative}")
        seen.add(str(relative))
        path = snapshot / relative
        require(path.is_file() and not path.is_symlink(), f"Missing regular file: {relative}")
        data = path.read_bytes()
        require(len(data) == entry["bytes"], f"Size mismatch: {relative}")
        require(hashlib.sha256(data).hexdigest() == entry["sha256"],
                f"Checksum mismatch: {relative}")

    evolution = read_json(snapshot / "reports/010-corner-audit/report.json")
    cases = evolution["cases"]
    generations = sum(case["checked_generations"] for case in cases)
    tokens = sum(case["checked_tokens"] for case in cases)
    require(evolution["status"] == "passed" and len(cases) == 3,
            "Unexpected evolution audit status or case count")
    require(all(case["passed"] and case["checked_generations"] == case["planned_generations"]
                for case in cases), "Incomplete or failed evolution case")
    require((generations, tokens) == (160, 2746), "Evolution totals differ from documented evidence")

    semantic = read_json(snapshot / "reports/004-internal-controller/semantic-v3.json")
    total = len(semantic["cases"])
    passed = sum(case["passed"] is True for case in semantic["cases"])
    failures = [case["name"] for case in semantic["cases"] if not case["passed"]]
    summary = semantic["summary"]
    require((total, passed, len(failures)) == (31, 26, 5), "Semantic totals differ from documented evidence")
    require(semantic["status"] == "failed", "Historical semantic failure status was lost")
    require((summary["total"], summary["passed"], summary["failed"]) == (total, passed, len(failures)),
            "Semantic summary disagrees with individual cases")
    require(summary["failed_cases"] == failures, "Semantic failed-case list disagrees")
    return {
        "snapshot_integrity": "passed",
        "snapshot_payload_files_checked": len(seen),
        "recorded_evolution": {"worlds": len(cases), "generations": generations, "tokens": tokens},
        "recorded_semantics": {"status": "failed", "total": total, "passed": passed,
                               "failed": len(failures), "failed_cases": failures},
        "model_executed": False,
        "scope": "File integrity and consistency of saved results, not a fresh model evaluation.",
    }


if __name__ == "__main__":
    print(json.dumps(verify(ROOT), indent=2))
