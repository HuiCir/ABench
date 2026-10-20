"""Validate a sandbox manifest without starting a workload.

This is a lightweight preflight for environments where the optional
``jsonschema`` package is unavailable.  It checks the security-sensitive
cross-field invariants that a generic JSON Schema cannot express (for
example, MPS smoke must not be scored and checkpoint bytes must be ephemeral).
It never mounts files, invokes a harness, or creates results.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import yaml


HASH = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
KINDS = {"python", "shell", "container", "claude_code_cli", "claude_code_p"}
BACKENDS = {"cpu", "cuda", "mps"}


def _mapping(value: Any, name: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{name} must be a mapping")
        return {}
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_uri_path(uri: Any, manifest_path: Path | None) -> Path | None:
    """Resolve a local file URI for preflight without requiring remote mounts."""
    if not isinstance(uri, str) or not uri.startswith("file://"):
        return None
    raw = uri[7:]
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    candidates = [Path.cwd() / candidate]
    if manifest_path is not None:
        # Try the manifest directory and its parents so a repository-relative
        # file://sandbox/... URI works regardless of the caller's cwd.
        base = manifest_path.resolve().parent
        candidates.extend(parent / candidate for parent in (base, *base.parents))
    for item in candidates:
        if item.is_file():
            return item.resolve()
    return candidates[0]


def validate(doc: dict[str, Any], manifest_path: Path | None = None) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    required = {
        "schema_version", "sandbox_id", "case_id", "run_id", "scoring",
        "harness", "data_assets", "model_assets", "compute", "workspace",
        "storage", "security",
    }
    missing = sorted(required - set(doc))
    errors.extend(f"missing top-level field: {field}" for field in missing)
    if doc.get("schema_version") != "1.0":
        errors.append("schema_version must be '1.0'")

    harness = _mapping(doc.get("harness"), "harness", errors)
    kind = harness.get("kind")
    if kind not in KINDS:
        errors.append(f"harness.kind must be one of {sorted(KINDS)}")
    if not harness.get("entrypoint"):
        errors.append("harness.entrypoint is required")
    if not isinstance(harness.get("timeout_sec"), int) or harness["timeout_sec"] < 1:
        errors.append("harness.timeout_sec must be a positive integer")
    if kind in {"claude_code_cli", "claude_code_p"}:
        if not harness.get("binary") or not harness.get("prompt_file"):
            errors.append("Claude harness requires binary and prompt_file")
        args = harness.get("args", [])
        if "-p" not in args and "--print" not in args:
            errors.append("Claude harness args must explicitly contain -p or --print")
    if kind == "container" and not harness.get("image"):
        errors.append("container harness requires image")

    for field in ("data_assets", "model_assets"):
        assets = doc.get(field)
        if not isinstance(assets, list):
            errors.append(f"{field} must be a list")
            continue
        seen: set[str] = set()
        for i, asset in enumerate(assets):
            a = _mapping(asset, f"{field}[{i}]", errors)
            aid = a.get("id")
            if not isinstance(aid, str) or not aid or aid in seen:
                errors.append(f"{field}[{i}].id must be unique and non-empty")
            seen.add(aid)
            if not isinstance(a.get("mount"), str) or not a["mount"].startswith("/"):
                errors.append(f"{field}[{i}].mount must be an absolute path")
            digest = a.get("sha256")
            if digest != "TO_BE_FROZEN" and (not isinstance(digest, str) or not HASH.fullmatch(digest)):
                errors.append(f"{field}[{i}].sha256 must be a SHA-256 or TO_BE_FROZEN")
            if a.get("read_only") is not True:
                errors.append(f"{field}[{i}] must be read_only=true")
            # If a declared local fixture is available, verify the bytes during
            # preflight. Remote/hidden assets are intentionally left for the
            # execution backend to resolve and are not guessed here.
            local_path = _local_uri_path(a.get("uri"), manifest_path)
            if local_path is not None and local_path.is_file() and digest != "TO_BE_FROZEN":
                actual = _sha256(local_path)
                if actual.lower() != str(digest).lower().removeprefix("sha256:"):
                    errors.append(f"{field}[{i}].sha256 does not match local file {local_path}")
            elif local_path is not None and not local_path.is_file():
                warnings.append(f"{field}[{i}] local URI is not available during preflight: {a.get('uri')}")

    compute = _mapping(doc.get("compute"), "compute", errors)
    backend = compute.get("backend")
    if backend not in BACKENDS:
        errors.append(f"compute.backend must be one of {sorted(BACKENDS)}")
    if backend == "mps" and doc.get("scoring") is True:
        errors.append("MPS runs must set top-level scoring=false")
    limits = _mapping(compute.get("limits"), "compute.limits", errors)
    for key in ("wall_ms", "cpu_threads", "ram_bytes", "output_bytes"):
        if not isinstance(limits.get(key), int) or limits[key] <= 0:
            errors.append(f"compute.limits.{key} must be a positive integer")
    measurement = _mapping(compute.get("measurement"), "compute.measurement", errors)
    if measurement.get("server_side") is not True:
        errors.append("compute.measurement.server_side must be true")
    if measurement.get("synchronize") is not True:
        errors.append("compute.measurement.synchronize must be true")
    if backend == "mps" and measurement.get("record_fallback_ops") is not True:
        errors.append("MPS compute must record fallback ops")
    if backend == "mps" and compute.get("fallback_backend") not in {"cpu", "none"}:
        errors.append("MPS compute must declare fallback_backend cpu or none")

    workspace = _mapping(doc.get("workspace"), "workspace", errors)
    paths = [workspace.get("root"), workspace.get("checkpoint_dir"), workspace.get("submission_dir"), workspace.get("scratch_dir")]
    if any(not isinstance(p, str) or not p.startswith("/") for p in paths):
        errors.append("workspace root/checkpoint/submission/scratch paths must be absolute")
    if isinstance(workspace.get("root"), str):
        root = workspace["root"].rstrip("/")
        for key in ("checkpoint_dir", "submission_dir", "scratch_dir"):
            p = workspace.get(key)
            if isinstance(p, str) and not (p == root or p.startswith(root + "/")):
                errors.append(f"workspace.{key} must be under workspace.root")

    storage = _mapping(doc.get("storage"), "storage", errors)
    policy = _mapping(storage.get("artifact_policy"), "storage.artifact_policy", errors)
    if policy.get("checkpoint_bytes") != "ephemeral":
        warnings.append("checkpoint_bytes is not ephemeral; final checkpoint promotion needs case-level approval")
    if "intermediate_checkpoints" not in policy.get("discard", []):
        errors.append("artifact_policy.discard must include intermediate_checkpoints")
    submit = _mapping(storage.get("submit"), "storage.submit", errors)
    if submit.get("once") is not True:
        errors.append("storage.submit.once must be true")
    if not isinstance(submit.get("required_files"), list) or not submit["required_files"]:
        errors.append("storage.submit.required_files must be non-empty")

    security = _mapping(doc.get("security"), "security", errors)
    if security.get("network") != "disabled":
        errors.append("security.network must be disabled")
    for key in ("hidden_inputs", "host_filesystem", "other_runs"):
        if security.get(key) != "inaccessible":
            errors.append(f"security.{key} must be inaccessible")
    if not isinstance(security.get("process_allowlist"), list):
        errors.append("security.process_allowlist must be a list")
    return errors, warnings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        doc = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR cannot read YAML: {exc}")
        return 1
    if not isinstance(doc, dict):
        print("ERROR manifest must be a mapping")
        return 1
    errors, warnings = validate(doc, args.manifest)
    print(f"manifest={args.manifest} errors={len(errors)} warnings={len(warnings)}")
    for item in errors:
        print("ERROR " + item)
    for item in warnings:
        print("WARNING " + item)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
