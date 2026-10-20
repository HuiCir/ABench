#!/usr/bin/env python3
"""Validate and attest the isolation wrapper used for a benchmark agent.

The runner accepts a command prefix because the actual harness command is
appended after it.  That prefix is deliberately constrained here: ``env`` or
an arbitrary shell command is not isolation and must never make a run
scoreable.  This module keeps the policy in ``spec/wrapper_policy.yaml`` and
returns a serialisable attestation for the run records.
"""
from __future__ import annotations

import hashlib
import os
import re
import shlex
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "spec" / "wrapper_policy.yaml"
_DIGEST = re.compile(r"@sha256:([0-9a-fA-F]{64})$")


def _policy() -> dict[str, Any]:
    raw = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != "abench.wrapper/v1":
        raise RuntimeError(f"invalid wrapper policy: {POLICY_PATH}")
    return raw


def _base(raw: str, tokens: list[str]) -> dict[str, Any]:
    return {
        "policy_version": "abench.wrapper/v1",
        "argv_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "argv": tokens,
        "accepted": False,
        "scoreable": False,
    }


def _has_network_none(tokens: list[str]) -> bool:
    return any(value == "none" for _, value in _option_values(tokens, "--network"))


def _option_values(tokens: list[str], name: str) -> list[tuple[int, str]]:
    """Return occurrences of a long option in ``--flag value`` or ``--flag=value`` form."""
    out: list[tuple[int, str]] = []
    for i, tok in enumerate(tokens):
        if tok == name and i + 1 < len(tokens):
            out.append((i, tokens[i + 1]))
        elif tok.startswith(name + "="):
            out.append((i, tok.split("=", 1)[1]))
    return out


def _mount_specs(tokens: list[str]) -> tuple[list[str], str | None]:
    """Extract bind specs and reject alternate mount syntaxes we do not attest."""
    specs: list[str] = []
    for i, tok in enumerate(tokens):
        if tok in ("--mount", "--mount=", "--volumes-from") or tok.startswith("--mount="):
            return [], "--mount/--volumes-from are not allowed; use an explicit -v/--volume bind"
        if tok in ("-v", "--volume"):
            if i + 1 >= len(tokens):
                return [], f"{tok} is missing its bind specification"
            specs.append(tokens[i + 1])
        elif tok.startswith("--volume="):
            specs.append(tok.split("=", 1)[1])
        elif tok.startswith("-v") and not tok.startswith("--"):
            # Docker/Podman also accept compact ``-vSRC:DST:MODE``.
            specs.append(tok[2:])
    return specs, None


def _env_specs(tokens: list[str]) -> tuple[list[str], str | None]:
    values: list[str] = []
    for i, tok in enumerate(tokens):
        if tok in ("--env-file", "--env-file=") or tok.startswith("--env-file="):
            return [], "--env-file is not allowed"
        if tok in ("-e", "--env"):
            if i + 1 >= len(tokens):
                return [], f"{tok} is missing its environment specification"
            values.append(tokens[i + 1])
        elif tok.startswith("--env="):
            values.append(tok.split("=", 1)[1])
        elif tok.startswith("-e") and not tok.startswith("--"):
            # Treat compact ``-eNAME=value`` as an environment option too;
            # otherwise a secret could bypass the ABENCH_* allow-list.
            values.append(tok[2:])
    return values, None


def _source_expr(name: str) -> set[str]:
    return {f"${name}", f"${{{name}}}"}


def _split_mount(spec: str) -> tuple[str, str, set[str]] | None:
    parts = spec.split(":")
    if len(parts) < 2:
        return None
    src, dst = parts[0], parts[1]
    opts = set(filter(None, ",".join(parts[2:]).split(",")))
    return src, dst, opts


def _container_attestation(raw: str, tokens: list[str], entry: dict[str, Any]) -> dict[str, Any]:
    out = _base(raw, tokens)
    launcher = os.path.basename(tokens[0]) if tokens else ""
    out.update({"launcher": launcher, "kind": "container", "wrapper_id": entry["id"]})
    try:
        run_at = tokens.index("run")
    except ValueError:
        out["reason"] = "container wrapper must contain `run`"
        return out
    run_tokens = tokens[run_at + 1 :]
    if "--rm" not in run_tokens or not _has_network_none(run_tokens):
        out["reason"] = "container wrapper requires --rm and --network none"
        return out
    if any(tok == "--privileged" or tok.startswith("--privileged=") for tok in run_tokens):
        out["reason"] = "privileged or host-network containers are not allowed"
        return out
    forbidden_isolation = {
        "--device", "--cap-add", "--security-opt", "--sysctl", "--runtime",
        "--pid=host", "--ipc=host", "--uts=host", "--userns=host", "--cgroupns=host",
    }
    forbidden_prefixes = ("--device=", "--cap-add=", "--security-opt=", "--sysctl=", "--runtime=")
    if any(tok in forbidden_isolation or tok.startswith(forbidden_prefixes) for tok in run_tokens):
        out["reason"] = "host devices, namespaces, extra capabilities, and security overrides are not allowed"
        return out
    for i, tok in enumerate(run_tokens[:-1]):
        if tok in ("--pid", "--ipc", "--uts", "--userns", "--cgroupns") and run_tokens[i + 1] == "host":
            out["reason"] = "host namespaces are not allowed"
            return out
    networks = [value for _, value in _option_values(run_tokens, "--network")]
    if not networks or any(value != "none" for value in networks):
        out["reason"] = "every --network option must be exactly none"
        return out
    # Find the first non-option token after `run`, skipping values of common
    # docker/podman options.  It is the image; the harness command is appended
    # later by sandbox_runner.py.
    value_flags = {"-v", "--volume", "-w", "--workdir", "-e", "--env", "--user", "--name", "--network"}
    image = None
    i = 0
    while i < len(run_tokens):
        tok = run_tokens[i]
        if tok in value_flags:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        image = tok
        break
    if not image:
        out["reason"] = "container wrapper is missing an image"
        return out
    out["image"] = image
    match = _DIGEST.search(image)
    if not match:
        out["reason"] = "container image must be pinned with @sha256:<64 hex>"
        return out
    out["image_digest"] = "sha256:" + match.group(1).lower()
    mounts, mount_error = _mount_specs(run_tokens)
    if mount_error:
        out["reason"] = mount_error
        return out
    env_specs, env_error = _env_specs(run_tokens)
    if env_error:
        out["reason"] = env_error
        return out
    # Do not let a wrapper smuggle an operator secret or an arbitrary host
    # directory into the container.  Every bind mount must originate from a
    # runner-owned ABENCH_* path, and explicit environment forwarding is
    # limited to the ABENCH_* interface.
    for spec in mounts:
        parsed = _split_mount(spec)
        if not parsed:
            out["reason"] = "container bind mounts must use source:destination[:options]"
            return out
        source, destination, _opts = parsed
        allowed_names = (
                "ABENCH_VIEW_DIR", "ABENCH_WORKSPACE", "ABENCH_SUBMISSION_DIR", "ABENCH_DATA_DIR",
                "ABENCH_MODEL_DIR", "ABENCH_CHECKPOINT_DIR", "ABENCH_HOME_DIR", "ABENCH_TMP_DIR")
        if not any(source in _source_expr(name) and destination in _source_expr(name) for name in allowed_names):
            out["reason"] = "container bind mounts must originate from runner-owned ABENCH_* paths"
            return out
    parsed_mounts = [_split_mount(spec) for spec in mounts]
    view_mounts = [m for m in parsed_mounts if m and m[0] in _source_expr("ABENCH_VIEW_DIR")]
    workspace_mounts = [m for m in parsed_mounts if m and m[0] in _source_expr("ABENCH_WORKSPACE")]
    if not view_mounts or any("ro" not in m[2] or "rw" in m[2] for m in view_mounts):
        out["reason"] = "container wrapper must mount ABENCH_VIEW_DIR read-only"
        return out
    # Docker defaults to rw when no mode is supplied; explicitly reject ro and
    # require at least one writable workspace bind.
    if not workspace_mounts or not any("rw" in m[2] and "ro" not in m[2] for m in workspace_mounts):
        out["reason"] = "container wrapper must mount ABENCH_WORKSPACE writable"
        return out
    for spec in env_specs:
        name = spec.split("=", 1)[0]
        if not name.startswith("ABENCH_"):
            out["reason"] = "container environment forwarding is limited to ABENCH_* names"
            return out
    out.update({"accepted": True, "scoreable": True, "reason": "allowlisted_digest_pinned_container"})
    return out


def _sudo_attestation(raw: str, tokens: list[str], entry: dict[str, Any]) -> dict[str, Any]:
    out = _base(raw, tokens)
    out.update({"launcher": os.path.basename(tokens[0]) if tokens else "", "kind": "user", "wrapper_id": entry["id"]})
    user = None
    for i, tok in enumerate(tokens[:-1]):
        if tok in ("-u", "--user"):
            user = tokens[i + 1]
            break
    if user != entry.get("required_user"):
        out["reason"] = "sudo wrapper must switch to the dedicated abench-agent user"
        return out
    if "--" not in tokens or "-E" in tokens or "--preserve-env" in tokens:
        out["reason"] = "sudo wrapper must terminate options with -- and may not preserve the operator environment"
        return out
    marker = tokens.index("--")
    prefix = tokens[:marker]
    if prefix not in ([tokens[0], "-u", user], [tokens[0], "--user", user]):
        out["reason"] = "sudo wrapper may only select the dedicated user before --"
        return out
    out.update({"user": user, "accepted": True, "scoreable": True, "reason": "allowlisted_dedicated_user"})
    return out


def inspect_wrapper(raw: str | None) -> dict[str, Any]:
    """Return a wrapper attestation; never treats an unknown prefix as safe."""
    if not raw:
        return {
            "policy_version": "abench.wrapper/v1", "accepted": False,
            "scoreable": False, "launcher": None, "kind": "host_soft",
            "wrapper_id": "host_soft", "reason": "no wrapper (smoke only)",
        }
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        return {"policy_version": "abench.wrapper/v1", "accepted": False,
                "scoreable": False, "reason": f"invalid wrapper syntax: {exc}"}
    if not tokens:
        return {"policy_version": "abench.wrapper/v1", "accepted": False,
                "scoreable": False, "reason": "empty wrapper"}
    launcher = os.path.basename(tokens[0])
    entries = [e for e in _policy().get("allowed", []) if isinstance(e, dict) and e.get("launcher") == launcher]
    if not entries:
        out = _base(raw, tokens)
        out.update({"launcher": launcher, "kind": "unknown", "wrapper_id": None,
                    "reason": f"launcher {launcher!r} is not in wrapper allow-list"})
        return out
    entry = entries[0]
    if entry.get("kind") == "container":
        return _container_attestation(raw, tokens, entry)
    if entry.get("kind") == "user":
        return _sudo_attestation(raw, tokens, entry)
    out = _base(raw, tokens)
    out["reason"] = f"unsupported wrapper policy kind {entry.get('kind')!r}"
    return out


def require_benchmark_wrapper(raw: str | None) -> dict[str, Any]:
    attestation = inspect_wrapper(raw)
    if not attestation.get("accepted") or not attestation.get("scoreable"):
        reason = attestation.get("reason", "wrapper is not scoreable")
        raise SystemExit(f"benchmark wrapper rejected: {reason}; use an allowlisted isolated wrapper")
    return attestation

