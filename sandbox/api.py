"""Small, dependency-free adapter interfaces for the ABench sandbox.

The concrete local/MPS, Linux/CUDA, and remote implementations may differ;
case code should depend only on these protocols.  This module intentionally
does not start processes, mount files, or retain artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class AssetRef:
    id: str
    mount: Path
    sha256: str
    uri: str
    read_only: bool = True


@dataclass(frozen=True)
class ComputeLimits:
    wall_ms: int
    cpu_threads: int
    ram_bytes: int
    accelerator_mem_bytes: int | None
    output_bytes: int


@dataclass(frozen=True)
class ComputeLease:
    backend: str
    device: str
    limits: ComputeLimits
    enforcement: str


@dataclass(frozen=True)
class Telemetry:
    wall_ms: float
    cpu_ms: float
    peak_ram_bytes: int
    peak_accelerator_bytes: int | None
    fallback_ops: tuple[str, ...] = ()
    fallback_tracking: str = "unavailable"


@dataclass(frozen=True)
class SubmittedArtifact:
    path: str
    kind: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class SubmissionReceipt:
    submission_sha256: str
    artifacts: tuple[SubmittedArtifact, ...]


class DataStore(Protocol):
    """Resolve only data assets named by the sandbox manifest."""

    def resolve(self, asset: AssetRef) -> Path:
        """Return the read-only mounted path after hash verification."""

    def open_split(self, asset_id: str, split: str) -> Iterable[Any]:
        """Open a declared split; hidden assets are never exposed to agents."""


class ModelStore(Protocol):
    """Resolve immutable base/teacher/tokenizer assets."""

    def resolve(self, asset: AssetRef) -> Path:
        """Return a read-only model path after hash verification."""

    def materialize(self, asset_id: str, destination: Path) -> Path:
        """Materialize a verified read-only view; never mutate the source."""


class ComputeProvider(Protocol):
    def request(self, backend: str, device: str, limits: ComputeLimits) -> ComputeLease:
        """Reserve the declared device and reject unsupported limits."""

    def telemetry(self, lease: ComputeLease) -> Telemetry:
        """Return server-side measurements, including MPS fallback events."""


class ArtifactStore(Protocol):
    def stage(self, run_root: Path) -> Path:
        """Create an empty submission staging directory."""

    def collect(self, staging: Path, allowlist: Sequence[str]) -> SubmissionReceipt:
        """Validate, hash, and collect only allowlisted submission artifacts."""

    def commit(self, receipt: SubmissionReceipt, metadata: Mapping[str, Any]) -> None:
        """Atomically persist the retained result manifest and artifacts."""

    def purge(self, run_root: Path, checkpoint_dir: Path) -> None:
        """Delete workspace, caches, and checkpoint bytes after attestation."""


class HarnessAdapter(Protocol):
    def invoke(self, workspace: Path, env: Mapping[str, str]) -> int:
        """Run one declared harness and return its process exit code."""


class SandboxRunner(Protocol):
    def run(self, manifest: Mapping[str, Any]) -> Mapping[str, Any]:
        """Execute prepare→mount→invoke→telemetry→collect→attest→cleanup."""

