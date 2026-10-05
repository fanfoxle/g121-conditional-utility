"""Sealed-source byte verification without dataset record interpretation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable


class SourceMismatch(ValueError):
    """A pinned source is absent or differs from its immutable identity."""


@dataclass(frozen=True)
class VerifiedFile:
    path: Path
    size: int
    sha256: str


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(
    path: Path, *, expected_size: int, expected_sha256: str
) -> VerifiedFile:
    if not path.is_file():
        raise SourceMismatch(f"pinned file is missing: {path}")
    resolved = path.resolve()
    size = resolved.stat().st_size
    if size != expected_size:
        raise SourceMismatch(
            f"byte-size mismatch for {resolved}: expected {expected_size}, observed {size}"
        )
    observed_sha256 = sha256_file(resolved)
    if observed_sha256 != expected_sha256.lower():
        raise SourceMismatch(
            f"SHA-256 mismatch for {resolved}: expected {expected_sha256.lower()}, "
            f"observed {observed_sha256}"
        )
    return VerifiedFile(resolved, size, observed_sha256)


def assert_secret_free_inventory(
    paths: Iterable[Path],
    *,
    workspace: Path,
    forbidden_names: frozenset[str] = frozenset(
        {"cloud.txt", ".env", "credentials.json", "id_rsa", "id_ed25519"}
    ),
) -> tuple[str, ...]:
    """Return canonical relative paths, failing closed on credential filenames.

    This is intentionally a path-level packaging contract.  Exact credential
    values are checked separately at transfer time, after P3 is authorized to
    read ``cloud.txt``.
    """
    root = workspace.resolve()
    inventory: list[str] = []
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise SourceMismatch(f"inventory path escapes workspace: {resolved}")
        relative = resolved.relative_to(root).as_posix()
        if any(part.lower() in forbidden_names for part in resolved.parts):
            raise SourceMismatch(f"credential path is forbidden: {relative}")
        inventory.append(relative)
    if len(inventory) != len(set(inventory)):
        raise SourceMismatch("inventory contains duplicate paths")
    return tuple(sorted(inventory))
