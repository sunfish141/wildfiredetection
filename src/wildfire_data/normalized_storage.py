"""Immutable, lossless normalized-record storage.

Raw provider responses are the evidence layer.  This module stores normalized
records as deterministic gzip-compressed JSON Lines so nested provider fields
and provenance remain lossless without imposing a prematurely fixed tabular
schema.  Analytics-specific Parquet views can always be regenerated from
these artifacts.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .data_archive import sanitize_manifest_value, write_atomic_json


NORMALIZED_STORAGE_SCHEMA_VERSION = 1
_SAFE_COMPONENT = re.compile(r"[^a-z0-9]+")


class NormalizedStorageIntegrityError(RuntimeError):
    """Raised when an existing normalized artifact does not match its hash."""


@dataclass(frozen=True)
class NormalizedArtifact:
    """Identity and paths for one content-addressed normalized record set."""

    normalized_artifact_id: str
    artifact_path: Path
    manifest_path: Path
    record_count: int
    created: bool


def write_normalized_jsonl(
    archive_root: str | Path,
    *,
    entity: str,
    records: Iterable[Mapping[str, Any]],
    partitions: Mapping[str, object],
    raw_artifact_ids: Iterable[str],
    transformation_version: str,
    generated_at: datetime | None = None,
) -> NormalizedArtifact:
    """Persist one append-only, content-addressed normalized record set.

    The records are written in canonical JSON Lines form.  Repeating the same
    source rows and transformation reuses the immutable artifact while still
    creating a new collection manifest that records the attempted generation.
    """
    entity_name = _required_text(entity, "entity")
    version = _required_text(transformation_version, "transformation_version")
    resolved_partitions = {
        _required_text(str(key), "partition key"): _required_text(
            str(value), f"partition {key}"
        )
        for key, value in partitions.items()
    }
    if not resolved_partitions:
        raise ValueError("partitions must not be empty")
    source_ids = tuple(_required_text(identifier, "raw_artifact_id") for identifier in raw_artifact_ids)
    if not source_ids:
        raise ValueError("raw_artifact_ids must not be empty")

    encoded_lines = []
    record_count = 0
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("normalized records must be mappings")
        try:
            encoded = json.dumps(
                record,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("normalized records must be JSON-safe") from exc
        encoded_lines.append(encoded)
        record_count += 1
    if not encoded_lines:
        raise ValueError("normalized records must not be empty")

    raw_jsonl = b"\n".join(encoded_lines) + b"\n"
    artifact_id = hashlib.sha256(raw_jsonl).hexdigest()
    archive_path = Path(archive_root)
    partition_path = archive_path / "normalized" / _component(entity_name)
    for key, value in sorted(resolved_partitions.items()):
        partition_path /= f"{_component(key)}={_component(value)}"
    artifact_path = partition_path / f"{artifact_id}.jsonl.gz"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    created = _publish_if_missing(artifact_path, gzip.compress(raw_jsonl, mtime=0))
    if not created:
        _verify_existing_artifact(artifact_path, artifact_id)

    captured_at = _normalise_datetime(generated_at)
    manifest = {
        "schema_version": NORMALIZED_STORAGE_SCHEMA_VERSION,
        "kind": "normalized-jsonl",
        "generation_id": uuid.uuid4().hex,
        "generated_at": _format_utc(captured_at),
        "entity": entity_name,
        "partitions": resolved_partitions,
        "transformation_version": version,
        "raw_artifact_ids": sorted(set(source_ids)),
        "artifact": {
            "relative_path": artifact_path.relative_to(archive_path).as_posix(),
            "compression": "gzip",
            "content_sha256": artifact_id,
            "record_count": record_count,
        },
    }
    manifest_path = (
        archive_path
        / "manifests"
        / "normalized"
        / _component(entity_name)
        / captured_at.strftime("%Y/%m/%d")
        / f"{artifact_id[:12]}_{manifest['generation_id']}.json"
    )
    write_atomic_json(manifest_path, sanitize_manifest_value(manifest))
    return NormalizedArtifact(
        normalized_artifact_id=artifact_id,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        record_count=record_count,
        created=created,
    )


def _publish_if_missing(target_path: Path, payload: bytes) -> bool:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.", suffix=".tmp", dir=target_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(payload)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        try:
            os.link(temporary_path, target_path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary_path.unlink(missing_ok=True)


def _verify_existing_artifact(target_path: Path, expected_hash: str) -> None:
    digest = hashlib.sha256()
    try:
        with gzip.open(target_path, "rb") as artifact_file:
            while chunk := artifact_file.read(1024 * 1024):
                digest.update(chunk)
    except (OSError, EOFError) as exc:
        raise NormalizedStorageIntegrityError(
            f"Could not read normalized artifact: {target_path}"
        ) from exc
    if digest.hexdigest() != expected_hash:
        raise NormalizedStorageIntegrityError(
            f"Normalized artifact does not match its content hash: {target_path}"
        )


def _required_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _component(value: str) -> str:
    component = _SAFE_COMPONENT.sub("-", value.lower()).strip("-")
    if not component:
        raise ValueError("storage component must contain a letter or number")
    return component


def _normalise_datetime(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise ValueError("generated_at must include a UTC offset")
    return value.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")
