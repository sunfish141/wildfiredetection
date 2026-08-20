"""Append-only storage primitives for reproducible wildfire data collection.

The module deliberately has no network or dataframe dependencies.  Collectors
can use it to retain the exact response bytes they received, then derive CSV,
Parquet, or raster products independently from those immutable inputs.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator, Mapping as MappingABC
from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


ARCHIVE_SCHEMA_VERSION = 1


class ArchiveIntegrityError(RuntimeError):
    """Raised when a content-addressed artifact no longer matches its digest."""


class CoverageStatus(str, Enum):
    """The explicit outcomes for one expected collection scope."""

    COMPLETE = "complete"
    EMPTY_CONFIRMED = "empty-confirmed"
    PARTIAL = "partial"
    FAILED = "failed"


_SECRET_FIELD_PATTERN = re.compile(
    r"(?:api[_-]?key|map[_-]?key|token|secret|password|authorization|cookie|credential|"
    r"signature|access[_-]?key|client[_-]?secret)",
    re.IGNORECASE,
)
_SAFE_COMPONENT_PATTERN = re.compile(r"[^a-z0-9]+")
_BEARER_PATTERN = re.compile(r"(?i)(bearer\s+)[^\s,;]+")
_APPEND_ORDER_LOCK = threading.Lock()
_LAST_APPEND_ORDER = 0


@dataclass(frozen=True)
class RawArtifact(MappingABC[str, Any]):
    """Locations and identity of one immutable raw response capture."""

    raw_artifact_id: str
    artifact_path: Path
    manifest_path: Path
    content_sha256: str
    byte_count: int
    created: bool

    def as_mapping(self) -> dict[str, Any]:
        """Return a JSON-safe summary for downstream manifests and joins."""
        return {
            "raw_artifact_id": self.raw_artifact_id,
            "artifact_path": str(self.artifact_path),
            "manifest_path": str(self.manifest_path),
            "content_sha256": self.content_sha256,
            "byte_count": self.byte_count,
            "created": self.created,
        }

    def __getitem__(self, key: str) -> Any:
        return self.as_mapping()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.as_mapping())

    def __len__(self) -> int:
        return len(self.as_mapping())


@dataclass(frozen=True)
class CoverageRecord:
    """One append-only observation about collection coverage."""

    entry_id: str
    coverage_key: str
    source: str
    product: str
    coverage_start: str
    coverage_end: str
    region: str
    tile: str | None
    expected_coverage_id: str | None
    status: CoverageStatus
    recorded_at: str
    path: Path


def write_atomic_json(path: str | Path, document: Any) -> Path:
    """Atomically replace a JSON document, leaving no partially written target.

    This is intended for mutable snapshots such as a current processing
    manifest.  Append-only records use the private immutable writer below.
    """
    target_path = Path(path)
    encoded = _json_bytes(document)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.", suffix=".tmp", dir=target_path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(encoded)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, target_path)
        _sync_directory(target_path.parent)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return target_path


def write_raw_artifact(
    archive_root: str | Path,
    *,
    source: str,
    payload: bytes | bytearray | memoryview,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    retrieved_at: datetime | None = None,
    capture_id: str | None = None,
    media_type: str | None = None,
) -> RawArtifact:
    """Save exact response bytes once and append a secret-free capture manifest.

    Raw files are content-addressed by the SHA-256 of their *uncompressed*
    bytes.  The returned ``raw_artifact_id`` is that SHA-256.  A repeated
    payload reuses the existing gzip file without changing it, while still
    writing a new manifest for the new collection attempt.
    """
    source_name = _required_text(source, "source")
    source_component = _storage_component(source_name)
    raw_payload = _as_bytes(payload)
    content_sha256 = hashlib.sha256(raw_payload).hexdigest()
    archive_path = Path(archive_root)
    artifact_path = archive_path / "raw" / source_component / f"{content_sha256}.gz"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    created = _write_gzip_artifact_if_missing(artifact_path, raw_payload)
    if not created:
        _verify_gzip_artifact(artifact_path, content_sha256, len(raw_payload))

    captured_at = _normalise_datetime(retrieved_at)
    resolved_capture_id = capture_id or uuid.uuid4().hex
    if not isinstance(resolved_capture_id, str) or not resolved_capture_id.strip():
        raise ValueError("capture_id must be a non-empty string when supplied")
    if any(character in resolved_capture_id for character in "/\\"):
        raise ValueError("capture_id must not contain a path separator")

    manifest = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "kind": "raw-artifact",
        "capture_id": resolved_capture_id,
        "source": source_name,
        "retrieved_at": _isoformat_utc(captured_at),
        "artifact": {
            "relative_path": artifact_path.relative_to(archive_path).as_posix(),
            "compression": "gzip",
            "content_sha256": content_sha256,
            "content_bytes": len(raw_payload),
        },
    }
    if media_type is not None:
        manifest["media_type"] = _required_text(media_type, "media_type")
    if metadata is not None and provenance is not None:
        raise ValueError("supply either metadata or provenance, not both")
    collection_provenance = provenance if provenance is not None else metadata
    if collection_provenance:
        manifest["provenance"] = sanitize_manifest_value(collection_provenance)

    manifest_path = _new_record_path(
        archive_path / "manifests" / "raw" / source_component,
        captured_at,
        f"{content_sha256[:12]}_{resolved_capture_id}",
    )
    _write_immutable_json(manifest_path, manifest)
    return RawArtifact(
        raw_artifact_id=content_sha256,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        content_sha256=content_sha256,
        byte_count=len(raw_payload),
        created=created,
    )


def write_raw_artifact_from_file(
    archive_root: str | Path,
    *,
    source: str,
    source_path: str | Path,
    metadata: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    retrieved_at: datetime | None = None,
    capture_id: str | None = None,
    media_type: str | None = None,
) -> RawArtifact:
    """Archive exact source-file bytes without loading a large response in memory.

    This is the streaming counterpart to :func:`write_raw_artifact`.  It is
    intended for versioned source archives such as large ZIP or GeoTIFF
    releases that are first downloaded into temporary staging outside the
    governed data root.
    """
    source_name = _required_text(source, "source")
    input_path = Path(source_path)
    if input_path.is_symlink() or not input_path.is_file():
        raise ValueError("source_path must be a regular source file")
    content_sha256, content_bytes = _file_digest(input_path)
    archive_path = Path(archive_root)
    source_component = _storage_component(source_name)
    artifact_path = archive_path / "raw" / source_component / f"{content_sha256}.gz"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    created = _write_gzip_file_artifact_if_missing(artifact_path, input_path)
    if not created:
        _verify_gzip_artifact(artifact_path, content_sha256, content_bytes)

    captured_at = _normalise_datetime(retrieved_at)
    resolved_capture_id = capture_id or uuid.uuid4().hex
    if not isinstance(resolved_capture_id, str) or not resolved_capture_id.strip():
        raise ValueError("capture_id must be a non-empty string when supplied")
    if any(character in resolved_capture_id for character in "/\\"):
        raise ValueError("capture_id must not contain a path separator")
    manifest = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "kind": "raw-artifact",
        "capture_id": resolved_capture_id,
        "source": source_name,
        "retrieved_at": _isoformat_utc(captured_at),
        "artifact": {
            "relative_path": artifact_path.relative_to(archive_path).as_posix(),
            "compression": "gzip",
            "content_sha256": content_sha256,
            "content_bytes": content_bytes,
        },
    }
    if media_type is not None:
        manifest["media_type"] = _required_text(media_type, "media_type")
    if metadata is not None and provenance is not None:
        raise ValueError("supply either metadata or provenance, not both")
    collection_provenance = provenance if provenance is not None else metadata
    if collection_provenance:
        manifest["provenance"] = sanitize_manifest_value(collection_provenance)
    manifest_path = _new_record_path(
        archive_path / "manifests" / "raw" / source_component,
        captured_at,
        f"{content_sha256[:12]}_{resolved_capture_id}",
    )
    _write_immutable_json(manifest_path, manifest)
    return RawArtifact(
        raw_artifact_id=content_sha256,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        content_sha256=content_sha256,
        byte_count=content_bytes,
        created=created,
    )


def sanitize_manifest_value(value: Any) -> Any:
    """Return JSON-safe metadata with common credential values redacted.

    Collection manifests are operational records and should be safe to commit
    or share.  Callers should still avoid passing secrets in the first place;
    this guard removes common credential-shaped fields and URL query values.
    """
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, nested_value in value.items():
            string_key = str(key)
            if _SECRET_FIELD_PATTERN.search(string_key):
                sanitized[string_key] = "<redacted>"
            else:
                sanitized[string_key] = sanitize_manifest_value(nested_value)
        return sanitized
    if isinstance(value, (list, tuple, set, frozenset)):
        return [sanitize_manifest_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _isoformat_utc(_normalise_datetime(value))
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


class CoverageLedger:
    """Append-only coverage records for expected source/product/date/tile jobs."""

    def __init__(self, archive_root: str | Path) -> None:
        self._entries_directory = Path(archive_root) / "manifests" / "coverage" / "entries"

    def record(
        self,
        *,
        source: str,
        product: str,
        coverage_start: str | date | datetime,
        coverage_end: str | date | datetime,
        region: str,
        status: CoverageStatus | str,
        tile: str | None = None,
        expected_coverage_id: str | None = None,
        detail: Mapping[str, Any] | None = None,
        message: str | None = None,
        artifact_sha256s: Iterable[str] = (),
        error: str | None = None,
        recorded_at: datetime | None = None,
    ) -> CoverageRecord:
        """Append an explicit outcome for one expected collection scope."""
        source_name = _required_text(source, "source")
        product_name = _required_text(product, "product")
        region_name = _required_text(region, "region")
        start = _coverage_time(coverage_start, "coverage_start")
        end = _coverage_time(coverage_end, "coverage_end")
        tile_name = _required_text(tile, "tile") if tile is not None else None
        expected_id = (
            _required_text(expected_coverage_id, "expected_coverage_id")
            if expected_coverage_id is not None
            else None
        )
        resolved_status = _coverage_status(status)
        captured_at = _normalise_datetime(recorded_at)

        scope = {
            "source": source_name,
            "product": product_name,
            "coverage_start": start,
            "coverage_end": end,
            "region": region_name,
            "tile": tile_name,
            "expected_coverage_id": expected_id,
        }
        coverage_key = hashlib.sha256(_json_bytes(scope)).hexdigest()
        entry_id = uuid.uuid4().hex
        document: dict[str, Any] = {
            "schema_version": ARCHIVE_SCHEMA_VERSION,
            "kind": "coverage-ledger-entry",
            "entry_id": entry_id,
            "recorded_at": _isoformat_utc(captured_at),
            "coverage_key": coverage_key,
            "scope": scope,
            "status": resolved_status.value,
            "artifact_sha256s": _validated_hashes(artifact_sha256s),
            "append_order": _next_append_order(),
        }
        if detail:
            document["detail"] = sanitize_manifest_value(detail)
        if message is not None:
            document["message"] = _sanitize_text(_required_text(message, "message"))
        if error is not None:
            document["error"] = _sanitize_text(_required_text(error, "error"))

        entry_path = _new_record_path(self._entries_directory, captured_at, entry_id)
        _write_immutable_json(entry_path, document)
        return CoverageRecord(
            entry_id=entry_id,
            coverage_key=coverage_key,
            source=source_name,
            product=product_name,
            coverage_start=start,
            coverage_end=end,
            region=region_name,
            tile=tile_name,
            expected_coverage_id=expected_id,
            status=resolved_status,
            recorded_at=document["recorded_at"],
            path=entry_path,
        )

    def entries(self) -> tuple[CoverageRecord, ...]:
        """Return all ledger entries in deterministic capture order."""
        if not self._entries_directory.exists():
            return ()
        records = []
        for entry_path in self._entries_directory.rglob("*.json"):
            try:
                document = json.loads(entry_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ArchiveIntegrityError(f"Invalid coverage ledger entry: {entry_path}") from exc
            try:
                written_at_ns = entry_path.stat().st_mtime_ns
            except OSError as exc:
                raise ArchiveIntegrityError(f"Could not stat coverage ledger entry: {entry_path}") from exc
            append_order = document.get("append_order", written_at_ns)
            if not isinstance(append_order, int) or isinstance(append_order, bool) or append_order < 0:
                raise ArchiveIntegrityError(f"Invalid coverage append order: {entry_path}")
            records.append((_coverage_record_from_document(document, entry_path), append_order, written_at_ns))
        return tuple(
            record
            for record, _append_order, _written_at_ns in sorted(
                records,
                key=lambda item: (item[0].recorded_at, item[1], item[2], item[0].entry_id),
            )
        )

    def latest_by_coverage(self) -> dict[str, CoverageRecord]:
        """Return the newest explicit status for each expected coverage scope."""
        latest: dict[str, CoverageRecord] = {}
        for record in self.entries():
            latest[record.coverage_key] = record
        return latest


def _as_bytes(payload: bytes | bytearray | memoryview) -> bytes:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")
    return bytes(payload)


def _next_append_order() -> int:
    """Return a process-monotonic persisted tie-breaker for ledger entries."""
    global _LAST_APPEND_ORDER
    with _APPEND_ORDER_LOCK:
        _LAST_APPEND_ORDER = max(time.time_ns(), _LAST_APPEND_ORDER + 1)
        return _LAST_APPEND_ORDER


def _write_gzip_artifact_if_missing(target_path: Path, payload: bytes) -> bool:
    """Publish a complete gzip file without ever replacing an existing one."""
    temporary_path = _write_temporary_bytes(
        target_path.parent,
        f".{target_path.name}.",
        gzip.compress(payload, mtime=0),
    )
    try:
        return _publish_immutable_file(temporary_path, target_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_gzip_file_artifact_if_missing(target_path: Path, source_path: Path) -> bool:
    temporary_path = _write_temporary_bytes(target_path.parent, f".{target_path.name}.", b"")
    try:
        with temporary_path.open("wb") as destination, source_path.open("rb") as source:
            with gzip.GzipFile(fileobj=destination, mode="wb", mtime=0) as compressed:
                shutil.copyfileobj(source, compressed, length=1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
        return _publish_immutable_file(temporary_path, target_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _file_digest(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as exc:
        raise ArchiveIntegrityError(f"Could not read source file: {path}") from exc
    return digest.hexdigest(), byte_count


def _verify_gzip_artifact(target_path: Path, expected_sha256: str, expected_bytes: int) -> None:
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with gzip.open(target_path, "rb") as compressed_file:
            while chunk := compressed_file.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
    except (OSError, EOFError) as exc:
        raise ArchiveIntegrityError(f"Unreadable raw artifact: {target_path}") from exc
    if digest.hexdigest() != expected_sha256 or byte_count != expected_bytes:
        raise ArchiveIntegrityError(f"Raw artifact checksum mismatch: {target_path}")


def _write_immutable_json(target_path: Path, document: Any) -> None:
    temporary_path = _write_temporary_bytes(
        target_path.parent,
        f".{target_path.name}.",
        _json_bytes(document),
    )
    try:
        if not _publish_immutable_file(temporary_path, target_path):
            raise ArchiveIntegrityError(f"Refusing to overwrite immutable record: {target_path}")
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_temporary_bytes(directory: Path, prefix: str, content: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=prefix, suffix=".tmp", dir=directory)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _publish_immutable_file(temporary_path: Path, target_path: Path) -> bool:
    """Link a finished temporary file into place, never replacing a target."""
    try:
        os.link(temporary_path, target_path)
    except FileExistsError:
        return False
    _sync_directory(target_path.parent)
    return True


def _new_record_path(directory: Path, recorded_at: datetime, identifier: str) -> Path:
    timestamp = _isoformat_utc(recorded_at).replace(":", "").replace("+00:00", "Z")
    return directory / recorded_at.strftime("%Y/%m/%d") / f"{timestamp}_{identifier}.json"


def _normalise_datetime(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return value.astimezone(timezone.utc)


def _isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _coverage_time(value: str | date | datetime, label: str) -> str:
    if isinstance(value, datetime):
        return _isoformat_utc(_normalise_datetime(value))
    if isinstance(value, date):
        return value.isoformat()
    return _required_text(value, label)


def _coverage_status(value: CoverageStatus | str) -> CoverageStatus:
    try:
        return CoverageStatus(value)
    except ValueError as exc:
        valid_statuses = ", ".join(status.value for status in CoverageStatus)
        raise ValueError(f"status must be one of: {valid_statuses}") from exc


def _validated_hashes(hashes: Iterable[str]) -> list[str]:
    validated = []
    for value in hashes:
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("artifact_sha256s must contain lowercase SHA-256 hex digests")
        validated.append(value)
    return sorted(set(validated))


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _storage_component(value: str) -> str:
    component = _SAFE_COMPONENT_PATTERN.sub("-", value.lower()).strip("-")
    if not component:
        raise ValueError("source must include at least one letter or number")
    return component


def _sanitize_text(value: str) -> str:
    redacted_bearer = _BEARER_PATTERN.sub(r"\1<redacted>", value)
    parts = urlsplit(redacted_bearer)
    if not parts.query:
        return redacted_bearer
    query = parse_qsl(parts.query, keep_blank_values=True)
    sanitized_query = [
        (key, "<redacted>" if _SECRET_FIELD_PATTERN.search(key) else nested_value)
        for key, nested_value in query
    ]
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(sanitized_query), parts.fragment))


def _json_bytes(document: Any) -> bytes:
    return (
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sync_directory(directory: Path) -> None:
    """Best-effort directory sync after publishing a file on POSIX filesystems."""
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _coverage_record_from_document(document: Any, entry_path: Path) -> CoverageRecord:
    if not isinstance(document, dict) or document.get("kind") != "coverage-ledger-entry":
        raise ArchiveIntegrityError(f"Invalid coverage ledger entry: {entry_path}")
    scope = document.get("scope")
    if not isinstance(scope, dict):
        raise ArchiveIntegrityError(f"Invalid coverage ledger entry: {entry_path}")
    try:
        return CoverageRecord(
            entry_id=_required_text(document.get("entry_id"), "entry_id"),
            coverage_key=_required_text(document.get("coverage_key"), "coverage_key"),
            source=_required_text(scope.get("source"), "source"),
            product=_required_text(scope.get("product"), "product"),
            coverage_start=_required_text(scope.get("coverage_start"), "coverage_start"),
            coverage_end=_required_text(scope.get("coverage_end"), "coverage_end"),
            region=_required_text(scope.get("region"), "region"),
            tile=_required_text(scope["tile"], "tile") if scope.get("tile") is not None else None,
            expected_coverage_id=(
                _required_text(scope["expected_coverage_id"], "expected_coverage_id")
                if scope.get("expected_coverage_id") is not None
                else None
            ),
            status=_coverage_status(document.get("status")),
            recorded_at=_required_text(document.get("recorded_at"), "recorded_at"),
            path=entry_path,
        )
    except (TypeError, ValueError) as exc:
        raise ArchiveIntegrityError(f"Invalid coverage ledger entry: {entry_path}") from exc
