"""Hard-cap accounting for the compact local wildfire dataset.

The policy counts every byte below ``data/``.  It deliberately does not evict
anything: callers must obtain admission before adding a new source response,
then record an explicit capped outcome when it cannot fit.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_POLICY_PATH = Path("config/storage_budget.json")


class StorageBudgetError(RuntimeError):
    """Raised when an attempted data addition cannot fit the configured cap."""


@dataclass(frozen=True)
class StorageBudgetCategory:
    """One bounded retention category in the local data package."""

    key: str
    cap_bytes: int
    priority_score: int
    pinned: bool
    retention: str


@dataclass(frozen=True)
class StorageBudgetPolicy:
    """Validated, whole-data-root storage policy."""

    schema_version: int
    whole_data_cap_bytes: int
    whole_data_cap_label: str
    scope: str
    categories: tuple[StorageBudgetCategory, ...]

    @property
    def categories_by_key(self) -> dict[str, StorageBudgetCategory]:
        return {category.key: category for category in self.categories}


@dataclass(frozen=True)
class StorageBudgetUsage:
    """Measured bytes, split deterministically into retention categories."""

    total_bytes: int
    category_bytes: Mapping[str, int]


@dataclass(frozen=True)
class StorageBudgetAdmission:
    """A non-mutating decision on whether one planned write can be admitted."""

    category: str
    requested_bytes: int
    usage: StorageBudgetUsage
    allowed: bool
    reason: str | None


def load_storage_budget(path: str | Path = DEFAULT_POLICY_PATH) -> StorageBudgetPolicy:
    """Load and validate the repository's JSON storage-budget policy."""
    policy_path = Path(path)
    try:
        document = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read storage budget: {policy_path}") from exc
    if not isinstance(document, dict):
        raise ValueError("storage budget must be a JSON object")
    categories_value = document.get("categories")
    if not isinstance(categories_value, list) or not categories_value:
        raise ValueError("storage budget categories must be a non-empty list")
    categories = tuple(_category_from_document(value) for value in categories_value)
    keys = [category.key for category in categories]
    if len(keys) != len(set(keys)):
        raise ValueError("storage budget category keys must be unique")
    cap = _positive_int(document.get("whole_data_cap_bytes"), "whole_data_cap_bytes")
    if sum(category.cap_bytes for category in categories) > cap:
        raise ValueError("storage budget category caps must not exceed whole_data_cap_bytes")
    return StorageBudgetPolicy(
        schema_version=_positive_int(document.get("schema_version"), "schema_version"),
        whole_data_cap_bytes=cap,
        whole_data_cap_label=_required_text(document.get("whole_data_cap_label"), "whole_data_cap_label"),
        scope=_required_text(document.get("scope"), "scope"),
        categories=categories,
    )


def measure_storage_usage(data_root: str | Path) -> StorageBudgetUsage:
    """Measure every regular file below a local data root without following links."""
    root = Path(data_root)
    if not root.exists():
        return StorageBudgetUsage(total_bytes=0, category_bytes={})
    if not root.is_dir():
        raise ValueError("data_root must be a directory")
    category_bytes: defaultdict[str, int] = defaultdict(int)
    total_bytes = 0
    for directory, _subdirectories, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for filename in filenames:
            path = directory_path / filename
            try:
                if path.is_symlink() or not path.is_file():
                    continue
                byte_count = path.stat().st_size
            except OSError:
                continue
            relative_path = path.relative_to(root).as_posix()
            category_bytes[category_for_relative_path(relative_path)] += byte_count
            total_bytes += byte_count
    return StorageBudgetUsage(total_bytes=total_bytes, category_bytes=dict(sorted(category_bytes.items())))


def category_for_relative_path(relative_path: str) -> str:
    """Assign a data-root-relative path to its declared retention category."""
    normalized = relative_path.strip("/")
    if _starts_with_any(
        normalized,
        (
            "raw/nasa-firms/",
            "normalized/fire-detections/",
        ),
    ):
        return "firms_and_detection_evidence"
    if _starts_with_any(
        normalized,
        (
            "raw/nasa-cmr-viirs-l2-observability/",
            "raw/nasa-lp-daac-viirs-l2-observability/",
            "raw/nasa-laads-viirs-geolocation/",
            "normalized/satellite-observation-coverage/",
            "l2/",
        ),
    ):
        return "viirs_l2_paired_cutouts"
    if _starts_with_any(
        normalized,
        (
            "raw/nifc-wfigs/",
            "raw/nifc-irwin/",
            "raw/nasa-feds",
            "raw/cwfis",
            "normalized/operational-perimeters/",
            "normalized/incident-snapshots/",
            "normalized/fire-progression/",
        ),
    ):
        return "operational_labels_and_progression"
    if _starts_with_any(
        normalized,
        (
            "raw/noaa-hrrr/",
            "raw/eccc-hrdps/",
            "normalized/forecast-weather/",
            "weather/",
        ),
    ):
        return "issued_weather_tiles"
    if _starts_with_any(
        normalized,
        (
            "raw/noaa-ncei-etopo-terrain/",
            "raw/cec-nalcms-land-cover/",
            "static/",
            "normalized/static-cell-features/",
        ),
    ):
        return "static_cell_features"
    if _starts_with_any(normalized, ("manifests/", "state/", "retention/", "staging/")):
        return "manifests_staging_and_headroom"
    if _starts_with_any(normalized, ("exports/", "derived/", "normalized/")):
        return "derived_training_views"
    return "unallocated"


def assess_admission(
    policy: StorageBudgetPolicy,
    data_root: str | Path,
    *,
    category: str,
    requested_bytes: int,
) -> StorageBudgetAdmission:
    """Check whole-root and category capacity before a new artifact is written."""
    if requested_bytes < 0:
        raise ValueError("requested_bytes must not be negative")
    categories = policy.categories_by_key
    if category not in categories:
        raise ValueError(f"Unknown storage budget category: {category!r}")
    usage = measure_storage_usage(data_root)
    if usage.total_bytes + requested_bytes > policy.whole_data_cap_bytes:
        return StorageBudgetAdmission(
            category=category,
            requested_bytes=requested_bytes,
            usage=usage,
            allowed=False,
            reason="whole-data cap would be exceeded",
        )
    category_cap = categories[category].cap_bytes
    category_used = usage.category_bytes.get(category, 0)
    if category_used + requested_bytes > category_cap:
        return StorageBudgetAdmission(
            category=category,
            requested_bytes=requested_bytes,
            usage=usage,
            allowed=False,
            reason=f"{category} category cap would be exceeded",
        )
    return StorageBudgetAdmission(
        category=category,
        requested_bytes=requested_bytes,
        usage=usage,
        allowed=True,
        reason=None,
    )


def require_admission(
    policy: StorageBudgetPolicy,
    data_root: str | Path,
    *,
    category: str,
    requested_bytes: int,
) -> StorageBudgetAdmission:
    """Return an allowed admission or raise before a collector persists bytes."""
    admission = assess_admission(
        policy,
        data_root,
        category=category,
        requested_bytes=requested_bytes,
    )
    if not admission.allowed:
        raise StorageBudgetError(
            f"Cannot admit {requested_bytes:,} bytes to {category}: {admission.reason}; "
            f"data root currently uses {admission.usage.total_bytes:,} bytes."
        )
    return admission


def write_storage_inventory(
    policy: StorageBudgetPolicy,
    data_root: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    """Write category bytes and retention scores without modifying source files."""
    root = Path(data_root)
    target_path = Path(output_path) if output_path is not None else root / "retention" / "storage_budget.csv"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # The inventory itself is part of ``data/``. Rewriting it a few times lets
    # the final whole-data row include its own final byte count without ever
    # changing a provider artifact. Three passes cover the only possible digit
    # width change caused by adding the CSV to an empty root.
    passes = 3 if _is_within_root(target_path, root) else 1
    for _ in range(passes):
        usage = measure_storage_usage(root)
        rows = _inventory_rows(policy, usage)
        _write_inventory_csv(target_path, rows)
    return target_path


def _inventory_rows(policy: StorageBudgetPolicy, usage: StorageBudgetUsage) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for category in policy.categories:
        used_bytes = usage.category_bytes.get(category.key, 0)
        rows.append(
            {
                "category": category.key,
                "retention_priority_score": category.priority_score,
                "pinned": str(category.pinned).lower(),
                "cap_bytes": category.cap_bytes,
                "used_bytes": used_bytes,
                "remaining_bytes": max(0, category.cap_bytes - used_bytes),
                "retention": category.retention,
            }
        )
    unallocated_bytes = usage.category_bytes.get("unallocated", 0)
    if unallocated_bytes:
        rows.append(
            {
                "category": "unallocated",
                "retention_priority_score": 0,
                "pinned": "false",
                "cap_bytes": 0,
                "used_bytes": unallocated_bytes,
                "remaining_bytes": 0,
                "retention": "Unclassified bytes count against the whole-data cap and require classification.",
            }
        )
    rows.append(
        {
            "category": "__whole_data__",
            "retention_priority_score": "",
            "pinned": "",
            "cap_bytes": policy.whole_data_cap_bytes,
            "used_bytes": usage.total_bytes,
            "remaining_bytes": max(0, policy.whole_data_cap_bytes - usage.total_bytes),
            "retention": policy.scope,
        }
    )
    return rows


def _write_inventory_csv(target_path: Path, rows: list[dict[str, object]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.", suffix=".tmp", dir=target_path.parent, text=True
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, target_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _category_from_document(value: object) -> StorageBudgetCategory:
    if not isinstance(value, dict):
        raise ValueError("storage budget category must be an object")
    priority = _positive_int(value.get("priority_score"), "priority_score")
    if priority > 100:
        raise ValueError("priority_score must not exceed 100")
    pinned = value.get("pinned")
    if not isinstance(pinned, bool):
        raise ValueError("pinned must be a boolean")
    return StorageBudgetCategory(
        key=_required_text(value.get("key"), "category key"),
        cap_bytes=_positive_int(value.get("cap_bytes"), "cap_bytes"),
        priority_score=priority,
        pinned=pinned,
        retention=_required_text(value.get("retention"), "retention"),
    )


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value.strip()


def _starts_with_any(value: str, prefixes: tuple[str, ...]) -> bool:
    return any(value == prefix.rstrip("/") or value.startswith(prefix) for prefix in prefixes)
