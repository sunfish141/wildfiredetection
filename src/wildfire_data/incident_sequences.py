"""Retrospective FEDS incident groups and splits, fixed before rollout fitting.

Association uses the CURRENT snapshot only. Whole-history spatial grouping is
split metadata, never a feature or inference boundary. Unassociated FIRMS
candidates remain unassigned, not fabricated incidents or observed negatives.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Callable, Mapping, Any

import numpy as np
import pandas as pd
from shapely import STRtree, points
from shapely.geometry import box

from .data_archive import CoverageLedger, CoverageStatus
from .feds_collection import DEFAULT_REGION_LABEL, _observed_snapshot_expected_coverage_id
from .feds_labels import feds_record_geometry
from .rollout_sequences import build_rollout_sequences, RolloutSnapshot
from .train_recursive_transition import _sha256_file, _verify_candidate_checksum, _atomic_json
from .training_grid import cell_from_id


INCIDENT_VERSION = "feds-current-perimeter-incident-complex/v1"
SPLIT_VERSION = "whole-incident-region-later-time/v1"
FEATURE_COLUMNS = (
    "example_id", "cell_id", "source_snapshot_time", "target_snapshot_time",
    "anchor_at", "target_end_at", "dataset_split", "target_newly_burned_12h",
)


@dataclass(frozen=True)
class IncidentPolicy:
    association_distance_m: float = 5000.
    # Separates known groups beyond eight 1.5 km steps plus the feature halo.
    separation_distance_m: float = 20000.
    region_size_cells: int = 1000
    region_feature_halo_cells: int = 1
    region_holdout_fraction: float = .2
    incident_holdout_fraction: float = .2
    calibration_fraction: float = .15
    later_test_at: str = "2026-08-02T12:00:00Z"
    seed: str = "wildfire-incident-split-v1"

    def __post_init__(self):
        for name in ("region_holdout_fraction", "incident_holdout_fraction", "calibration_fraction"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if (not math.isfinite(self.association_distance_m) or self.association_distance_m <= 0
                or not math.isfinite(self.separation_distance_m) or self.separation_distance_m < 20000
                or not isinstance(self.region_size_cells, int) or self.region_size_cells < 1
                or not isinstance(self.region_feature_halo_cells, int) or self.region_feature_halo_cells < 1):
            raise ValueError("invalid incident distance/region policy")
        cutoff = pd.Timestamp(self.later_test_at)
        if cutoff.tzinfo is None:
            raise ValueError("later_test_at must have a timezone")


@dataclass(frozen=True)
class IncidentSequence:
    sequence_id: str
    incident_group_id: str
    split: str
    snapshots: tuple[RolloutSnapshot, ...]


def stable_fraction(value: str) -> float:
    return int(hashlib.sha256(value.encode()).hexdigest()[:13], 16) / 16**13


def region_key(cell_id: str, policy: IncidentPolicy) -> str:
    cell = cell_from_id(cell_id)
    return f"albers-{policy.region_size_cells}km:x={cell.x_index // policy.region_size_cells}:y={cell.y_index // policy.region_size_cells}"


def region_is_held(cell_id: str, policy: IncidentPolicy) -> bool:
    return any(stable_fraction(policy.seed + ":region:" + key) < policy.region_holdout_fraction
               for key in region_context_keys(cell_id, policy.region_size_cells, policy.region_feature_halo_cells))


@lru_cache(maxsize=200000)
def region_context_keys(cell_id, region_size_cells, halo_cells):
    cell = cell_from_id(cell_id)
    return tuple(sorted({f"albers-{region_size_cells}km:x={(cell.x_index+dx)//region_size_cells}:y={(cell.y_index+dy)//region_size_cells}"
                         for dx in range(-halo_cells, halo_cells + 1)
                         for dy in range(-halo_cells, halo_cells + 1)}))


def feds_incident_key(record: Mapping[str, Any]) -> str:
    """Normalize 42, 42.0 and '42.0'; scope provider IDs to region and year."""
    region = record.get("region")
    if region not in ("CONUS", "Canada"):
        raise ValueError("incident record is outside the eligible FEDS regions")
    number = Decimal(str(record.get("fire_id")))
    if not number.is_finite() or number < 0 or number != number.to_integral_value():
        raise ValueError("FEDS fire ID must be an integer")
    year = pd.Timestamp(record["source_snapshot_time"]).year
    return f"feds-incident/v1:{region}:{year}:{int(number)}"


class _Union:
    def __init__(self):
        self.parent = {}

    def find(self, value):
        self.parent.setdefault(value, value)
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def join(self, left, right):
        a, b = self.find(left), self.find(right)
        self.parent[max(a, b)] = min(a, b)


def associate_incidents(
    examples: pd.DataFrame,
    snapshot_loader: Callable[[pd.Timestamp], list[dict]],
    *, policy: IncidentPolicy = IncidentPolicy(),
) -> tuple[pd.DataFrame, dict]:
    """Associate all candidate classes to contemporaneous FEDS perimeters.

    Merge ambiguous associations and groups whose whole-history candidate
    bounding boxes are within 20 km. This deliberately overgroups neighboring
    fires and provider-ID changes, rather than allowing shared context to leak.
    """
    build_rollout_sequences(examples)  # validate times, gaps and cell identities
    if examples.example_id.isna().any() or not examples.example_id.is_unique:
        raise ValueError("example IDs must be unique and present")
    frame = examples.reset_index(drop=True)
    times = pd.to_datetime(frame.source_snapshot_time, utc=True, format="mixed")
    union = _Union()
    members_by_row = [[] for _ in range(len(frame))]
    sources_by_row = [[] for _ in range(len(frame))]
    bounds = {}
    source_snapshots = []
    for timestamp, indices in frame.groupby(times, sort=True).groups.items():
        records = snapshot_loader(timestamp)
        eligible = [r for r in records if r.get("region") in ("CONUS", "Canada")
                    and r.get("time_alignment_eligible") is not False]
        by_id = {}
        for record in eligible:
            if pd.Timestamp(record["source_snapshot_time"]) != timestamp:
                raise ValueError("loader returned a different source snapshot")
            key = feds_incident_key(record)
            if key in by_id:
                raise ValueError("ambiguous FEDS incident revision in snapshot")
            by_id[key] = record
        keys = sorted(by_id)
        if not keys:
            raise ValueError(f"no eligible observed FEDS evidence at {timestamp}")
        geometries = [feds_record_geometry(by_id[key]) for key in keys]
        indices = list(indices)
        cells = [cell_from_id(value) for value in frame.iloc[indices].cell_id]
        xy = np.asarray([cell.center_projected for cell in cells])
        pairs = STRtree(geometries).query(points(xy), predicate="dwithin", distance=policy.association_distance_m)
        for local, geometry_index in pairs.T:
            position, key = indices[local], keys[geometry_index]
            members_by_row[position].append(key)
            source_id = by_id[key].get("source_record_id")
            if not source_id:
                raise ValueError("FEDS incident association requires source record identity")
            sources_by_row[position].append(source_id)
            union.find(key)
            x, y = xy[local]
            old = bounds.get(key, (x, y, x, y))
            bounds[key] = (min(x, old[0]), min(y, old[1]), max(x, old[2]), max(y, old[3]))
        # Record a content fingerprint even for in-memory/test providers.
        content = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        source_snapshots.append({"source_snapshot_time": timestamp.isoformat(),
                                 "records_sha256": hashlib.sha256(content).hexdigest(),
                                 "raw_artifact_ids": sorted({r["raw_artifact_id"] for r in eligible})})
    for members in members_by_row:
        for member in members[1:]:
            union.join(members[0], member)
    keys = sorted(bounds)
    footprints = [box(*bounds[key]).buffer(.01) for key in keys]
    if footprints:
        pairs = STRtree(footprints).query(footprints, predicate="dwithin", distance=policy.separation_distance_m)
        for a, b in pairs.T:
            union.join(keys[a], keys[b])
    members_by_group = defaultdict(list)
    for key in keys:
        members_by_group[union.find(key)].append(key)
    group_id = {root: "feds-complex-v1:" + hashlib.sha256("\n".join(members).encode()).hexdigest()[:24]
                for root, members in members_by_group.items()}
    annotations = pd.DataFrame({
        "example_id": frame.example_id,
        "incident_group_id": [group_id[union.find(m[0])] if m else "" for m in members_by_row],
        "incident_member_keys": [json.dumps(sorted(m)) for m in members_by_row],
        "incident_source_record_ids": [json.dumps(sorted(s)) for s in sources_by_row],
        "incident_assignment_reason": [f"current-FEDS-perimeter-within-{policy.association_distance_m:g}m" if m else "unassigned-no-current-FEDS-support"
                                       for m in members_by_row],
        "region_key": [region_key(value, policy) for value in frame.cell_id],
    })
    groups = {}
    for root, members in members_by_group.items():
        key = group_id[root]
        selected = annotations.incident_group_id == key
        group_regions = sorted({key for cell_id in frame.loc[selected, "cell_id"]
                                for key in region_context_keys(cell_id, policy.region_size_cells, policy.region_feature_halo_cells)})
        region_holdout = any(stable_fraction(policy.seed + ":region:" + r) < policy.region_holdout_fraction
                             for r in group_regions)
        later = bool((times[selected] >= pd.Timestamp(policy.later_test_at)).any())
        fraction = stable_fraction(policy.seed + ":incident:" + key)
        if later:
            split = "later_time"
        elif region_holdout:
            split = "held_region"
        elif fraction < policy.incident_holdout_fraction:
            split = "held_incident"
        elif fraction < policy.incident_holdout_fraction + (1 - policy.incident_holdout_fraction) * policy.calibration_fraction:
            split = "calibration"
        else:
            split = "train"
        groups[key] = {"member_keys": members, "regions": group_regions, "split": split,
                       "touches_held_region": region_holdout, "reaches_later_period": later,
                       "row_count": int(selected.sum())}
    annotations["incident_split"] = [groups[key]["split"] if key else "unassigned" for key in annotations.incident_group_id]
    return annotations, {"incident_version": INCIDENT_VERSION, "split_version": SPLIT_VERSION,
                         "policy": asdict(policy), "groups": groups, "source_snapshots": source_snapshots}


def build_incident_sequences(examples: pd.DataFrame, *, later_test_at: str) -> tuple[IncidentSequence, ...]:
    """Build complete observed fragments; never bridge missing 12-hour states."""
    if examples.incident_split.isna().any():
        raise ValueError("incident splits must be present")
    result = []
    for key, group in examples.groupby("incident_group_id", sort=True):
        if not key or pd.isna(key):
            continue
        splits = set(group.incident_split)
        if len(splits) != 1:
            raise ValueError("whole incident group must have one split")
        split = next(iter(splits))
        positions = np.flatnonzero(examples.incident_group_id.eq(key))
        if split == "later_time":
            positions = positions[(pd.to_datetime(examples.iloc[positions].source_snapshot_time, utc=True, format="mixed")
                                   >= pd.Timestamp(later_test_at)).to_numpy()]
        local = examples.iloc[positions]
        if local.empty:
            continue
        for sequence in build_rollout_sequences(local):
            snapshots = tuple(replace(s, row_positions=tuple(int(positions[i]) for i in s.row_positions))
                              for s in sequence.snapshots)
            identifier = key + ":" + snapshots[0].source_snapshot_time.isoformat()
            result.append(IncidentSequence(identifier, key, split, snapshots))
    return tuple(result)


def selected_snapshot_loader(data_root: Path):
    """Index completed ledger selections once; reject unmanifested revisions."""
    entries = CoverageLedger(data_root).entries()
    latest = {r.expected_coverage_id: r for r in entries}
    selected = {}
    for entry in entries:
        if (entry.expected_coverage_id or "").startswith("feds-nrt-primarykey-snapshot-observed:") and entry.status is CoverageStatus.COMPLETE:
            document = json.loads(entry.path.read_text())
            artifact_id = document.get("detail", {}).get("normalized_artifact_id")
            if artifact_id:
                selected[entry.expected_coverage_id] = (artifact_id, entry.path)
    paths = defaultdict(list)
    for path in (data_root / "normalized/fire-progression").rglob("*.jsonl.gz"):
        paths[path.name.removesuffix(".jsonl.gz")].append(path)
    lineage = []

    def load(timestamp):
        entry = latest.get(_observed_snapshot_expected_coverage_id(timestamp.to_pydatetime(), DEFAULT_REGION_LABEL))
        if entry is None or entry.status is not CoverageStatus.COMPLETE:
            raise ValueError(f"FEDS snapshot is not complete: {timestamp}")
        artifact_id, selection_path = selected.get(entry.expected_coverage_id, (None, None))
        matches = paths.get(artifact_id, [])
        if len(matches) != 1:
            raise ValueError(f"selected FEDS normalized artifact is missing or ambiguous at {timestamp}: {artifact_id}, {len(matches)} paths")
        path = matches[0]
        content = gzip.decompress(path.read_bytes())
        if hashlib.sha256(content).hexdigest() != artifact_id:
            raise ValueError("FEDS normalized artifact checksum mismatch")
        lineage.append({"path": str(path), "content_sha256": artifact_id,
                        "coverage_path": str(entry.path), "coverage_sha256": _sha256_file(entry.path),
                        "selection_coverage_path": str(selection_path), "selection_coverage_sha256": _sha256_file(selection_path)})
        return [json.loads(line) for line in content.splitlines() if line.strip()]

    return load, lineage


def build_release_incidents(release: Path, data_root: Path, output: Path, policy=IncidentPolicy()):
    if output.exists():
        raise ValueError("incident output must be a new directory")
    release_manifest = json.loads((release / "dataset_manifest.json").read_text())
    if (release_manifest.get("schema_version") != 2 or release_manifest.get("weather", {}).get("available") is not False
            or release_manifest.get("kind") != "wildfire-spread-candidate-dataset-release"):
        raise ValueError("requires completed schema-v2 no-weather release")
    path = release / "candidate_examples.csv.gz"
    _verify_candidate_checksum(release, path)
    examples = pd.read_csv(path, usecols=[*FEATURE_COLUMNS, "feds_snapshot_context_raw_artifact_ids"])
    if len(examples) != release_manifest["candidate_row_count"]:
        raise ValueError("release row count mismatch")
    loader, lineage = selected_snapshot_loader(data_root)
    def pinned_loader(timestamp):
        records = loader(timestamp)
        rows = examples[pd.to_datetime(examples.source_snapshot_time, utc=True, format="mixed") == timestamp]
        expected = {identity for value in rows.feds_snapshot_context_raw_artifact_ids for identity in json.loads(value)}
        # The release records pages contributing positive labels, not every
        # contemporaneous context fire. Extra complete current-snapshot pages
        # are legitimate split-only evidence and are separately checksum-pinned.
        available = {identity for r in records for identity in
                     [r["raw_artifact_id"], *r.get("equivalent_raw_artifact_ids", [])]}
        if not available.intersection(expected):
            raise ValueError("selected FEDS revision differs from base release lineage")
        return records
    annotations, metadata = associate_incidents(examples, pinned_loader, policy=policy)
    joined = examples.merge(annotations, on="example_id", validate="one_to_one")
    sequences = build_incident_sequences(joined, later_test_at=policy.later_test_at)
    output.mkdir(parents=True, exist_ok=False)
    annotation_path = output / "incident_assignments.csv.gz"
    annotations.to_csv(annotation_path, index=False, compression={"method": "gzip", "mtime": 0})
    manifest = {"kind": "completed-incident-sequence-view", "schema_version": 1, "status": "complete",
                "source_release": str(release), "source_release_manifest_sha256": _sha256_file(release / "dataset_manifest.json"),
                "assignment_artifact": {"path": str(annotation_path), "sha256": _sha256_file(annotation_path), "row_count": len(annotations)},
                **metadata, "normalized_source_lineage": lineage,
                "split_row_counts": annotations.incident_split.value_counts().to_dict(),
                "sequences": [{"sequence_id": s.sequence_id, "incident_group_id": s.incident_group_id,
                               "split": s.split, "snapshot_count": len(s.snapshots),
                               "start": s.snapshots[0].source_snapshot_time.isoformat(),
                               "end": s.snapshots[-1].source_snapshot_time.isoformat()} for s in sequences]}
    _atomic_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--later-test-at", default=IncidentPolicy().later_test_at)
    args = parser.parse_args()
    manifest = build_release_incidents(args.release, args.data_root, args.output, IncidentPolicy(later_test_at=args.later_test_at))
    print(json.dumps({"split_row_counts": manifest["split_row_counts"], "group_count": len(manifest["groups"]),
                      "sequence_count": len(manifest["sequences"])}, indent=2))


if __name__ == "__main__":
    main()
