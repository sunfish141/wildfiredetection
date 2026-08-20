"""Inventory and archive VIIRS Level-2 active-fire files with provenance.

Collection 2 VNP14IMG, VJ114IMG, and VJ214IMG files contain fire mask and
algorithm QA. Latitude/longitude live in matching VNP03IMG, VJ103IMG, and
VJ203IMG geolocation products, respectively. This legacy active-fire-file
collector therefore cannot by itself create a complete L2 observation; the
compact 20 GB policy requires a paired cutout before admitting L2 pixels.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .collection_catalog import CollectionTarget, target_by_key
from .data_archive import CoverageLedger, CoverageRecord, CoverageStatus, RawArtifact, write_raw_artifact


CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
DEFAULT_BBOX = "-179,24,-52,84"
DEFAULT_REGION = "United States and Canada"
DEFAULT_PLATFORMS = ("snpp", "noaa20", "noaa21")
CMR_PAGE_SIZE = 2_000
EMPTY_CONFIRMATION_LAG = timedelta(hours=24)


@dataclass(frozen=True)
class ViirsL2Product:
    """One Collection 2 active-fire product and its required geo companion."""

    platform_key: str
    platform_name: str
    short_name: str
    geolocation_short_name: str
    collection_version: str


VIIRS_L2_PRODUCTS = {
    "snpp": ViirsL2Product(
        platform_key="snpp",
        platform_name="Suomi NPP",
        short_name="VNP14IMG",
        geolocation_short_name="VNP03IMG",
        collection_version="002",
    ),
    "noaa20": ViirsL2Product(
        platform_key="noaa20",
        platform_name="NOAA-20",
        short_name="VJ114IMG",
        geolocation_short_name="VJ103IMG",
        collection_version="002",
    ),
    "noaa21": ViirsL2Product(
        platform_key="noaa21",
        platform_name="NOAA-21",
        short_name="VJ214IMG",
        geolocation_short_name="VJ203IMG",
        collection_version="002",
    ),
}


@dataclass(frozen=True)
class ViirsL2Granule:
    """A CMR-described active-fire L2 file selected for archival."""

    product: ViirsL2Product
    native_granule_id: str
    observation_start: datetime
    observation_end: datetime
    download_url: str
    native_filename: str
    collection_concept_id: str | None
    spatial_extent: Mapping[str, Any] | None
    reported_size_bytes: int | None
    checksum_algorithm: str | None
    checksum_value: str | None


@dataclass(frozen=True)
class ViirsL2GranuleReceipt:
    """One archived (or failed) L2 granule acquisition."""

    granule: ViirsL2Granule
    raw_artifact: RawArtifact | None
    coverage: CoverageRecord


@dataclass(frozen=True)
class ViirsL2DayCollection:
    """Evidence and final coverage outcome for one product and UTC date."""

    product: ViirsL2Product
    coverage_date: date
    inventory_artifacts: tuple[RawArtifact, ...]
    granules: tuple[ViirsL2Granule, ...]
    granule_receipts: tuple[ViirsL2GranuleReceipt, ...]
    coverage: CoverageRecord
    skipped_granules: int
    skipped_terminal_coverage: bool = False


@dataclass(frozen=True)
class ViirsL2RangeCollection:
    """Every durable outcome from a VIIRS L2 range collection attempt."""

    days: tuple[ViirsL2DayCollection, ...]
    dry_run: bool

    @property
    def inventory_response_count(self) -> int:
        return sum(len(day.inventory_artifacts) for day in self.days)

    @property
    def discovered_granule_count(self) -> int:
        return sum(len(day.granules) for day in self.days)

    @property
    def archived_granule_count(self) -> int:
        return sum(
            receipt.coverage.status is CoverageStatus.COMPLETE
            for day in self.days
            for receipt in day.granule_receipts
        )

    @property
    def skipped_granule_count(self) -> int:
        return sum(day.skipped_granules for day in self.days)

    @property
    def incomplete_window_count(self) -> int:
        terminal = {CoverageStatus.COMPLETE, CoverageStatus.EMPTY_CONFIRMED}
        return sum(day.coverage.status not in terminal for day in self.days)


class ViirsL2InventoryError(ValueError):
    """Raised when a successful CMR response cannot describe usable swaths."""


def product_for_platform(platform: str) -> ViirsL2Product:
    """Resolve a CLI-friendly platform key to its Level-2 product."""
    normalized = platform.strip().lower().replace("-", "")
    aliases = {"suominpp": "snpp", "npp": "snpp", "noaa20": "noaa20", "noaa21": "noaa21"}
    platform_key = aliases.get(normalized, normalized)
    try:
        return VIIRS_L2_PRODUCTS[platform_key]
    except KeyError as exc:
        supported = ", ".join(DEFAULT_PLATFORMS)
        raise ValueError(f"Unknown VIIRS platform {platform!r}; use one of: {supported}") from exc


def products_for_platforms(platforms: Iterable[str]) -> tuple[ViirsL2Product, ...]:
    """Resolve platforms while preserving order and removing duplicates."""
    products = []
    seen = set()
    for platform in platforms:
        product = product_for_platform(platform)
        if product.platform_key not in seen:
            products.append(product)
            seen.add(product.platform_key)
    if not products:
        raise ValueError("platforms must contain at least one platform")
    return tuple(products)


def cmr_inventory_parameters(
    product: ViirsL2Product,
    *,
    coverage_date: date,
    bbox: str,
    page_number: int = 1,
) -> dict[str, str | int]:
    """Build a single-day CMR inventory query for a product and spatial scope."""
    if not bbox.strip():
        raise ValueError("bbox must be non-empty")
    if page_number < 1:
        raise ValueError("page_number must be positive")
    start, end = _day_bounds(coverage_date)
    return {
        "short_name": product.short_name,
        "version": product.collection_version,
        "temporal": f"{_cmr_time(start)},{_cmr_time(end)}",
        "bounding_box": bbox,
        "page_size": CMR_PAGE_SIZE,
        "page_num": page_number,
    }


def parse_cmr_inventory_payload(
    payload: bytes,
    *,
    product: ViirsL2Product,
) -> tuple[int, tuple[ViirsL2Granule, ...]]:
    """Parse one raw CMR UMM-JSON response without discarding its source data."""
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ViirsL2InventoryError("CMR inventory response is not valid JSON") from exc
    if not isinstance(document, dict):
        raise ViirsL2InventoryError("CMR inventory response must be a JSON object")
    raw_items = document.get("items")
    if not isinstance(raw_items, list):
        raise ViirsL2InventoryError("CMR inventory response has no items list")
    hits = _nonnegative_int(document.get("hits"), "CMR hits")
    granules = []
    for item in raw_items:
        granules.append(_parse_cmr_granule(item, product))
    return hits, tuple(granules)


def is_netcdf_payload(payload: bytes) -> bool:
    """Recognize the standard NetCDF classic and NetCDF4/HDF5 signatures."""
    return payload.startswith((b"CDF\x01", b"CDF\x02", b"CDF\x05", b"\x89HDF\r\n\x1a\n"))


def collect_viirs_l2_range(
    archive_root: str,
    *,
    start_date: date,
    end_date: date,
    platforms: Iterable[str] = DEFAULT_PLATFORMS,
    bbox: str = DEFAULT_BBOX,
    region: str = DEFAULT_REGION,
    earthdata_token: str | None = None,
    dry_run: bool = False,
    session: requests.Session | None = None,
    timeout: tuple[int, int] = (10, 180),
    retrieved_at: datetime | None = None,
    empty_confirmation_lag: timedelta = EMPTY_CONFIRMATION_LAG,
) -> ViirsL2RangeCollection:
    """Archive CMR inventory evidence and matching active-fire NetCDF files.

    This collector intentionally leaves geolocation pairing to the compact
    source-pair workflow. ``dry_run`` persists CMR inventory responses,
    leaving non-empty date/product windows ``partial`` for a later run.
    """
    if end_date < start_date:
        raise ValueError("end_date must not be before start_date")
    if not bbox.strip():
        raise ValueError("bbox must be non-empty")
    if not region.strip():
        raise ValueError("region must be non-empty")
    if empty_confirmation_lag < timedelta(0):
        raise ValueError("empty_confirmation_lag must not be negative")
    token = (earthdata_token or "").strip()
    if not dry_run and not token:
        raise ValueError("earthdata_token is required unless dry_run is enabled")

    products = products_for_platforms(platforms)
    target = target_by_key("viirs_l2_observability")
    captured_at = _retrieved_at(retrieved_at)
    ledger = CoverageLedger(archive_root)
    latest_by_expected_id = {
        record.expected_coverage_id: record
        for record in ledger.entries()
        if record.expected_coverage_id is not None
    }

    owns_session = session is None
    active_session = session or _retrying_session()
    days = []
    try:
        coverage_date = start_date
        while coverage_date <= end_date:
            for product in products:
                parent_expected_id = _parent_expected_id(product, coverage_date, bbox, region)
                existing_parent = latest_by_expected_id.get(parent_expected_id)
                if _is_terminal(existing_parent):
                    days.append(
                        ViirsL2DayCollection(
                            product=product,
                            coverage_date=coverage_date,
                            inventory_artifacts=(),
                            granules=(),
                            granule_receipts=(),
                            coverage=existing_parent,
                            skipped_granules=0,
                            skipped_terminal_coverage=True,
                        )
                    )
                    continue

                day_result = _collect_product_day(
                    archive_root,
                    target=target,
                    product=product,
                    coverage_date=coverage_date,
                    bbox=bbox,
                    region=region,
                    token=token,
                    dry_run=dry_run,
                    session=active_session,
                    timeout=timeout,
                    retrieved_at=captured_at,
                    empty_confirmation_lag=empty_confirmation_lag,
                    latest_by_expected_id=latest_by_expected_id,
                )
                days.append(day_result)
                latest_by_expected_id[parent_expected_id] = day_result.coverage
            coverage_date += timedelta(days=1)
    finally:
        if owns_session:
            active_session.close()
    return ViirsL2RangeCollection(days=tuple(days), dry_run=dry_run)


def _collect_product_day(
    archive_root: str,
    *,
    target: CollectionTarget,
    product: ViirsL2Product,
    coverage_date: date,
    bbox: str,
    region: str,
    token: str,
    dry_run: bool,
    session: requests.Session,
    timeout: tuple[int, int],
    retrieved_at: datetime,
    empty_confirmation_lag: timedelta,
    latest_by_expected_id: dict[str, CoverageRecord],
) -> ViirsL2DayCollection:
    """Collect one date/product inventory, then all pending listed swaths."""
    inventory_artifacts, granules, inventory_error = _fetch_inventory(
        archive_root,
        target=target,
        product=product,
        coverage_date=coverage_date,
        bbox=bbox,
        session=session,
        timeout=timeout,
        retrieved_at=retrieved_at,
    )
    start, end = _day_bounds(coverage_date)
    parent_expected_id = _parent_expected_id(product, coverage_date, bbox, region)
    inventory_hashes = [artifact.raw_artifact_id for artifact in inventory_artifacts]
    if inventory_error is not None:
        coverage = _record_parent_coverage(
            archive_root,
            target=target,
            product=product,
            coverage_date=coverage_date,
            bbox=bbox,
            region=region,
            status=CoverageStatus.FAILED,
            artifact_sha256s=inventory_hashes,
            detail={"inventory_response_count": len(inventory_artifacts)},
            error=inventory_error,
            retrieved_at=retrieved_at,
        )
        return ViirsL2DayCollection(
            product=product,
            coverage_date=coverage_date,
            inventory_artifacts=inventory_artifacts,
            granules=(),
            granule_receipts=(),
            coverage=coverage,
            skipped_granules=0,
        )

    if not granules:
        status = (
            CoverageStatus.EMPTY_CONFIRMED
            if end <= retrieved_at - empty_confirmation_lag
            else CoverageStatus.PARTIAL
        )
        coverage = _record_parent_coverage(
            archive_root,
            target=target,
            product=product,
            coverage_date=coverage_date,
            bbox=bbox,
            region=region,
            status=status,
            artifact_sha256s=inventory_hashes,
            detail={
                "inventory_response_count": len(inventory_artifacts),
                "expected_granule_count": 0,
                "empty_confirmation_lag_hours": empty_confirmation_lag.total_seconds() / 3600,
            },
            message=(
                "CMR reported no matching swaths after the empty-confirmation lag."
                if status is CoverageStatus.EMPTY_CONFIRMED
                else "CMR reported no matching swaths, but the date is still within the publication lag."
            ),
            retrieved_at=retrieved_at,
        )
        return ViirsL2DayCollection(
            product=product,
            coverage_date=coverage_date,
            inventory_artifacts=inventory_artifacts,
            granules=(),
            granule_receipts=(),
            coverage=coverage,
            skipped_granules=0,
        )

    receipts = []
    skipped_granules = 0
    completed_granule_count = 0
    granule_hashes = []
    for granule in granules:
        expected_id = _granule_expected_id(product, granule, bbox, region)
        existing_granule = latest_by_expected_id.get(expected_id)
        if _is_terminal(existing_granule):
            skipped_granules += 1
            completed_granule_count += 1
            continue
        if dry_run:
            continue
        receipt = _archive_granule(
            archive_root,
            target=target,
            product=product,
            granule=granule,
            region=region,
            bbox=bbox,
            inventory_artifact_ids=inventory_hashes,
            token=token,
            session=session,
            timeout=timeout,
            retrieved_at=retrieved_at,
        )
        receipts.append(receipt)
        latest_by_expected_id[expected_id] = receipt.coverage
        if receipt.coverage.status is CoverageStatus.COMPLETE:
            completed_granule_count += 1
            if receipt.raw_artifact is not None:
                granule_hashes.append(receipt.raw_artifact.raw_artifact_id)

    parent_status = (
        CoverageStatus.COMPLETE
        if completed_granule_count == len(granules)
        else CoverageStatus.PARTIAL
    )
    coverage = _record_parent_coverage(
        archive_root,
        target=target,
        product=product,
        coverage_date=coverage_date,
        bbox=bbox,
        region=region,
        status=parent_status,
        artifact_sha256s=[*inventory_hashes, *granule_hashes],
        detail={
            "inventory_response_count": len(inventory_artifacts),
            "expected_granule_count": len(granules),
            "completed_granule_count": completed_granule_count,
            "skipped_completed_granule_count": skipped_granules,
            "download_attempted_granule_count": len(receipts),
            "dry_run": dry_run,
            "parent_expected_coverage_id": parent_expected_id,
            "fire_file_asset_roles": ["fire_mask", "quality"],
            "required_geolocation_product": product.geolocation_short_name,
        },
        message=(
            "All CMR-listed active-fire Level-2 files are archived by the legacy collector; "
            "matching geolocation is not included."
            if parent_status is CoverageStatus.COMPLETE
            else "Some listed Level-2 swaths remain pending or failed; retry this date/product window."
        ),
        retrieved_at=retrieved_at,
    )
    return ViirsL2DayCollection(
        product=product,
        coverage_date=coverage_date,
        inventory_artifacts=inventory_artifacts,
        granules=granules,
        granule_receipts=tuple(receipts),
        coverage=coverage,
        skipped_granules=skipped_granules,
    )


def _fetch_inventory(
    archive_root: str,
    *,
    target: CollectionTarget,
    product: ViirsL2Product,
    coverage_date: date,
    bbox: str,
    session: requests.Session,
    timeout: tuple[int, int],
    retrieved_at: datetime,
) -> tuple[tuple[RawArtifact, ...], tuple[ViirsL2Granule, ...], str | None]:
    """Fetch and archive every page of one CMR daily inventory query."""
    artifacts = []
    granules_by_id = {}
    page_number = 1
    total_hits = None
    while True:
        parameters = cmr_inventory_parameters(
            product,
            coverage_date=coverage_date,
            bbox=bbox,
            page_number=page_number,
        )
        try:
            response = session.get(
                CMR_GRANULES_URL,
                params=parameters,
                headers={"Accept": "application/vnd.nasa.cmr.umm_results+json"},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            return tuple(artifacts), (), f"CMR inventory request failed: {exc}"

        response_headers = dict(response.headers)
        artifact = write_raw_artifact(
            archive_root,
            source="NASA CMR:viirs_l2_observability",
            payload=response.content,
            retrieved_at=retrieved_at,
            media_type=_response_media_type(response_headers, "application/json"),
            provenance={
                "source_url": getattr(response, "url", CMR_GRANULES_URL),
                "request_parameters": parameters,
                "response_headers": response_headers,
                "response_status_code": response.status_code,
                "target": {"key": target.key, "entity": target.entity},
                "inventory_kind": "CMR UMM-JSON granule search",
                "product": product.short_name,
                "collection_version": product.collection_version,
                "platform": product.platform_name,
                "coverage_date": coverage_date,
                "bbox": bbox,
                "page_number": page_number,
            },
        )
        artifacts.append(artifact)
        if not 200 <= response.status_code < 300:
            return (
                tuple(artifacts),
                (),
                f"CMR inventory returned HTTP {response.status_code}",
            )
        try:
            page_hits, page_granules = parse_cmr_inventory_payload(
                response.content,
                product=product,
            )
        except ViirsL2InventoryError as exc:
            return tuple(artifacts), (), str(exc)
        if total_hits is None:
            total_hits = page_hits
        elif page_hits != total_hits:
            return tuple(artifacts), (), "CMR inventory hit count changed between pages"
        for granule in page_granules:
            granules_by_id[granule.native_granule_id] = granule
        if total_hits <= page_number * CMR_PAGE_SIZE:
            break
        if not page_granules:
            return tuple(artifacts), (), "CMR inventory returned an empty page before all hits"
        page_number += 1
    ordered = tuple(
        sorted(granules_by_id.values(), key=lambda granule: (granule.observation_start, granule.native_granule_id))
    )
    return tuple(artifacts), ordered, None


def _archive_granule(
    archive_root: str,
    *,
    target: CollectionTarget,
    product: ViirsL2Product,
    granule: ViirsL2Granule,
    region: str,
    bbox: str,
    inventory_artifact_ids: list[str],
    token: str,
    session: requests.Session,
    timeout: tuple[int, int],
    retrieved_at: datetime,
) -> ViirsL2GranuleReceipt:
    """Download one whole protected NetCDF file and record its exact outcome."""
    expected_id = _granule_expected_id(product, granule, bbox, region)
    provenance = {
        "source_url": granule.download_url,
        "target": {"key": target.key, "entity": target.entity},
        "product": product.short_name,
        "collection_version": product.collection_version,
        "platform": product.platform_name,
        "native_granule_id": granule.native_granule_id,
        "native_filename": granule.native_filename,
        "observation_start": granule.observation_start,
        "observation_end": granule.observation_end,
        "spatial_extent": granule.spatial_extent,
        "collection_concept_id": granule.collection_concept_id,
        "reported_size_bytes": granule.reported_size_bytes,
        "checksum_algorithm": granule.checksum_algorithm,
        "checksum_value": granule.checksum_value,
        "asset_role": "fire_mask_and_algorithm_qa",
        "fire_file_asset_roles": ["fire_mask", "quality"],
        "required_geolocation_product": product.geolocation_short_name,
        "inventory_artifact_ids": inventory_artifact_ids,
    }
    try:
        response = session.get(
            granule.download_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        coverage = CoverageLedger(archive_root).record(
            source=target.provider,
            product=product.short_name,
            coverage_start=granule.observation_start,
            coverage_end=granule.observation_end,
            region=region,
            tile=granule.native_granule_id,
            expected_coverage_id=expected_id,
            status=CoverageStatus.FAILED,
            error=f"L2 granule request failed: {exc}",
            detail={
                "parent_scope": _scope_component(bbox),
                "inventory_artifact_ids": inventory_artifact_ids,
                "native_filename": granule.native_filename,
            },
            recorded_at=retrieved_at,
        )
        return ViirsL2GranuleReceipt(granule=granule, raw_artifact=None, coverage=coverage)

    response_headers = dict(response.headers)
    artifact = write_raw_artifact(
        archive_root,
        source="NASA LP DAAC:viirs_l2_observability",
        payload=response.content,
        retrieved_at=retrieved_at,
        media_type=_response_media_type(response_headers, "application/x-netcdf"),
        provenance={
            **provenance,
            "response_url": getattr(response, "url", granule.download_url),
            "response_headers": response_headers,
            "response_status_code": response.status_code,
        },
    )
    error = _granule_validation_error(granule, response.status_code, response_headers, response.content)
    coverage = CoverageLedger(archive_root).record(
        source=target.provider,
        product=product.short_name,
        coverage_start=granule.observation_start,
        coverage_end=granule.observation_end,
        region=region,
        tile=granule.native_granule_id,
        expected_coverage_id=expected_id,
        status=CoverageStatus.COMPLETE if error is None else CoverageStatus.FAILED,
        artifact_sha256s=[artifact.raw_artifact_id],
        detail={
            "parent_scope": _scope_component(bbox),
            "inventory_artifact_ids": inventory_artifact_ids,
            "native_filename": granule.native_filename,
            "asset_role": "fire_mask_and_algorithm_qa",
            "fire_file_asset_roles": ["fire_mask", "quality"],
            "required_geolocation_product": product.geolocation_short_name,
        },
        error=error,
        recorded_at=retrieved_at,
    )
    return ViirsL2GranuleReceipt(granule=granule, raw_artifact=artifact, coverage=coverage)


def _record_parent_coverage(
    archive_root: str,
    *,
    target: CollectionTarget,
    product: ViirsL2Product,
    coverage_date: date,
    bbox: str,
    region: str,
    status: CoverageStatus,
    artifact_sha256s: Iterable[str],
    detail: Mapping[str, Any],
    message: str | None = None,
    error: str | None = None,
    retrieved_at: datetime,
) -> CoverageRecord:
    start, end = _day_bounds(coverage_date)
    return CoverageLedger(archive_root).record(
        source=target.provider,
        product=product.short_name,
        coverage_start=start,
        coverage_end=end,
        region=region,
        tile=_scope_component(bbox),
        expected_coverage_id=_parent_expected_id(product, coverage_date, bbox, region),
        status=status,
        artifact_sha256s=artifact_sha256s,
        detail={
            "target_key": target.key,
            "platform": product.platform_name,
            "collection_version": product.collection_version,
            "bbox": bbox,
            **dict(detail),
        },
        message=message,
        error=error,
        recorded_at=retrieved_at,
    )


def _parse_cmr_granule(item: Any, product: ViirsL2Product) -> ViirsL2Granule:
    if not isinstance(item, dict):
        raise ViirsL2InventoryError("CMR inventory item must be an object")
    umm = item.get("umm")
    if not isinstance(umm, dict):
        raise ViirsL2InventoryError("CMR inventory item has no UMM metadata")
    reference = umm.get("CollectionReference")
    if not isinstance(reference, dict):
        raise ViirsL2InventoryError("CMR granule has no collection reference")
    if reference.get("ShortName") != product.short_name or reference.get("Version") != product.collection_version:
        raise ViirsL2InventoryError("CMR returned a granule from an unexpected product/version")
    granule_id = umm.get("GranuleUR")
    if not isinstance(granule_id, str) or not granule_id.strip():
        raise ViirsL2InventoryError("CMR granule has no native granule ID")
    temporal_extent = umm.get("TemporalExtent")
    range_datetime = temporal_extent.get("RangeDateTime") if isinstance(temporal_extent, dict) else None
    if not isinstance(range_datetime, dict):
        raise ViirsL2InventoryError(f"CMR granule {granule_id} has no observation interval")
    observation_start = _parse_timestamp(range_datetime.get("BeginningDateTime"), "BeginningDateTime")
    observation_end = _parse_timestamp(range_datetime.get("EndingDateTime"), "EndingDateTime")
    if observation_end <= observation_start:
        raise ViirsL2InventoryError(f"CMR granule {granule_id} has an invalid observation interval")
    download_url = _download_url(umm.get("RelatedUrls"), granule_id)
    data_granule = umm.get("DataGranule")
    if not isinstance(data_granule, dict):
        data_granule = {}
    checksum_algorithm, checksum_value = _checksum(data_granule.get("Checksum"))
    return ViirsL2Granule(
        product=product,
        native_granule_id=granule_id,
        observation_start=observation_start,
        observation_end=observation_end,
        download_url=download_url,
        native_filename=urlsplit(download_url).path.rsplit("/", maxsplit=1)[-1],
        collection_concept_id=_optional_text(_mapping_value(item.get("meta"), "collection-concept-id")),
        spatial_extent=_mapping_or_none(umm.get("SpatialExtent")),
        reported_size_bytes=_reported_size_bytes(data_granule),
        checksum_algorithm=checksum_algorithm,
        checksum_value=checksum_value,
    )


def _download_url(related_urls: Any, granule_id: str) -> str:
    if not isinstance(related_urls, list):
        raise ViirsL2InventoryError(f"CMR granule {granule_id} has no related URLs")
    for related_url in related_urls:
        if not isinstance(related_url, dict):
            continue
        url = related_url.get("URL")
        if (
            related_url.get("Type") == "GET DATA"
            and isinstance(url, str)
            and url.startswith("https://")
            and urlsplit(url).path.lower().endswith(".nc")
        ):
            return url
    raise ViirsL2InventoryError(f"CMR granule {granule_id} has no HTTPS NetCDF download URL")


def _checksum(value: Any) -> tuple[str | None, str | None]:
    if not isinstance(value, dict):
        return None, None
    algorithm = _optional_text(value.get("Algorithm"))
    checksum_value = _optional_text(value.get("Value"))
    if (algorithm is None) != (checksum_value is None):
        raise ViirsL2InventoryError("CMR granule has an incomplete checksum")
    return algorithm, checksum_value


def _reported_size_bytes(data_granule: Mapping[str, Any]) -> int | None:
    distributions = data_granule.get("ArchiveAndDistributionInformation")
    if not isinstance(distributions, list) or not distributions:
        return None
    first = distributions[0]
    if not isinstance(first, dict):
        return None
    size = first.get("Size")
    unit = first.get("SizeUnit")
    if not isinstance(size, (int, float)) or size < 0 or not isinstance(unit, str):
        return None
    units = {"kb": 1_000, "mb": 1_000_000, "gb": 1_000_000_000}
    multiplier = units.get(unit.lower())
    return round(size * multiplier) if multiplier is not None else None


def _granule_validation_error(
    granule: ViirsL2Granule,
    status_code: int,
    headers: Mapping[str, Any],
    payload: bytes,
) -> str | None:
    if not 200 <= status_code < 300:
        return f"HTTP {status_code}"
    if not is_netcdf_payload(payload):
        return "Downloaded payload does not have a NetCDF signature"
    content_length = headers.get("Content-Length")
    if content_length is not None and not headers.get("Content-Encoding"):
        try:
            expected_length = int(str(content_length))
        except ValueError:
            return "Response has an invalid Content-Length"
        if expected_length != len(payload):
            return "Response Content-Length does not match archived bytes"
    if granule.checksum_algorithm and granule.checksum_value:
        try:
            actual_checksum = hashlib.new(
                _hashlib_name(granule.checksum_algorithm), payload
            ).hexdigest()
        except ValueError:
            return f"Unsupported CMR checksum algorithm: {granule.checksum_algorithm}"
        if actual_checksum.lower() != granule.checksum_value.lower():
            return "Downloaded payload checksum does not match CMR metadata"
    return None


def _hashlib_name(algorithm: str) -> str:
    return algorithm.lower().replace("-", "").replace("_", "")


def _parent_expected_id(
    product: ViirsL2Product, coverage_date: date, bbox: str, region: str
) -> str:
    return ":".join(
        (
            "viirs_l2_observability",
            "inventory",
            product.short_name,
            product.collection_version,
            coverage_date.isoformat(),
            _collection_scope_component(region, bbox),
        )
    )


def _granule_expected_id(
    product: ViirsL2Product, granule: ViirsL2Granule, bbox: str, region: str
) -> str:
    return ":".join(
        (
            "viirs_l2_observability",
            "granule",
            product.short_name,
            product.collection_version,
            granule.native_granule_id,
            "embedded-observability",
            _collection_scope_component(region, bbox),
        )
    )


def _scope_component(bbox: str) -> str:
    return hashlib.sha256(bbox.strip().encode("utf-8")).hexdigest()[:12]


def _collection_scope_component(region: str, bbox: str) -> str:
    return hashlib.sha256(f"{region.strip()}|{bbox.strip()}".encode("utf-8")).hexdigest()[:12]


def _day_bounds(coverage_date: date) -> tuple[datetime, datetime]:
    start = datetime.combine(coverage_date, time.min, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def _cmr_time(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _retrieved_at(value: datetime | None) -> datetime:
    resolved = value or datetime.now(timezone.utc)
    if resolved.tzinfo is None:
        raise ValueError("retrieved_at must include a UTC offset")
    return resolved.astimezone(timezone.utc)


def _is_terminal(record: CoverageRecord | None) -> bool:
    return record is not None and record.status in {
        CoverageStatus.COMPLETE,
        CoverageStatus.EMPTY_CONFIRMED,
    }


def _response_media_type(headers: Mapping[str, Any], fallback: str) -> str:
    value = headers.get("Content-Type")
    if not isinstance(value, str) or not value.strip():
        return fallback
    return value.split(";", maxsplit=1)[0].strip() or fallback


def _parse_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ViirsL2InventoryError(f"CMR granule has no {label}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ViirsL2InventoryError(f"CMR granule has an invalid {label}") from exc
    if parsed.tzinfo is None:
        raise ViirsL2InventoryError(f"CMR granule {label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _nonnegative_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ViirsL2InventoryError(f"{label} must be an integer") from exc
    if parsed < 0:
        raise ViirsL2InventoryError(f"{label} must not be negative")
    return parsed


def _optional_text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, dict) else None


def _mapping_value(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def _retrying_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session
