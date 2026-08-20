"""Collect compact ETOPO terrain blocks for all FIRMS-context tiles in a date range."""

from __future__ import annotations

import argparse
from datetime import date

from .etopo_terrain import (
    DEFAULT_SOURCE_BLOCK_DEGREES,
    collect_etopo_terrain,
    context_tile_ids_from_detections,
    etopo_source_blocks,
)
from .forecast_tile_planning import iter_normalized_firms_detections
from .storage_budget import DEFAULT_POLICY_PATH, load_storage_budget


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=_parse_date, help="inclusive FIRMS UTC date")
    parser.add_argument("--end", required=True, type=_parse_date, help="inclusive FIRMS UTC date")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--block-degrees", type=float, default=DEFAULT_SOURCE_BLOCK_DEGREES)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.end < arguments.start:
        parser.error("--end must not be before --start")

    tile_ids = context_tile_ids_from_detections(
        iter_normalized_firms_detections(
            arguments.data_root,
            start_date=arguments.start,
            end_date=arguments.end,
        )
    )
    blocks = etopo_source_blocks(tile_ids, block_degrees=arguments.block_degrees)
    if arguments.dry_run:
        estimated_bytes = sum(block.cell_count * 6 + 1_048_576 for block in blocks)
        print(
            f"Planned {len(blocks):,} ETOPO source blocks for {len(tile_ids):,} FIRMS-context tiles; "
            f"conservative retained estimate: {estimated_bytes:,} bytes. No source requests were made."
        )
        return 0

    collection = collect_etopo_terrain(
        arguments.data_root,
        context_tile_ids=tile_ids,
        storage_budget=load_storage_budget(arguments.policy),
        block_degrees=arguments.block_degrees,
    )
    print(
        f"Processed {len(collection.blocks):,} ETOPO source blocks for {collection.context_tile_count:,} "
        f"FIRMS-context tiles; {collection.complete_count:,} complete, "
        f"{collection.skipped_count:,} already complete, "
        f"{collection.partial_or_failed_count:,} partial or failed."
    )
    return 0 if collection.partial_or_failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
