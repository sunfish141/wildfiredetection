"""Archive the fixed CEC NALCMS 2020 v2 Canada and U.S. land-cover releases."""

from __future__ import annotations

import argparse
from pathlib import Path

from .nalcms_collection import NALCMS_RELEASES, collect_nalcms_land_cover
from .storage_budget import DEFAULT_POLICY_PATH, load_storage_budget


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--staging-directory", default="/tmp")
    parser.add_argument(
        "--release",
        choices=[release.key for release in NALCMS_RELEASES],
        action="append",
        dest="release_keys",
        help="archive only one named country release; repeat to select several",
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.dry_run:
        print(
            "Will archive two public NALCMS 2020 v2 source ZIPs (Canada and United States) "
            "after Content-Length quota admission. No source requests were made."
        )
        return 0
    releases = tuple(
        release
        for release in NALCMS_RELEASES
        if not arguments.release_keys or release.key in arguments.release_keys
    )
    collection = collect_nalcms_land_cover(
        arguments.data_root,
        storage_budget=load_storage_budget(arguments.policy),
        releases=releases,
        staging_directory=Path(arguments.staging_directory),
    )
    print(
        f"Processed {len(collection.releases)} NALCMS source releases; "
        f"{collection.complete_count} complete, {collection.skipped_count} already complete, "
        f"{collection.partial_or_failed_count} partial or failed."
    )
    return 0 if collection.partial_or_failed_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
