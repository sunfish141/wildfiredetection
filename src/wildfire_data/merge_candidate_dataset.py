"""Merge contiguous completed FIRMS/FEDS candidate-build manifests."""

from __future__ import annotations

import argparse
from datetime import date

from .candidate_dataset import CandidateDatasetError, merge_candidate_dataset_builds


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=_parse_date)
    parser.add_argument("--end", required=True, type=_parse_date)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--input-manifest", action="append", required=True)
    arguments = parser.parse_args(argv)
    try:
        manifest = merge_candidate_dataset_builds(
            arguments.data_root,
            input_manifests=arguments.input_manifest,
            start_date=arguments.start,
            end_date=arguments.end,
        )
    except CandidateDatasetError as exc:
        parser.error(str(exc))
    print(f"Published merged completed candidate-view manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
