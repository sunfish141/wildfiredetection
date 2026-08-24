"""Export one completed candidate view as a self-contained upload directory."""

from __future__ import annotations

import argparse

from .candidate_dataset import CandidateDatasetError, export_candidate_dataset_release


def main(argv: list[str] | None = None) -> int:
    """Export exactly one completed candidate manifest; never glob artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", required=True, help="new release directory outside data/")
    parser.add_argument("--candidate-manifest", help="completed candidate-view manifest")
    arguments = parser.parse_args(argv)
    try:
        release = export_candidate_dataset_release(
            arguments.data_root,
            arguments.output,
            candidate_manifest=arguments.candidate_manifest,
        )
    except CandidateDatasetError as exc:
        parser.error(str(exc))
    print(
        f"Exported {release.candidate_row_count:,} candidate rows and "
        f"{release.unscored_positive_count:,} unscored positives to {release.directory}."
    )
    print(f"Release manifest: {release.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
