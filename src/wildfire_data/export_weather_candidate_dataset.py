"""Export one completed weather-bearing candidate view."""

from __future__ import annotations

import argparse

from .weather_candidate_dataset import (
    WeatherCandidateDatasetError,
    export_weather_candidate_dataset_release,
)


def main(argv: list[str] | None = None) -> int:
    """Create a new upload directory from one completed weather manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--weather-candidate-manifest", required=True)
    arguments = parser.parse_args(argv)
    try:
        release = export_weather_candidate_dataset_release(
            arguments.data_root,
            arguments.output,
            weather_candidate_manifest=arguments.weather_candidate_manifest,
        )
    except WeatherCandidateDatasetError as exc:
        parser.error(str(exc))
    print(f"Exported {release.candidate_row_count:,} weather-bearing candidate rows to {release.directory}.")
    print(f"Release manifest: {release.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
