"""Inspect and record the hard 20 GB local-data budget."""

from __future__ import annotations

import argparse

from .storage_budget import (
    DEFAULT_POLICY_PATH,
    load_storage_budget,
    measure_storage_usage,
    write_storage_inventory,
)


def main(argv: list[str] | None = None) -> int:
    """Write a scored storage inventory and fail if the whole cap is exceeded."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--policy", default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--output")
    arguments = parser.parse_args(argv)
    policy = load_storage_budget(arguments.policy)
    output = write_storage_inventory(
        policy,
        arguments.data_root,
        output_path=arguments.output,
    )
    usage = measure_storage_usage(arguments.data_root)
    remaining = policy.whole_data_cap_bytes - usage.total_bytes
    print(
        f"Local data uses {usage.total_bytes:,} of {policy.whole_data_cap_bytes:,} bytes; "
        f"{max(0, remaining):,} bytes remain. Wrote {output}."
    )
    return 0 if usage.total_bytes <= policy.whole_data_cap_bytes else 1


if __name__ == "__main__":
    raise SystemExit(main())
