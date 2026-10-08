#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from deepbench.peterson_export import create_publication_archive


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export only manifest-verified Peterson publication artifacts."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sidecar = create_publication_archive(args.results_dir, args.output)
    print(f"peterson_publication_export=OK archive={args.output} sha256={sidecar}")


if __name__ == "__main__":
    main()
