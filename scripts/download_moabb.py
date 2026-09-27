#!/usr/bin/env python3
"""Download the public datasets through MOABB's official cache mechanism."""

from __future__ import annotations

import argparse

from deepbench.datasets import download_moabb_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", nargs="+", choices=("AlexMI", "Zhou2020", "BNCI2014_001"), required=True
    )
    parser.add_argument("--subjects", nargs="+")
    args = parser.parse_args()
    for dataset in args.datasets:
        download_moabb_dataset(dataset, args.subjects)


if __name__ == "__main__":
    main()
