#!/usr/bin/env python3
"""Download the public datasets through MOABB's official cache mechanism."""

from __future__ import annotations

import argparse

from deepbench.config import DATASET_SPECS
from deepbench.datasets import download_moabb_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    moabb_datasets = tuple(name for name in DATASET_SPECS if name != "MI-OpenBCI")
    parser.add_argument("--datasets", nargs="+", choices=moabb_datasets, required=True)
    parser.add_argument("--subjects", nargs="+")
    args = parser.parse_args()
    for dataset in args.datasets:
        download_moabb_dataset(dataset, args.subjects)


if __name__ == "__main__":
    main()
