# Security notes

This repository is an offline research pipeline. It does not expose a network service and does not
load external checkpoints. Do not use `torch.load` or equivalent deserialization on untrusted
`.pt`, `.pth`, pickle, NumPy, MAT, or result files.

The verified environment intentionally pins the scientific stack used by the tests. A dependency
audit on 2026-09-27 reported advisories affecting Torch 2.12.1 and indirect/environment packages.
The published code does not exercise the untrusted-checkpoint path associated with the Torch risk.
Before upgrading to Torch 2.13 or later, rerun the architecture, determinism, smoke, and statistical
tests on the target CUDA machine and record the new code/environment fingerprint. Do not silently
mix upgraded results with the pinned experiment cells.

Report suspected vulnerabilities privately to the repository owner. Do not attach EEG datasets,
credentials, tokens, or identifiable participant data to a public issue.
