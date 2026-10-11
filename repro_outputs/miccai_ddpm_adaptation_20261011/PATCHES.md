# Patch record

- Integration branch: `exp/peterson-model-search-v12`.
- Base revision: `c650259d866c86189ea4592176e2920794e559ab`.
- Risk: medium; the diffusion model semantics and frozen schedule change, so a
  new result directory is mandatory.
- Scope: V12 model construction, objective defaults, configuration contract,
  tests and scientific provenance documentation.
- Explicit exclusion: no reconstruction code, external data, reverse sampler
  or MICCAI metric was imported.

The existing dedicated experiment branch is retained instead of creating a
second reproduction branch because it is the approved V12 integration line.
The change remains reversible as a single commit.
