# Peterson journal V10 design

## Scope

The main workflow uses only MI-OpenBCI/Peterson (10 subjects, MI versus rest). Souza2023 and
transfer-learning code remain available for reproducibility of prior work but are disabled in
`setup.sh` and `run_cayetano.sh` unless explicitly invoked.

## Frozen grid

- Models: CSP+LDA, EEGNet, FBCNet, ShallowConvNet, and EEGConformer, matching the
  manuscript reviewed by NeurIPS.
- Seeds: 0–4. CSP is deterministic; its repeated cells preserve paired grid completeness but
  are marked `optimization_stochastic=false` and are averaged within subject before inference.
- Protocols: within-split, five-fold within-session, and LOSO.
- Primary temporal condition: six overlapping 2 s windows derived from each 4 s trial.
- Controls: one unaugmented 2 s center crop and full 4 s input; two non-overlapping windows
  versus two repeated center crops; and six overlapping windows versus six repeated center
  crops. Center versus overlap is the direct input-matched augmentation contrast.
- ICA: no ICA primary; train-only kurtosis-selected FastICA sensitivity on the primary
  within-split condition.
- Total: 3,250 subject/model/seed/protocol/condition/ICA cells.

Train/test splitting always precedes preprocessing and window generation. Window probabilities
are averaged to exactly one prediction per original test trial. Statistics use subjects—not
windows or seeds—as the inferential unit.

## Completion gate

The run is complete only when the expected-cell audit finds exactly 3,250 valid cells, every
deep fold has 300 logged epochs, every CSP fold carries the classical-estimator marker, the
five-seed grid is complete, and paired statistics are generated without exploratory markers.

## Claim boundary

This profile supports a Peterson within-dataset decoder and augmentation benchmark. It does not
support a causal hardware-quality comparison or broad cross-dataset generalization. The ten-
subject, single-dataset limitation remains explicit.
