# Peterson → Souza transfer benchmark v8

## Objective

Build a journal-grade, leakage-safe benchmark with two distinct roles:

1. **MI-OpenBCI/Peterson** is the primary low-cost motor-imagery benchmark and the supervised source dataset.
2. **Souza2023** is used only as the target of a cross-dataset, cross-task transfer study from Peterson. The previous standalone Souza augmentation/ICA matrix is not repeated.

LOSO remains outside v8 and will be added only in a later frozen phase.

## Scientific interpretation

Peterson labels motor imagery versus rest. Souza labels left-hand versus right-hand motor imagery. The source classifier head therefore has no valid target-label interpretation and must always be reinitialized.

The physiological audit concerns **event-related desynchronization/synchronization (ERD/ERS)** in the mu and beta bands, not a classical stimulus-locked ERP. It is a diagnostic analysis and does not select training hyperparameters.

The transfer question is:

> Does supervised pretraining on Peterson learn low-cost EEG representations that improve held-out left-versus-right decoding in Souza compared with the same architecture trained from scratch under identical target data, channels, preprocessing, windows, validation, and compute policy?

## Datasets and common input space

### Peterson source and primary benchmark

- Subjects: `S02`–`S10` and `S12`.
- Task: motor imagery versus rest.
- Channels: the existing 15-channel `MI_CHANNELS` order.
- Sampling rate: 128 Hz.
- Primary protocols before LOSO: `within_split` and `within_session`.

### Souza target

- Subjects: `002`–`006`; the attachment labeled 001 is a duplicate of 004 and remains excluded.
- Task: left hand versus right hand.
- Target evaluation protocols: `within_session` and `cross_session`.
- Souza is aligned to the exact 15 Peterson channels in the Peterson order; `Fp1` is excluded.
- Sampling rate: 128 Hz.

The existing 16-channel standalone Souza v7 cells are not valid scratch controls for transfer because their input space and inference policy differ.

## Sliding-window primary regime

- Window length: 2 s = 256 samples.
- Six overlapping windows per training trial.
- Window start positions span the complete available trial using the existing deterministic policy.
- The trial split is created before window extraction.
- Test trials are expanded into the same six windows.
- Window probabilities are averaged to produce exactly one score and one prediction per original trial.
- Accuracy is the primary endpoint; Cohen's kappa is secondary.

The no-sliding full trial is a Peterson ablation. `center_x6` remains a compute-matched control for the effect of temporal diversity. The legacy policy that tested only one central crop is not used for the v8 primary result.

## Peterson primary experiments

For all five models and seeds 0–4:

- models: EEGNet, FBCNet, ShallowConvNet, EEGConformer, EEGInceptionMI;
- protocols: `within_split`, `within_session`;
- primary condition: sliding overlap with trial-level probability aggregation in both protocols;
- full-trial no-window ablation in both protocols;
- `center_x2`, `nonoverlap`, and `center_x6` controls in `within_split`;
- primary preprocessing: no ICA;
- ICA-by-kurtosis sensitivity on the primary overlap condition in `within_split`;
- signal-quality, ERD/ERS, seed-variability, and 256-sample model-latency audits.

This preserves every relevant Peterson experiment from the existing profile while changing the
primary regime from full trial to sliding. The classical CSP+LDA check remains a non-transferable
Peterson reference and is evaluated on full trials and the same two-second window support.

Peterson primary evaluation models are distinct from source-pretraining checkpoints. No performance claim is made from evaluating a source checkpoint on trials used to fit it.

## Supervised source pretraining

For each architecture and seed:

1. Pool Peterson subjects in the common 15-channel, 256-sample sliding representation.
2. Reserve one complete Peterson subject for validation, rotating deterministically through
   `S02`–`S06` for seeds 0–4.
3. Fit preprocessing statistics only on source-training subjects.
4. Train for at most 300 epochs, with a minimum of 30 epochs, patience 50, and improvement
   threshold `1e-4` on source-validation loss; restore the best checkpoint.
5. Store checkpoint weights, source train/validation subjects, epoch, recipe, data hash, code hash, environment hash, channel order, sampling rate, window policy, and seed.

The held source subject is used only for checkpoint selection. The target Souza data are never used during source pretraining.

## Souza transfer arms

Every outer target fold runs the following three arms with identical inputs and target partitions:

### Scratch control

- Random initialization.
- Same 15 channels, windows, aggregation, inner validation, training budget, and target metrics as transfer arms.
- This is the only standalone Souza control retained in v8.

### Linear probe

- Load the Peterson backbone.
- Reinitialize the complete `final_layer` hierarchy.
- Freeze all backbone parameters.
- Train only the target head.

### Full fine-tuning

- Load the same Peterson backbone.
- Reinitialize the complete `final_layer` hierarchy.
- Initially expose all parameters to optimization.
- Use a lower backbone learning rate than the new head.
- Select the stopping epoch only from target inner validation.

The five architectures expose their target head under `final_layer`, including FBCNet's parametrized linear layer. Tests must prove that the backbone is loaded exactly, the old head is not retained, and the intended parameter set is trainable in each arm.

## Nested target validation

For each Souza outer fold:

- construct the outer train/test split first;
- for `within_session`, derive a fixed 20% stratified inner validation split only from the
  outer-training trials;
- for `cross_session`, reserve one complete outer-training run as inner validation using a fixed
  split independent of optimization seed;
- keep complete original trials together before generating windows;
- fit filtering, normalization, early stopping, and any learning-rate decision without outer-test data;
- select a stopping epoch with a maximum of 300 epochs, minimum 30, patience 40, and threshold
  `1e-4` on inner-validation loss;
- retrain on the complete outer-training partition for the selected number of epochs, refitting
  preprocessing only on that complete outer-training partition, then evaluate the outer test once.

Scratch and linear-probe heads use the architecture's predeclared base learning rate. Full
fine-tuning uses that base rate for the new head and `0.1 ×` the base rate for the transferred
backbone. No learning-rate grid is selected from target results.

No target hyperparameter may be selected from v7 test outcomes. The v8 study is reported as a prospective revised analysis motivated by v7, not as a preregistered analysis predating v7.

## ERD/ERS physiological audit

Generate a diagnostic artifact independent of model selection:

- mu band: 8–13 Hz;
- beta band: 13–30 Hz;
- Peterson: MI-minus-rest log-power change over C3, Cz, and C4;
- Souza: left/right C3–C4 lateralization and contralateral-minus-ipsilateral indices;
- summarize by subject and run;
- keep raw trial counts and confidence intervals;
- do not exclude subjects or tune windows based on this audit.

This audit tests whether the expected sensorimotor structure is present. It does not guarantee classifier transfer and must remain separate from model-selection data flow.

## Hypotheses

### Primary transfer hypothesis

Full fine-tuning from Peterson increases Souza subject-level accuracy relative to scratch under the same target protocol.

### Secondary hypotheses

- Linear probing above scratch indicates directly reusable source representations.
- Full fine-tuning above linear probing indicates that target adaptation is necessary because source and target labels differ.
- Higher-capacity EEGConformer and EEGInceptionMI may benefit more from pretraining, but model-specific effects are secondary and multiplicity-corrected.
- Transfer may improve within-session calibration without improving cross-session robustness.

## Statistical plan

- Average seeds within subject before inferential comparison.
- Subject is the biological unit; seeds, windows, folds, and runs do not inflate `n`.
- Compare `full_finetune - scratch`, `linear_probe - scratch`, and `full_finetune - linear_probe` within dataset, protocol, metric, and model.
- Apply Holm correction across the predeclared strategy/model family.
- Report mean paired difference, 95% interval, median difference, paired rank-biserial effect, and all subject-level values.
- With five Souza subjects, exact two-sided Wilcoxon cannot achieve `p < 0.05`; interpret effect size, interval, direction consistency, and protocol replication rather than claiming significance that the design cannot support.

### Practical success criterion

Transfer is practically promising only if full fine-tuning:

- improves mean accuracy by at least 0.05 over scratch in at least one target protocol;
- has a positive paired difference in at least four of five subjects;
- does not produce a material kappa degradation;
- shows a compatible direction in the other target protocol.

Failure to meet the criterion is retained as a scientifically interpretable negative transfer result. Further test-driven tuning stops at that point.

## Reproducibility and artifacts

The transfer output directory contains:

- `pretrain/`: one source checkpoint and manifest per model/seed;
- `cells/`: one target result per model/seed/subject/protocol/arm;
- `fold_cache/`: resumable target fold outputs;
- `statistics/`: completeness, descriptive, paired transfer, and seed-variability tables;
- `quality/erd_ers/`: physiological audit tables;
- `manifests/`: exact expected grid and environment/profile hashes;
- `logs/`: durable two-GPU logs;
- no manuscript or LaTeX generation.

## Two-GPU execution

The run has dependency-aware stages:

1. Pretrain model/seed checkpoints sharded deterministically across `cuda:0` and `cuda:1`.
2. Verify every required source checkpoint and manifest.
3. Run Peterson primary cells on both GPUs using non-overlapping model/seed shards.
4. Run Souza transfer cells on both GPUs using the verified checkpoints and non-overlapping model/seed shards.
5. Merge only through atomic result files and run a strict completeness/integrity audit.
6. Generate statistics and ERD/ERS diagnostics; do not generate LaTeX.

The launcher must fail on a missing checkpoint, stale fingerprint, duplicate cell, unexpected cell, incomplete five-seed grid, channel mismatch, or test-selected configuration.

## Validation gates before Cayetano

1. Unit tests for common-channel alignment, deterministic window extraction, trial-level probability aggregation, head reset, freezing, differential learning rates, nested split isolation, and checkpoint validation.
2. Synthetic transfer test proving backbone load and head replacement.
3. One-model/one-seed/one-subject CPU or MPS smoke for source pretraining and each target arm.
4. Resume smoke proving that completed source and target artifacts are skipped only when fingerprints match.
5. Full test suite, lint, and shell syntax checks.
6. Code review focused on leakage, stale checkpoint reuse, seed semantics, and target-test isolation.

Only after all gates pass is the two-GPU command released.

## Explicit exclusions

- LOSO in v8.
- Standalone Souza augmentation and ICA matrices from v7.
- Reuse of the Peterson binary output head.
- Self-supervised pretraining; it is a possible later journal extension after supervised transfer is measured.
- Research-grade external datasets.
- Test-driven searches for windows, bands, learning rates, or stopping epochs.
- Manuscript or LaTeX modification.
