# V9 publication closure and causal EEGNet transfer study

## Status and objective

This specification starts from the completed, immutable V9 result bundle:

- 25 Peterson source checkpoints;
- 2,000 Peterson benchmark cells;
- 750 Peterson-to-Souza transfer cells;
- subjects `S02`–`S10`, `S12` for Peterson and `002`–`006` for Souza;
- no LOSO.

The work has two goals:

1. close the analysis gaps that prevent the V9 results from being reported correctly;
2. test whether EEGNet transfer is caused by useful Peterson pretraining or by generic initialization effects.

The existing V9 directory and archive remain read-only. New analyses and experiments use new output directories and profiles. No LaTeX or manuscript files are generated.

## Scientific claims allowed by V9

Peterson supports a primary low-cost MI-versus-rest benchmark with overlapping windows and trial-level aggregation. Souza supports a left-versus-right target study. V9 only supports an exploratory statement that EEGNet transfer is positive on average and strongly concentrated in subject `003`. It does not support a general transfer claim, a transformer advantage, a classical ERP mechanism, or a Souza sliding-window benefit.

The mechanistic interpretation concerns sensorimotor ERD/ERS and learned spectro-spatial representations. ERP terminology is excluded.

## Workstream A: publication analysis from existing V9 cells

### Input validation

The analysis command must reject an input directory unless it contains exactly:

- 25 source checkpoint manifests and 25 weight files;
- 2,000 Peterson cells from commit `573817d` and code hash `7121fb...`;
- 750 Souza transfer cells from the same commit and code hash;
- five seeds for every expected biological cell;
- no unknown subject, model, condition, protocol, ICA policy, or transfer arm;
- matching stored metrics when recomputed from predictions.

The source checkpoints from `4645dd8` are accepted only when their state hashes and the restricted-loader compatibility rules pass.

### Peterson statistics

Generate official V9 tables without retraining:

- descriptive accuracy, kappa, F1 macro, AUC, between-subject SD, and within-subject seed SD;
- ten paired model comparisons per primary protocol after averaging seeds within subject;
- sliding-window contrasts against full, non-overlap, `center_x2`, and `center_x6` where available;
- ICA-kurtosis sensitivity against `ica-none`;
- subject-level values for every contrast;
- class-collapse flags when the positive prediction rate is below 5% or above 95%.

The biological unit is the subject. Model comparisons follow the frozen Peterson plan: two-sided paired Wilcoxon tests with Holm correction within each declared family. Descriptive exact sign-flip p-values may be included as a sensitivity analysis but cannot replace the frozen test silently.

### Souza transfer statistics

Retain the existing exact subject-level contrasts and add:

- balanced accuracy;
- Brier score;
- expected calibration error with a fixed ten-bin policy;
- leave-one-subject-out influence diagnostics, clearly labeled descriptive rather than independent replications;
- per-run results for cross-session cells;
- an explicit `subject_003_removed` summary.

Bootstrap intervals over five subjects remain descriptive. A percentile interval excluding zero must not be labeled statistically significant when the exact paired test does not reject.

### Output

Write a self-contained `publication_analysis/` directory containing CSV and JSON files, a manifest with hashes of every input cell and analysis source file, and a machine-readable claim-status file with `supported`, `exploratory`, or `unsupported` labels. Do not write tables directly into a manuscript.

## Workstream B: latency correction

The current 256-sample measurement is a model-forward latency per window. The fields `amortized_median_ms_per_trial` and `individual_latency_ms` must not describe it as an original trial.

The replacement profiler emits two scopes:

1. `window_forward`: one 256-sample model input, preserving batch 1 and batch 64 results;
2. `trial_window_pipeline`: one original trial, six deterministic windows, GPU forward passes, softmax, and mean-probability aggregation.

The second scope excludes acquisition and signal filtering but includes window extraction and trial aggregation. It reports median and p95 for Peterson 512-sample trials and Souza 384-sample trials. Every row contains `measurement_unit`, `windows_per_trial`, `scope`, input shape, hardware, warm-up, iterations, dtype, and synchronization policy. Legacy field names are removed from the new schema rather than reinterpreted.

## Workstream C: causal EEGNet transfer

### Scope

Only EEGNet is included initially. V9 already shows no general transfer signal for the other four architectures. Expanding all controls to all models before establishing causality would spend compute without increasing the biological sample size.

The target protocols remain `within_session` and `cross_session`. Sliding overlap remains fixed. Outer and inner partitions are identical across every arm and seed. Subjects, channels, sampling rate, preprocessing, window starts, epoch selection, target head reset, and trial aggregation remain frozen from V9.

### Source conditions

Train five source states for each required condition:

1. `supervised_true`: Peterson MI-versus-rest labels unchanged;
2. `supervised_permuted`: Peterson labels permuted independently within each source subject using a stored deterministic permutation seed;
3. `random`: a seeded EEGNet backbone that receives no Peterson optimization.

The true and permuted conditions use identical source examples, validation-subject rotation, selected-epoch procedure, refit on all Peterson subjects, optimizer, batches, and compute budget. The permutation is applied at the original-trial level before windows are created. Its index hash and post-permutation class counts are stored.

Self-supervised Peterson pretraining is a gated extension, not part of the first causal run. It requires a separate reviewed objective and augmentation specification because an improvised contrastive loss would create a new tuning space.

### Target arms

Run these six arms:

- `scratch`;
- `random_linear_probe`;
- `supervised_true_linear_probe`;
- `supervised_true_full_finetune`;
- `supervised_permuted_linear_probe`;
- `supervised_permuted_full_finetune`.

All arms create a fresh binary target head. Linear probes freeze the backbone. Full fine-tuning uses the V9 differential learning-rate policy. Scratch is rerun under the causal profile so every comparison shares the same code and manifests.

The complete first causal grid contains 300 target cells:

`5 subjects × 5 seeds × 2 protocols × 6 arms`.

### Primary causal contrasts

The predeclared comparisons are:

- true full fine-tune minus scratch;
- true linear probe minus scratch;
- true full fine-tune minus permuted full fine-tune;
- true linear probe minus permuted linear probe;
- true linear probe minus random linear probe;
- permuted linear probe minus random linear probe.

Seeds are averaged within subject before inference. Holm correction covers the twelve protocol-specific contrasts. Report subject values, mean and median paired differences, exact sign-flip p, paired rank-biserial effect, and descriptive intervals.

### Causal success gate

Useful Peterson supervision is supported only if, for at least one true-label arm:

- the mean improvement over scratch is at least 5 percentage points;
- at least four of five subjects improve;
- the direction is compatible in both protocols;
- the arm outperforms its permuted-label counterpart;
- the leave-003-out mean remains positive;
- kappa does not materially decline.

If the gain remains confined to `003`, the result is reported as subject-specific transfer. It does not unlock broad architecture claims.

## Conditional extensions

These experiments are implemented or launched only after the causal success gate passes:

1. gradual unfreezing and L2-SP for EEGNet;
2. target-data curves at 10%, 25%, 50%, and 100%;
3. mu, beta, and mu+beta inputs;
4. C3/Cz/C4 versus all 15 common channels;
5. outer-train-only run normalization for cross-session adaptation;
6. a separately specified self-supervised Peterson objective.

Target-data subsets are nested, deterministic, and stratified within the available run/class structure. The outer test is never used to choose a fraction, band, channel set, regularization strength, or adaptation method.

## Reproducibility and resume policy

Every source state and target cell records:

- profile version;
- git revision and scientific-code hash;
- environment hash and versions;
- source condition and state hash;
- data hashes;
- subject, seed, protocol, arm, and fold hashes;
- window policy;
- epoch-selection history;
- trainable parameter names and count;
- predictions and probabilities at original-trial level.

Atomic writes and strict stale-artifact rejection are mandatory. Resume skips an artifact only after validating the complete expected identity. A new two-GPU launcher shards source and target jobs without overlap, writes durable logs, and runs completeness, statistics, latency, and claim-status checks after both workers succeed.

## Validation gates

Before Cayetano commands are released:

1. tests fail first for every new behavior;
2. unit tests cover within-subject permutation, unchanged class balance, fixed splits, source-state identities, head reset, frozen parameters, causal-grid completeness, calibration metrics, collapse flags, and trial latency scope;
3. a synthetic test proves that true and permuted conditions differ only in labels;
4. a one-subject, one-seed CPU/MPS smoke completes all six target arms with reduced epochs;
5. analysis of the immutable V9 directory succeeds and reproduces the audited headline values;
6. the full test suite, formatting checks, shell syntax checks, and a code review pass;
7. `git diff` confirms that no dataset, result archive, manuscript, or LaTeX file was modified.

## Explicit exclusions

- LOSO;
- external datasets;
- redownloading Peterson or Souza;
- standalone Souza augmentation or ICA grids;
- treating seeds, folds, runs, or windows as biological replicates;
- selecting hyperparameters from outer-test results;
- manuscript or LaTeX edits;
- overwriting V8 or V9 outputs;
- pushing or publishing without explicit user approval.

## Acceptance criteria

The implementation is ready when it can produce:

- publication-ready machine-readable Peterson and Souza analyses from the existing V9 bundle;
- correctly scoped window and full-trial latency artifacts;
- a resumable, deterministic 300-cell EEGNet causal grid on two GPUs;
- strict completeness and provenance reports;
- commands for Cayetano that require only the two existing dataset directories and a new output directory.
