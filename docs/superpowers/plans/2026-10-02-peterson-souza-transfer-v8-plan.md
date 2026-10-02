# Peterson → Souza transfer v8 implementation plan

## 1. Windowed trial inference

Files: `src/deepbench/preprocessing.py`, `src/deepbench/evaluation.py`,
`tests/test_preprocessing.py`, `tests/test_protocols.py`.

- Add a deterministic trial-window expansion API returning windows and trial indices.
- Expand `overlap`, `nonoverlap`, `center_x6`, and `center_x2` consistently at test time.
- Average class probabilities by original trial and derive one prediction per trial.
- Preserve `full` as one input per trial.
- Test exact starts, mappings, class order, aggregation, and trial-level output counts.

Verify: focused preprocessing/protocol tests fail first, then pass.

## 2. Common transfer input space

Files: `src/deepbench/datasets.py`, `src/deepbench/config.py`, `tests/test_datasets.py`.

- Define the 15 common channels in Peterson order.
- Add a non-mutating recording alignment helper for transfer.
- Reject missing, reordered, or duplicate channel assumptions.
- Preserve the standalone Souza loader for historical v7 artifacts.

Verify: aligned Souza is `(trials, 15, 384)` with unchanged labels, runs, and data fingerprint.

## 3. Transferable model construction

Files: `src/deepbench/models.py`, `src/deepbench/transfer.py`, `tests/test_transfer.py`.

- Define the `final_layer` head boundary for all five models.
- Load only compatible backbone weights from Peterson.
- Reinitialize target heads deterministically.
- Implement scratch, linear-probe, and full-finetune parameter policies.
- Use optimizer parameter groups for a `0.1 ×` backbone LR and base head LR.
- Validate checkpoint schema, input shape, channel order, source task, seed, and hashes.

Verify: every model loads an identical backbone, a different head, and the exact intended
trainable parameter set.

## 4. Early stopping and refit isolation

Files: `src/deepbench/models.py`, `src/deepbench/transfer.py`, `tests/test_transfer.py`.

- Add predefined validation support and best-epoch extraction.
- Source: one whole Peterson subject for validation, rotated across seeds.
- Target within-session: 20% stratified inner validation from outer train.
- Target cross-session: one complete outer-training run as validation.
- Split trials before creating windows.
- Refit on complete outer train for the selected epoch and evaluate outer test once.

Verify: index-disjointness tests prove no outer-test or cross-trial leakage.

## 5. Source pretraining and checkpoint manifests

Files: `src/deepbench/transfer.py`, `scripts/pretrain_peterson.py`,
`tests/test_transfer_runner.py`.

- Pool Peterson recordings by source train/validation subjects.
- Apply train-fitted preprocessing and overlap windows.
- Train one source checkpoint per model/seed.
- Write atomic checkpoint and JSON manifest with scientific provenance.
- Resume only when checkpoint and manifest fingerprints match.

Verify: one-model/one-seed short source smoke and stale-checkpoint rejection.

## 6. Souza target transfer runner

Files: `src/deepbench/transfer.py`, `scripts/run_transfer.py`,
`tests/test_transfer_runner.py`.

- Run scratch, linear probe, and full fine-tuning for within-session and cross-session.
- Store trial-level predictions, window policy, selected epoch, trainable counts, source
checkpoint hash, split hashes, preprocessing reports, and histories.
- Add fold-level atomic resume.

Verify: synthetic and one-subject/one-model/one-seed short target smokes for all three arms.

## 7. Peterson v8 profile

Files: `src/deepbench/transfer_profile.py`, `scripts/run_transfer_profile.py`,
`tests/test_transfer_profile.py`.

- Keep v7 profile/audit backward compatible.
- Declare the separate v8 Peterson primary grid and Souza transfer grid.
- Exclude LOSO and the legacy standalone Souza augmentation/ICA blocks.
- Require exact five-seed completeness.

Verify: plan-only expected-cell tests and unexpected/missing-cell rejection.

## 8. ERD/ERS audit and transfer statistics

Files: `src/deepbench/erd_ers.py`, `src/deepbench/transfer_statistics.py`,
`scripts/audit_erd_ers.py`, `scripts/run_transfer_statistics.py`, and tests.

- Compute Peterson MI-rest mu/beta changes and Souza C3-C4 lateralization by subject/run.
- Generate paired transfer contrasts after seed averaging.
- Apply Holm correction and report effect sizes and intervals.
- Block inference on incomplete or mixed-profile cells.

Verify: deterministic synthetic signals produce expected ERD and lateralization direction.

## 9. Two-GPU orchestration

Files: `run_transfer_cayetano.sh`, `scripts/write_transfer_report.py`, shell-scope tests.

- Pretrain checkpoint shards on both GPUs and wait for both.
- Validate checkpoint completeness before target jobs.
- Run non-overlapping Peterson and Souza model/seed shards on both GPUs.
- Preserve durable logs, exit status, atomic resumability, and no LaTeX generation.
- Print exact setup, launch, monitoring, resume, and export commands.

Verify: `bash -n`, plan-only run, shard-disjointness test, and failure propagation test.

## 10. Final verification and publication branch

- Run the complete test suite and Ruff.
- Run CPU/MPS smokes with reduced epochs in a separate output directory.
- Review the diff for leakage, checkpoint staleness, hidden target tuning, and task-label reuse.
- Update technical runbooks only; do not edit manuscript/LaTeX.
- Commit implementation in reviewable units.
- Push `feat/peterson-souza-transfer-v8` to GitHub after all gates pass.
- Provide the exact Cayetano two-GPU command and expected artifacts.
