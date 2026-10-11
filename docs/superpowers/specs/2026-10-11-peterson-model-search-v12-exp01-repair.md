# V12 exp01 repair after the partial Cayetano run

## Evidence boundary

This repair uses only the 163 discovery-subject cells exported from commit
`2e55dd6`. No holdout subject or outer-test label was inspected. The partial
archive remains diagnostic evidence and must not be merged with the repaired
run because the executable code identity changes.

## Observed failures

1. The second shard stopped at `exp01-cfg-019/S10` because both validation
   folds predicted only class 0. Reconstructing the saved checkpoints produced
   accuracy 0.500, kappa 0.000 and AUROC 0.476. Class collapse is therefore a
   real model outcome, not a corrupt file or an auditor false positive.
2. The diffusion head injected timestep and class embeddings after all
   nonlinear backbone processing. The class-dependent output difference was
   consequently independent of the input signal, and diffusion-energy scores
   stayed almost constant around 0.5.
3. Training histories exposed only total loss, which prevented separating
   classification, denoising, ranking and consistency behavior.

## Frozen repair

- Inject class and timestep conditioning before the backbone sequence blocks,
  allowing nonlinear signal-condition interaction while keeping the same
  parameter budget and candidate catalog.
- Record validation class collapse as the signed quality flag
  `validation_class_collapse`; do not abort a shard or discard the cell.
- Require prediction evidence, recompute stored accuracy and kappa, and detect
  collapse separately in every fold so opposite collapsed classes cannot look
  stable after aggregation.
- Exclude any candidate with a collapsed discovery cell from promotion, while
  retaining it in the ranking and reporting its collapse rate.
- Store every loss component for both training and validation epochs.
- Expand the smoke manifest to four cells spanning discriminative,
  diffusion-energy and hybrid formulations, including a conformer-lite case.

## Hypotheses after the partial readout

- **Retain:** multiscale/filter-bank and smooth temporal stems may improve
  discovery accuracy. Their apparent advantage is provisional because only
  three subjects completed for most of these candidates.
- **Refine:** hybrid training may act as auxiliary denoising regularization.
  The current evidence does not show that energy averaging improves inference.
- **Retest once:** corrected conditional diffusion must produce signal-dependent
  class effects and non-degenerate energy scores. If its complete repaired run
  remains near chance or below useful discrimination, pure diffusion-energy is
  removed from the next search stage.
- **Do not expand:** width alone was inconsistent in the six-subject completed
  families, so no broader width sweep is justified yet.
- **Guardrail:** class-collapse rate is a stability endpoint and a hard
  promotion exclusion, not a hidden failed run.

## Required rerun

Run exp01 from an empty, newly named output directory. The 163 old cells cannot
be resumed because their code and model semantics differ. A complete run still
contains 216 discovery cells; only after its audit may exp02 be designed.
