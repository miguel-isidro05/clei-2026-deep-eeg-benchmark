# Peterson Exp04: QCR-SSNet discovery experiment

## Objective

Exp04 tests whether one compact network can preserve clean-signal accuracy while improving
classification when one or two low-cost EEG channels are unavailable. It runs only on the frozen
Peterson discovery partition. The holdout subjects remain inaccessible.

The model is promoted only if the complete predeclared configuration satisfies either the clean
superiority route or the robustness route against the exact FBCNet-V10 baseline. Ablations can
explain an effect, but they cannot become the proposed model.

## Prerequisite

Exp04 starts from a corrected Exp03 run. The Exp03 readout must reproduce two distinct operating
points:

- a clean candidate at least 1 percentage point above its control, with wins in at least four of
  six discovery subjects;
- a channel-dropout candidate whose channel-loss stress floor exceeds the clean candidate by at
  least 3 points and whose clean accuracy is no more than 10 points below the control.

The full Exp04 launcher verifies the Exp03 decision, manifest, strict audit, and current corrected
Exp03 code hash before it creates the Exp04 source manifest.

## Architecture

QCR-SSNet shares one frozen six-band filter bank between two compact FBCNet-style
spectro-spatial experts. One expert inherits the clean Exp03 width; the second inherits the
channel-dropout width. A train-fitted quality profile describes each channel with log RMS,
log-difference RMS, flatline rate, and extreme-value rate.

The full model adds two small components:

1. A residual channel gate applies bounded quality-dependent weights before the robust expert. It
   starts as the identity.
2. A quality router mixes the clean and robust logits. Its initial bias favors the clean expert so
   that the untrained model does not damage clean inference.

Training uses paired clean and deterministically corrupted views. The loss contains clean and
corrupted classification terms, Jensen-Shannon consistency, router supervision, and expert
specialization. All promotion metrics use the fixed last epoch, matching FBCNet-V10. The code
records a clean-primary validation objective for diagnostics but never uses its best epoch for
model comparison.

## Experimental matrix

The frozen full run contains ten candidates: clean and robust single-expert controls, three router
variants, the quality-gate and consistency ablations, the full QCR-SSNet model, and two loss
ablations. CSP+LDA and FBCNet-V10 are external baselines.

Each neural candidate and FBCNet-V10 uses six discovery subjects, five seeds, and five fixed
within-session folds. CSP+LDA is deterministic and runs once per subject. This produces 336 result
cells. Sliding-window overlap is the primary condition, matching the established Peterson
pipeline.

## Evaluation and promotion

Every cell stores clean validation predictions and exhaustive masks for one and two missing
channels: 15 single-channel masks and all 105 two-channel combinations. Promotion uses the worst
individual mask, while the readout also retains the mean across masks. The auditor recomputes all
metrics from predictions, validates source and
candidate signatures, rejects outer-test material, checks the discovery partition, and verifies
the data, split, environment, code, and Git identities.

The clean route requires:

- no collapsed cells;
- mean accuracy at least 1 point above FBCNet-V10;
- positive subject-level differences in at least four of six subjects;
- lower-quartile accuracy within 1 point of FBCNet-V10;
- mean kappa within 0.02 of FBCNet-V10.

The robustness route requires:

- no collapsed cells;
- clean accuracy no more than 0.5 points below FBCNet-V10;
- at least 3 points of improvement over FBCNet-V10 for both missing-channel conditions.

The analysis reports paired subject-level deltas, exact sign-flip p-values, seed variability,
stress performance, parameter count, and inference time. A full run never accesses the Peterson
holdout subjects and does not claim confirmation; a successful Exp04 result advances the frozen
model to the later confirmatory protocols.
