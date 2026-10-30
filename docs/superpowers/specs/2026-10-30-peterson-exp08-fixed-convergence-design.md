# Peterson Exp08 — fixed-convergence repair

## Purpose

Exp08 is an isolated repair of Exp07. Exp07 remains immutable evidence that grouped
end-to-end training with validation-selected epochs undertrained the deep backbones.
Exp08 tests the same five architecture pairs after converging each backbone on the
individual temporal crops.

## Hypothesis

Training EEGNet, EEG-TCNet, and EEGNeX crop-wise for exactly 300 epochs, then freezing
the backbone before fitting the six LPWF weights, will remove early-collapse failures
and isolate the contribution of LPWF from backbone convergence.

## Scope

- Dataset: Peterson MI-OpenBCI only.
- Task: motor imagery versus rest.
- Protocol: within-session, overlapping six-window condition.
- Discovery subjects: S09, S03, S02, S10, S12, S08.
- Holdout subjects: S07, S04, S05, S06; never accessed.
- Screen seeds: 15, 16, 17.
- Confirmation seeds: 18, 19.
- Controls: unchanged FBCNet and e11.
- Backbones: the five paired families used by Exp07.

## Training contract

For every subject, seed, outer fold, and backbone family:

1. Split trials before generating crops.
2. Fit preprocessing only on outer train.
3. Generate six crops per trial.
4. Train one backbone crop-wise for 300 fixed epochs using AdamW, learning rate
   6.25e-4, weight decay 1e-4, batch size 64, and a 300-epoch cosine schedule.
5. Do not call early stopping or epoch selection.
6. Freeze all backbone parameters and buffers.
7. Evaluate uniform averaging and fit a six-parameter LPWF head from the exact same
   backbone state. LPWF sees outer train only.
8. Evaluate outer test once.

The auditor requires the uniform and LPWF cells to report the same backbone hash for
each subject, seed, fold, and family. It also verifies that the hash is unchanged by
LPWF fitting.

## Decision gates

LPWF must improve its uniform pair by at least 0.5 percentage points, win in at least
four of six subjects, avoid kappa and lower-quartile regressions beyond their frozen
guardrails, avoid increased seed instability, and have zero collapsed folds. At most
two families advance. Final promotion additionally requires at least one percentage
point over e11. Otherwise the signed outcome is `retain_e11`.

## e11 reference

e11 is not an eleventh neural backbone. It is the exact FBCNet control plus six global
trainable positional logits, one per temporal crop. A softmax converts them into six
weights used to combine crop logits. The weights are identical for every trial: e11 is
positional fusion, not content-dependent attention. It adds six parameters to the
4,994-parameter FBCNet, for 5,000 parameters total.

## Outputs and stopping rule

The full screen contains 216 cells. Confirmation contains 24–72 cells depending on
the signed screening decision. Exp08 is the single repair authorized by the Exp07
audit; no further architecture search is implied by a negative result.

