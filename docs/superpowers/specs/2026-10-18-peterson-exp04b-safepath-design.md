# Peterson Exp04b: SafePath discovery search

## Decision

Exp04b is a discovery-only extension between the rejected Exp04 joint model and
the frozen Exp05 holdout. It must never load Peterson holdout subjects. Its job
is to identify one architecture before the holdout is opened.

The primary model is SafePath-QCR. A fold-specific clean expert is trained and
then frozen. A separate robust expert learns from simulated channel corruption.
The final logits are

`z = z_clean + r(q) * delta_z`.

When the router declares an input clean, `r(q)` is exactly zero and the returned
logits must be bitwise equal to `z_clean`. This is the clean safety invariant.

## Fixed data scope

- Dataset: Peterson MI-OpenBCI only.
- Task: motor imagery versus rest.
- Protocol: within-session overlap.
- Discovery subjects: S09, S03, S02, S10, S12, S08.
- Holdout subjects S07, S04, S05, S06 remain inaccessible.
- Outer test indices from each search fold remain private.

## Search budget

| Wave | Design | Logical jobs |
| --- | --- | ---: |
| W1 | 24 isolated mechanisms, seed 0, six subjects | 144 |
| W2 | 12 router/fusion/objective interactions, seeds 0 and 1 | 144 |
| W3 | three promoted candidates and three neural controls, five seeds; one deterministic SCM-MDRM control | 186 |
| Total | Discovery only | 474 |

W4 is inference-only stress evaluation and does not add training jobs.

## Factors

W1 isolates clean anchor, robust dropout, clean-gradient protection, router,
fusion, tail objective, and corruption curriculum. W2 uses the fixed factorial
product of three promotable routers, two fusion rules, and two tail objectives.
W3 receives the top three promotable W2 candidates plus matched FBCNet, exact G2,
and the rejected joint Exp04 recipe as a negative control. SCM conditional
reconstruction plus MDRM is the deterministic classical control.

## Stress suite

The primary stress endpoints enumerate all 15 single-channel and all 105
two-channel missing masks. Secondary conditions cover flatline, attenuation,
Gaussian channel noise, clipping/impulses, and temporal dropout. Physical masks
are aggregated across folds before selecting the worst mask.

## Promotion rule

One primary candidate may advance only if it has zero class collapses, clean
damage no worse than 0.5 percentage points, at least 3 percentage points of mean
gain for both one- and two-channel missingness, gains in at least four of six
subjects, and a physical-mask floor no worse than matched FBCNet. W1 and W2 are
exploratory. W3 uses five seeds and Holm correction. No Exp04b result is a
confirmatory holdout result.

## Execution and provenance

Two CUDA processes own disjoint subject groups. Clean anchors are cached by
subject, seed, fold, architecture, preprocessing hash, and code hash. Every
cell, checkpoint, catalog, selection decision, environment, and manifest is
content addressed. A completed marker is written only after both shards and the
strict audit succeed.
