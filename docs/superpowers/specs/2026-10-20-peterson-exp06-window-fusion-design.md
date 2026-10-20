# Peterson Exp06: low-capacity temporal fusion after Exp05

## Decision inherited from Exp05

Exp05 promoted `exp05-e11`, a FBCNet whose six overlapping two-second
windows are combined with six trainable, input-independent softmax weights.
Across the five discovery seeds it improved the paired FBCNet control by
1.83 percentage points while adding six parameters. The improvement appeared
in five of six discovery participants and in all five seeds. Exp06 treats that
signed decision as its parent and does not reopen the Exp05 catalogue.

The holdout participants `S07,S04,S05,S06` remain inaccessible. Exp06 uses
only `S09,S03,S02,S10,S12,S08`, new screen seeds `5,6,7` and independent
confirmation seeds `8,9`.

## Research question

Can a compact, position-aware temporal fusion module improve the promoted
global-window weighting without introducing the high-capacity, content-aware
attention that failed in Exp05?

Primary metric: paired subject-level accuracy delta relative to the promoted
`e11` reference. FBCNet remains as an immutable contextual baseline.

Guardrails: Cohen's kappa, lower subject quartile, seed dispersion, class
collapse, trainable parameters and runtime.

## Literature-grounded design constraints

1. FBCNet supplies the neurophysiological filter-bank, spatial-filter and
   variance backbone (Mane et al., IEEE EMBC 2020, DOI
   `10.1109/EMBC44109.2020.9175874`).
2. Sliding-window prediction and temporal fusion are established for MI EEG;
   ATCNet combines sliding windows with compact attention and temporal
   convolution (Altaheri et al., IEEE TII 2023, DOI
   `10.1109/TII.2022.3197419`).
3. Cropped training can improve EEG ConvNet decoding, but trial-level splits
   must precede crop construction (Schirrmeister et al., Human Brain Mapping
   2017, DOI `10.1002/hbm.23730`).
4. Sparsemax provides an exact-sparse alternative to softmax attention
   (Martins and Astudillo, ICML 2016, arXiv `1602.02068`).
5. CSP initialization is retained as a train-only spatial prior because it was
   the second positive mechanism in Exp05. No test or holdout data may enter
   CSP fitting.

Exp05 provides a stronger project-specific constraint than generic model
fashion: content-aware window attention, multiscale temporal branches,
trainable Sinc filters and fine overlapping filter banks all degraded the
paired control. Exp06 therefore limits every candidate to a small residual or
structured change around `e11`.

## Frozen controls

- `exp06-fbc-control`: the exact Exp05 FBCNet control and ordinary crop
  averaging.
- `exp06-e11-reference`: the exact promoted grouped-window model with six
  softmax-normalized global logits initialized to uniform weights.

## Ten pre-specified candidates

| ID | Mechanism | Hypothesis | Added capacity |
|---|---|---|---:|
| `exp06-e01` | temperature-scaled global weights | Learned sharpness adapts how selectively positions are pooled. | 7 |
| `exp06-e02` | sparsemax global weights | Exact sparsity can suppress consistently uninformative crops. | 6 |
| `exp06-e03` | Gaussian positional kernel | A two-parameter contiguous temporal focus reduces variance. | 2 |
| `exp06-e04` | residual uniform blend | A learned scalar shrinks `e11` toward uniform pooling when data are weak. | 7 |
| `exp06-e05` | class-conditional global weights | MI and rest may occupy different informative temporal positions. | 12 |
| `exp06-e06` | residual temporal convolution | A zero-initialized depthwise width-three operator models adjacent-window evidence. | 18 or fewer |
| `exp06-e07` | CSP-init plus global weights | Train-only CSP supplies a spatial prior while `e11` supplies temporal reliability. | 6 |
| `exp06-e08` | center loss plus global weights | Compact class clusters can stabilize the promoted representation. | classifier feature centers |
| `exp06-e09` | quality weighting plus global weights | Train-fitted artifact weights reduce noisy-trial influence without changing inference. | 6 |
| `exp06-e10` | window dropout plus global weights | Dropping one crop during training discourages dependence on a single position. | 6 |

All parameter counts above are additions relative to the FBCNet control except
the center vectors. Exact counts are recorded from the executable model.

## Execution contract

Screen contains both controls and all ten candidates:

`12 configurations x 6 subjects x 3 seeds = 216 cells`.

At most three new candidates are promoted. Confirmation always reruns FBCNet,
the `e11` reference and the promoted candidates on seeds `8,9`, for at most:

`5 configurations x 6 subjects x 2 seeds = 60 cells`.

The primary promotion comparison is candidate versus `e11`, not versus
FBCNet. Screen eligibility requires zero collapses, delta at least 0.5 pp,
wins in at least four of six subjects, kappa loss no worse than 0.01, lower
quartile loss no worse than 0.5 pp and seed standard-deviation increase no
greater than 0.5 pp. Final preference requires a pooled five-seed gain of at
least 1 pp. P-values are descriptive because all participants remain in the
discovery cohort.

## Reproducibility and interpretation

The launcher requires the signed Exp05 `final-promotion.json` and verifies
that it selected `exp05-e11`. Every manifest records the parent decision hash,
revision, catalogue hash and cell identities. Every result records trial-level
predictions, nested split hashes, selected epoch, parameter count, runtime and
the learned temporal weights. GPU identity is metadata, never part of the
scientific configuration.

Exp06 can select one final recipe for a later, single opening of the holdout.
It cannot by itself support a population-level claim or establish literature
novelty.
