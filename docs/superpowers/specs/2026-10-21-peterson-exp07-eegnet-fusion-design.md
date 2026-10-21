# Peterson Exp07 — backbone-agnostic positional window fusion

## Decision being tested

Exp05 and Exp06 reproduced a small gain from six learned, input-independent
window weights on FBCNet. Exp07 tests whether that effect is a reusable low-cost
MI decoding module or only an FBCNet-specific interaction. It does not open the
Peterson holdout and it does not claim confirmatory inference.

## Primary hypothesis

If learned positional window fusion captures reproducible MI timing, rather than
an FBCNet quirk, LPWF should improve uniform six-window aggregation by at least
0.5 percentage points in multiple EEGNet-family backbones. A replacement for e11
must additionally exceed e11 by at least 1.0 percentage point after internal
confirmation.

## Frozen scope

- Dataset/task: MI-OpenBCI, motor imagery versus rest.
- Protocol: within-session, five outer folds, trial-grouped overlap windows.
- Discovery subjects: S09, S03, S02, S10, S12, S08.
- Closed holdout: S07, S04, S05, S06.
- Screen seeds: 10, 11, 12.
- Confirmation seeds: 13, 14.
- Nested epoch selection: train-only inner split; refit from scratch on the
  complete outer-train partition for the selected epoch count.
- Maximum: 300 epochs, AdamW, cosine schedule, batch 64.

## Candidate pairs

Each backbone is evaluated twice with identical initialization and training:

1. EEGNet-4,2, temporal kernel 32: uniform versus LPWF.
2. EEGNet-8,2, temporal kernel 32: uniform versus LPWF.
3. EEGNet-8,2, temporal kernel 64: uniform versus LPWF.
4. EEG-TCNet: uniform versus LPWF.
5. EEGNeX-8,32: uniform versus LPWF.

LPWF adds exactly six logits, initialized to zero. Their softmax is therefore
identical to uniform aggregation before training. Exact FBCNet and e11 are
included as context/reference controls.

Literature provenance:

- EEGNet: Lawhern et al., 2018, DOI 10.1088/1741-2552/aace8c.
- EEG-TCNet: Ingolfsson et al., 2020, DOI 10.1109/SMC42975.2020.9283028.
- EEGNeX: Chen et al., 2024, DOI 10.1016/j.bspc.2023.105475.
- FBCNet: Mane et al., 2021, arXiv:2104.01233.

## Budget

- Screen: 12 configurations x 6 subjects x 3 seeds = 216 cells.
- Confirmation: two controls plus both members of at most two promoted pairs,
  across 6 subjects and 2 seeds = at most 72 cells.
- Maximum total: 288 cells.

## Screen gate

One backbone pair is eligible when LPWF versus its uniform control has:

- mean accuracy delta >= 0.5 pp;
- wins on at least 4/6 subjects;
- no class collapse;
- non-negative kappa delta;
- lower-quartile delta no worse than -0.5 pp;
- seed-standard-deviation increase no greater than 0.5 pp.

At most two eligible backbone pairs enter confirmation.

## Final promotion gate

The LPWF member must retain the within-pair gate over all five seeds and also,
relative to e11:

- improve mean accuracy by at least 1.0 pp;
- win on at least 4/6 subjects;
- have no class collapse;
- have non-negative kappa delta;
- preserve the lower quartile within 0.5 pp;
- increase seed standard deviation by no more than 0.5 pp.

Exact sign-flip p-values are descriptive only because discovery has six subjects.
The holdout remains closed regardless of the outcome.

## Stopping rule

- If an EEGNet-family LPWF candidate passes, freeze it for the single final
  holdout evaluation.
- If LPWF transfers to several backbones but none beats e11, retain e11 and
  describe LPWF as a general low-capacity aggregation result, not a new decoder.
- If the effect does not transfer, stop architecture search, retain e11 as an
  FBCNet-specific extension, and open the holdout once under the separate final
  evaluation protocol.

