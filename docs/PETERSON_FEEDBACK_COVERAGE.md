# Peterson-only feedback coverage

This matrix is an execution gate, not a claim that a single dataset removes every review
limitation.

| Review issue | V10 evidence | Status after a complete run |
|---|---|---|
| Repeated optimization | Five fixed seeds for every deep-model cell; deterministic CSP is serialized on the same seed grid and identified as non-stochastic | Closed |
| Paired inference | Subject-level seed aggregation, paired Wilcoxon tests, Holm correction, 95% t intervals, and rank-biserial effect sizes | Closed |
| Unfair CSP comparison | CSP+LDA uses the same folds, preprocessing, train-only fitting, temporal inputs, window conditions, and trial-level probability aggregation | Closed |
| Sliding-window inflation | Primary overlap condition plus non-overlap and compute-matched repeated-center controls; inference remains one prediction per original trial | Closed |
| Generalization | Within-split, five-fold within-session, and LOSO are reported separately | Closed within Peterson; no external-dataset generalization claim |
| ICA ambiguity | No-ICA is primary; train-only FastICA with a frozen kurtosis rule is sensitivity-only and fully logged | Closed |
| Training provenance | Frozen model recipes, 300 epochs, code/data/environment fingerprints, fold histories, and exact expected-cell manifest | Closed |
| Latency overclaim | Deep-model output is labeled forward time per model input, excludes CSP cross-backend ranking, and explicitly excludes an online BCI pipeline | Closed by narrowing the claim |
| Research-grade hardware comparison | No research-grade dataset is present | Not addressed; the paper must not make a causal low-cost-vs-research-grade claim |
| Small sample / single dataset | Ten Peterson subjects | Inherent limitation; disclose it |
| Editorial wording and references | Experiment code does not edit the manuscript | Deferred until results are accepted |

The confirmatory question supported by this profile is: **under a frozen within-dataset
Peterson MI-vs-rest protocol, how do CSP+LDA and four deep decoders compare, and how much of
their performance is attributable to window augmentation?**
