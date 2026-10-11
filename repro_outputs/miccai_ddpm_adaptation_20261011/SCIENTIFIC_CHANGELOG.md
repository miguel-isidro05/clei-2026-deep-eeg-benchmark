# Scientific changelog

- Changed diffusion/hybrid conditioning from a late additive hidden-state
  offset to direct concatenation before the first temporal convolution.
- Changed the frozen linear schedule from 100 to 1,000 steps. With the same
  beta limits, terminal cumulative signal falls from approximately `0.364` to
  `4.04e-5`, covering nearly pure-noise states.
- Preserved the four-sample energy estimate (`inference_k=4`); no reverse DDPM
  sampling loop was added.
- Removed heads that are mathematically unused by pure discriminative and pure
  diffusion-energy candidates.
- Preserved dataset, task, discovery subjects, hidden holdout, folds, seed,
  windows, metrics, optimizer budget, promotion criteria and catalog size.
- No paper-level effectiveness claim is made until the new 216-cell run passes
  its auditor.
