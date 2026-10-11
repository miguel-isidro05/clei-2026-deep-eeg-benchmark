# Comparability report

## Source identity

Public commit: `6d7240564d46ff950288325bd9bdcbcf40747212`.

| File | SHA-256 | Public/local relation |
| --- | --- | --- |
| `pipeline.py` | `e48a24c398df637b09f4f83338a2dc435870abf56097e4829978c3b37c4655fd` | byte-identical |
| `pipeline2.py` | `11b02b4a744eda76bf372ac5c5504a6e82973b0205e08b0a5155903da9d1ce06` | byte-identical |
| `SGDM.py` | `671500c2bba3f9228185502af91fdfabb73e8c9f1f53749ae9a20eded6cf8e8e` | byte-identical |
| local `ddpm_2a.py` | `62b9570f99d28dd224cace7d371e8c004b106c171039f5b8b795e9cd03c0f7f0` | local-only |

## What is comparable

- Full-resolution 1D temporal processing.
- Direct concatenation of observed/noisy signal and conditioning before the
  first convolution.
- Sinusoidal timestep representation and linear beta schedule.
- Noise-prediction mean squared error as the denoising term.

## What is not comparable

- PDDM reconstructs a missing channel; V12 classifies MI versus rest.
- PDDM conditions on observed EEG channels; V12 conditions on timestep and
  candidate class.
- PDDM evaluates reconstruction and may execute the reverse chain; V12 compares
  class-conditional denoising energies at sampled timesteps.
- Datasets, channel layouts, splits and outcome metrics differ.

Therefore PDDM supports the implementation mechanism but does not constitute a
performance baseline or proof that diffusion energy will improve Peterson.

Exp01 also must not be interpreted as a causal formulation ablation. Direct
conditioning expands the stem input and pure formulations omit unused heads,
so capacities differ. It is a model-selection screen; any later causal claim
requires parameter-matched controls or joint reporting of performance,
parameters and compute.
