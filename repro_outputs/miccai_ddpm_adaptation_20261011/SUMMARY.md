# MICCAI DDPM adaptation summary

The audited PDDM implementation is useful as a mechanism reference, not as a
drop-in model. V12 now feeds timestep and class conditioning alongside the
noisy EEG before the first temporal convolution and freezes a 1,000-step linear
noise schedule. The Peterson objective remains MI-versus-rest classification
through discriminative, diffusion-energy and hybrid formulations.

The candidate grid remains 36 configurations and 216 discovery cells. The old
163-cell partial run is not resumable or poolable because model semantics and
the diffusion schedule changed. A new empty output directory is required.

Local MPS smoke evidence covers four cells across both shards and the three
formulations. It verifies execution and artifact contracts only; it is not
scientific performance evidence.
