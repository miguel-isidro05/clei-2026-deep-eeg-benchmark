# Execution log

## 2026-10-11

1. Compared the MICCAI PDDM directory with public commit
   `6d7240564d46ff950288325bd9bdcbcf40747212`.
2. Confirmed byte identity for `pipeline.py`, `pipeline2.py` and `SGDM.py`.
3. Ran the original public DDPM on synthetic data: epoch loss `1.6467`, output
   shape `(2, 1, 64)`, all values finite.
4. Replaced late additive V12 conditioning with direct pre-stem
   concatenation, removed unused heads and froze the 1,000-step schedule.
5. Ran a four-cell MPS smoke across discriminative, diffusion-energy and
   hybrid paths in `/tmp/peterson-v12-final-smoke-20261011`.
6. The smoke completed all four cells. One-epoch class-collapse flags are
   retained as signed diagnostics and are not interpreted as final accuracy.
7. The 72 V12 tests and all 251 repository tests passed; Ruff and
   `git diff --check` passed.
8. `pip-audit` reported 40 known advisories in seven environment packages.
   Dependencies were not changed inside this scientific patch because doing so
   would alter the frozen numerical environment; remediation is a separate
   environment-validation task.
