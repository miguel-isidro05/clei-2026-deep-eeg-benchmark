# Texto base para responder a los revisores

No copie los campos entre corchetes hasta que `expected_cell_audit.csv` y `completeness.csv`
confirmen que no falta ninguna celda.

## Robustez multisemilla

> We thank the reviewer for identifying the limitation of the original single-seed experiments.
> We repeated every neural experiment with five independent optimization seeds (0–4), while
> keeping the train/test partitions and the ICA seed fixed. We now report subject-level mean ± SD
> and 95% confidence intervals, together with the within-subject SD across optimization seeds.
> Model comparisons use subjects, rather than folds, windows, or seeds, as the statistical units.
> The revised results show [INSERT RESULT AND CI].

## Pruebas pareadas y multiplicidad

> We replaced the previous omnibus/resampling analysis with predeclared two-sided paired Wilcoxon
> tests. The five models yield ten pairwise comparisons within each dataset, protocol, condition,
> ICA policy, and metric family. Holm correction is applied within each family. Accuracy is the
> primary metric and Cohen's kappa is secondary. Confidence intervals for subject means and paired
> mean differences use the Student t distribution; no bootstrap or Friedman test is used.

## Cross-session y MOABB

> We added leave-one-session-out evaluation on Zhou2020 and BNCI2014_001 (BCI Competition IV 2a).
> Zhou2020 provides a task-matched right-hand-versus-rest external dataset, whereas BNCI2014_001
> provides a canonical left-versus-right motor-imagery benchmark. We report these datasets
> separately and do not interpret their differences as causal effects of low-cost hardware.

## Identificacion de componentes ICA

> We found that the original description overstated what could be inferred from the available
> channels. Kurtosis does not identify ocular or muscular physiology by itself and, in our audit,
> it marked all components as candidates in a representative subject. We therefore removed ICA
> rejection from the primary analysis. We retain a clearly labeled exploratory sensitivity in
> which FastICA is fitted exclusively on the training partition, components with kurtosis greater
> than 10 are candidates, and at most the two highest-kurtosis components are removed. Each fold
> records convergence, iteration count, component kurtosis, candidates, and exclusions. We make
> no ocular/muscular labeling claim. The sensitivity difference was [INSERT PAIRED RESULT AND CI].

## Latencia

> We now report controlled model-inference latency after warm-up using identical device, input
> shape, batch size, and iteration count for all models. Batch-one median and 95th-percentile times
> represent individual model-forward latency. Batch-64 per-trial values are labeled amortized
> throughput. These measurements exclude EEG acquisition, window accumulation, transmission,
> preprocessing, interface, and actuation and are not described as end-to-end BCI response time.
