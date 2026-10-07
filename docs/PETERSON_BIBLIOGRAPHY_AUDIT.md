# Peterson bibliography audit

This is a focused source audit for the Peterson-only benchmark. It does not edit the manuscript.

## Verified primary method sources

| Method or analysis | Primary source | Persistent identifier | Audit decision |
|---|---|---|---|
| MI-OpenBCI dataset | Peterson et al., *Data in Brief* 42, 2022, 108225 | DOI `10.1016/j.dib.2022.108225` | Correct dataset source; add article number and DOI. |
| MOABB framework | Jayaram and Barachant, *Journal of Neural Engineering* 15(6), 2018, 066011 | DOI `10.1088/1741-2552/aadea0` | Correct benchmarking source; add article number and DOI. |
| EEGNet | Lawhern et al., *Journal of Neural Engineering* 15(5), 2018 | DOI `10.1088/1741-2552/aace8c` | Correct method family; add the DOI to the final bibliography. |
| FBCNet | Mane et al., IEEE EMBC 2020, pp. 2950–2953 | DOI `10.1109/EMBC44109.2020.9175874` | The current conference citation is the peer-reviewed primary source and is preferable to citing only arXiv:2104.01233. |
| ShallowConvNet | Schirrmeister et al., *Human Brain Mapping* 38(11), 2017 | DOI `10.1002/hbm.23730` | Correct primary architecture source; add the DOI. |
| EEG Conformer | Song et al., *IEEE TNSRE* 31, 2023, pp. 710–719 | DOI `10.1109/TNSRE.2022.3230250` | Correct primary architecture source; add pages and DOI. |
| Holm correction | Holm, *Scandinavian Journal of Statistics* 6(2), 1979, pp. 65–70 | Stable identifier `10.2307/4615733` | Correct source for familywise multiplicity control. |
| Brier score | Brier, *Monthly Weather Review* 78, 1950, pp. 1–3 | DOI `10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2` | Required only if the calibration diagnostic is discussed. |

## Corrections and required additions before manuscript submission

1. The current bibliography entries for EEGNet, FBCNet, ShallowConvNet, and EEG Conformer omit
   their persistent identifiers. Add the identifiers above during the later manuscript pass.
2. Add the primary CSP+LDA source used to motivate the classical baseline; the current revision
   has no CSP citation despite including CSP+LDA as one of five benchmarked decoders.
3. Add the original Wilcoxon signed-rank source or a modern statistical reference because the
   confirmatory analysis depends on that test. The current revision cites Holm but not Wilcoxon.
4. Cite a source for Cohen's kappa if the metric is formally defined rather than merely named.
5. Remove EEG-Inception, Zhou2020, and Tavakolan references if they are no longer discussed in the
   Peterson-only manuscript. A bibliography should follow the final text, not preserve abandoned
   experiments.
6. Keep Friedman explicitly exploratory in any result text. The frozen confirmatory family remains
   participant-level two-sided Wilcoxon with Holm correction.

## Claim discipline

- Architecture papers justify model definitions; their reported accuracies are not directly
  comparable with MI-OpenBCI because datasets, tasks, preprocessing, and validation differ.
- The Peterson study supports a within-dataset low-cost EEG decoding benchmark only.
- Neither the method citations nor the current results support a causal low-cost-versus-research-
  grade hardware claim or external-dataset generalization.
