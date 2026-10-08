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
| CSP | Ramoser, Müller-Gerking, and Pfurtscheller, *IEEE Transactions on Rehabilitation Engineering* 8(4), 2000, pp. 441–446 | DOI `10.1109/86.895946` | Primary motor-imagery CSP reference for the classical spatial-filter baseline. |
| Wilcoxon signed-rank | Wilcoxon, *Biometrics Bulletin* 1(6), 1945, pp. 80–83 | Stable identifier `10.2307/3001968` | Primary source for the paired signed-rank procedure. |
| Holm correction | Holm, *Scandinavian Journal of Statistics* 6(2), 1979, pp. 65–70 | Stable identifier `10.2307/4615733` | Correct source for familywise multiplicity control. |
| Cohen's kappa | Cohen, *Educational and Psychological Measurement* 20(1), 1960, pp. 37–46 | DOI `10.1177/001316446002000104` | Primary source for chance-corrected agreement. |
| Brier score | Brier, *Monthly Weather Review* 78, 1950, pp. 1–3 | DOI `10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2` | Required only if the calibration diagnostic is discussed. |

## Corrections and required additions before manuscript submission

1. The current bibliography entries for EEGNet, FBCNet, ShallowConvNet, and EEG Conformer omit
   their persistent identifiers. Add the identifiers above during the later manuscript pass.
2. Add the CSP, Wilcoxon, and Cohen sources listed above when their corresponding method or metric
   is introduced in the manuscript.
3. Remove EEG-Inception, Zhou2020, and Tavakolan references if they are no longer discussed in the
   Peterson-only manuscript. A bibliography should follow the final text, not preserve abandoned
   experiments.
4. Keep Friedman explicitly exploratory in any result text. The frozen confirmatory family remains
   participant-level two-sided Wilcoxon with Holm correction.

## Claim discipline

- Architecture papers justify model definitions; their reported accuracies are not directly
  comparable with MI-OpenBCI because datasets, tasks, preprocessing, and validation differ.
- The Peterson study supports a within-dataset low-cost EEG decoding benchmark only.
- Neither the method citations nor the current results support a causal low-cost-versus-research-
  grade hardware claim or external-dataset generalization.
