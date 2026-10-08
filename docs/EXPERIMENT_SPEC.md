# Especificación confirmatoria Peterson V10

## Pregunta y alcance

El benchmark confirmatorio usa únicamente MI-OpenBCI/Peterson: diez participantes, 15 canales
low-cost y clasificación de imaginación motora frente a reposo. Evalúa rendimiento intra-sujeto,
robustez entre particiones y generalización a sujetos no observados. Souza2023 y el estudio de
transferencia permanecen como análisis históricos separados; no forman parte de V10.

El estudio no identifica causalmente un efecto del costo del hardware ni demuestra generalización
a otros datasets.

## Modelos

- CSP+LDA.
- EEGNet.
- FBCNet.
- ShallowConvNet, implementado como `ShallowFBCSPNet`.
- EEGConformer.

Las cuatro redes proceden de Braindecode 1.5.2. CSP+LDA usa covarianzas OAS, seis filtros CSP y
LDA con solver SVD. EEGInceptionMI, CNN2D, ATCNet y los modelos del estudio de transferencia no
pertenecen a la matriz confirmatoria V10.

## Preprocesamiento común

- Banda inicial 1–40 Hz para ajustar ICA cuando corresponde.
- Análisis primario sin eliminación de componentes ICA.
- Sensibilidad exploratoria: FastICA ajustada solo en training, kurtosis mayor que 10 y máximo dos
  componentes, sin selección manual.
- Banda final 8–30 Hz y remuestreo a 128 Hz.
- Trials Peterson de 4 s, equivalentes a 512 muestras.
- Z-score por canal calculado solo en training y aplicado a test.

Cada fold de sensibilidad conserva convergencia, iteraciones, seed, kurtosis y componentes
excluidos. La regla no se interpreta como identificación fisiológica ocular o muscular.

## Protocolos

- `within_split`: partición estratificada 70/30 dentro de participante.
- `within_session`: cinco folds estratificados dentro de participante.
- `loso`: leave-one-subject-out.

La condición primaria usa seis ventanas solapadas de 2 s distribuidas sobre el trial completo. Las
probabilidades de las ventanas se promedian para producir exactamente una predicción por trial.
Los controles son:

- `full`: trial completo de 4 s, sin sliding window.
- `center`: crop central de 2 s.
- `center_x2`: el crop central repetido dos veces para emparejar `nonoverlap`.
- `nonoverlap`: dos ventanas no solapadas de 2 s.
- `center_x6`: el crop central repetido seis veces para emparejar `overlap`.
- `overlap`: seis ventanas solapadas de 2 s.

Las comparaciones `nonoverlap-center_x2` y `overlap-center_x6` aíslan diversidad temporal bajo el
mismo número de ejemplos, batches por época y actualizaciones. El análisis se detiene si esos
conteos difieren.

## Entrenamiento, semillas y unidad estadística

Las redes usan las semillas 0–4 y 300 épocas fijas. Las particiones se fijan con
`split_seed=2026`; las semillas de modelo cambian inicialización y orden de batches, no train/test.
CSP es determinista, pero se serializa en la misma cuadrícula para completar el diseño pareado y
se marca `optimization_stochastic=false`.

Antes de cualquier inferencia, las semillas se promedian dentro de participante. El participante
es la unidad estadística; ventanas, folds y semillas no se tratan como observaciones biológicas
independientes.

## Hipótesis y estadística

Métrica primaria: accuracy. Métrica secundaria: Cohen's kappa.

La familia confirmatoria contiene los diez pares de modelos dentro de cada protocolo y métrica en
la condición `overlap`, sin ICA. Se usa Wilcoxon bilateral por participante, corrección de Holm y
correlación rank-biserial pareada. Los intervalos del 95% para medias y diferencias pareadas usan
la distribución t de Student, sin bootstrap. Friedman y los diagnósticos restantes son
exploratorios.

## Criterios de cierre

- Exactamente 3,250 celdas válidas y sin duplicados.
- Cinco seeds completas por celda deep y marcador determinista en CSP.
- 300 épocas registradas en cada fold deep.
- Hashes de código, datos, entorno, receta y particiones presentes.
- Tablas confirmatorias, sensibilidad de augmentación e ICA, calidad, latencia y figuras
  generadas únicamente después del auditor de integridad.
