# Peterson Exp05 — búsqueda mecanística sobre FBCNet

## Objetivo

Identificar, sin abrir el holdout, un único decoder para MI-vs-rest en EEG de
bajo costo que mejore la exactitud limpia de FBCNet. Exp05 conserva el núcleo
fisiológico y la receta de FBCNet y modifica una sola familia de mecanismo por
candidato. La robustez a artefactos es secundaria y no compensa una caída en
exactitud limpia.

## Población y particiones

- Discovery: `S09,S03,S02,S10,S12,S08`.
- Holdout bloqueado: `S07,S04,S05,S06`.
- Protocolo: `within_session`, cinco folds externos estratificados.
- Las divisiones ocurren a nivel de ensayo antes de crear ventanas.
- Selección de época: validación interna creada únicamente desde el train
  externo.
- Evaluación: una sola vez en el test externo después de reinicializar y
  reentrenar por el número de épocas seleccionado sobre todo el train externo.

## Control congelado

FBCNet con seis bandas `(8,12)…(28,30)`, 32 filtros espaciales, log-varianza,
`stride_factor=4`, seis ventanas solapadas de 2 s, AdamW, `lr=6.25e-4`,
`weight_decay=1e-4`, batch 64 y un máximo de 300 épocas.

## Candidatos

Veinte intervenciones individuales: pesos globales o condicionados por banda,
regularización de atención, banda mu individual train-only, filtros Sinc
restringidos, banco solapado fino, temporal multiescala, logvar multiescala,
logvar robusta, median-of-means, dos agregadores de ventanas, inicialización
CSP congelada o ajustable, ortogonalidad espacial, cabeza SPD, center loss,
SupCon, FTSurrogate y ponderación acotada por calidad.

No se incluyen difusión, routers, doble experto, channel dropout ni ICA como
ruta principal.

## Ejecución y decisión

1. Screen: control + 20 candidatos, seis sujetos, seeds `0,1,2`.
2. El análisis selecciona como máximo cuatro mecanismos, priorizando diversidad.
3. Confirmación interna: control y promovidos, seeds `3,4`.
4. El holdout permanece cerrado. Abrirlo exige una decisión posterior y una
   receta única congelada.

Un candidato pasa el screen si tiene cero colapsos, delta medio >=0.5 pp,
victorias en >=4/6 sujetos, caída de kappa <=0.01, caída de cuartil inferior
<=0.5 pp y dispersión entre seeds no más de 0.5 pp peor que el control. El
objetivo preferido tras cinco seeds es >=1 pp. Los p-valores de discovery son
descriptivos y no se usan como confirmación.

## Reproducibilidad

Los manifests fijan catálogo, celdas, revisión, configuración y particiones.
Cada resultado registra hashes de datos, código, split y entorno, índices de
train/validación/test, época seleccionada, parámetros, predicciones por ensayo,
matriz de confusión y flags de calidad. Los dos shards poseen sujetos disjuntos.
