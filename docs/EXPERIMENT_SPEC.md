# Especificacion del benchmark deep EEG

## Pregunta y alcance

El benchmark evalua cinco decoders deep bajo protocolos identicos y separa tres preguntas:

1. Rendimiento calibrado dentro de sujeto y sesion.
2. Transferencia entre sesiones del mismo sujeto.
3. Transferencia a sujetos no observados cuando el dataset lo permite.

MI-OpenBCI es el dataset low-cost principal. Zhou2020 aporta una tarea research-grade comparable,
right-hand motor imagery frente a rest, con siete sesiones. Tavakolan2017 aporta la misma
comparacion right-hand frente a rest, cuatro sesiones y una cohorte de 12 sujetos con menor costo
computacional. Las diferencias
entre datasets se interpretan como validacion externa contextual, no como efectos causales del
hardware.

## Modelos

- EEGNet
- FBCNet
- ShallowConvNet, implementado como `ShallowFBCSPNet`
- EEGConformer
- EEGInceptionMI

Las cinco implementaciones proceden de Braindecode 1.5.2. CSP, DeepConvNet y modelos caseros no
forman parte del benchmark nuevo.

Esta sustitucion del conjunto historico CSP/EEGNet/CNN2D/ShallowConvNet/ATCNet es una decision
explicita del autor para responder la revision con decoders deep modernos y evitar que el nuevo
paper mezcle clasificadores clasicos con redes neuronales. El handoff MICCAI se conserva como
referencia metodologica, pero no define el conjunto final de modelos de esta revision.

## Preprocesamiento comun

- Banda inicial 1 a 40 Hz para ajustar ICA.
- El analisis primario no elimina componentes ICA, porque el criterio de kurtosis no identifica de
  forma fiable fisiologia ocular o muscular en este montaje.
- Como sensibilidad exploratoria, FastICA se ajusta solo en training y selecciona componentes con
  kurtosis mayor que 10, con un maximo de dos.
- Banda final 8 a 30 Hz.
- Remuestreo a 128 Hz.
- Épocas fijadas por dataset: cuatro segundos (512 muestras) para MI-OpenBCI y Zhou2020, y tres
  segundos (384 muestras) para Tavakolan2017. Todos los modelos reciben exactamente el mismo
  soporte temporal dentro de cada dataset.
- Z-score por canal calculado solo en training.
- MOABB aplica la banda inicial 1-40 Hz durante la carga. El preprocesamiento registra ese origen
  y no repite el mismo filtro; MI-OpenBCI recibe la banda inicial dentro de cada fold.

Cada fold de sensibilidad guarda convergencia, iteraciones, seed, kurtosis, candidatos y
componentes excluidos. No existe seleccion manual. Esta condicion no se describe como
identificacion ocular o muscular.

## Protocolos

- `within_split`: particion estratificada 70/30 dentro de cada sesion; es el baseline
  within-session del perfil paper.
- `within_session`: cinco folds estratificados dentro de cada sesion.
- `cross_session`: leave-one-session-out dentro de cada sujeto.
- `loso`: leave-one-subject-out, reservado para datasets con montaje compatible entre sujetos.

La tabla primaria usa trials completos y sin sliding windows. La augmentacion es un experimento
separado con cuatro condiciones aplicadas por igual a todos los modelos:

- `center_x2`: el crop central de dos segundos repetido dos veces para emparejar `nonoverlap`.
- `nonoverlap`: dos ventanas no solapadas de dos segundos durante training.
- `center_x6`: el crop central de dos segundos repetido seis veces para emparejar `overlap`.
- `overlap`: seis ventanas solapadas de dos segundos durante training.

La evaluacion siempre permanece a nivel de trial.
Cada pareja emparejada contiene el mismo número de ejemplos, batches por época y actualizaciones
del optimizador. El análisis se detiene si esos conteos difieren.

## Calidad de señal

Antes del entrenamiento se registran proporción de valores finitos, varianza, desviación estándar,
RMS, amplitud pico a pico, canales constantes o casi planos, alertas de amplitud relativa y
conteos por clase y sesión. Las alertas no excluyen datos automáticamente. En los datasets de
MOABB no se estima ruido de línea a partir de épocas ya filtradas a 1-40 Hz; se marca como no
disponible.

## Semillas y unidad estadistica

El perfil de paper usa las semillas 0, 1, 2, 3 y 4. Se conservan las predicciones de cada celda.
Para comparar modelos, primero se promedian las semillas dentro de cada sujeto. El sujeto es la
unidad estadistica y ninguna ventana, fold o semilla se trata como una observacion independiente.
Las particiones se fijan con `split_seed=2026` para todos los modelos y repeticiones. Las cinco
semillas modifican inicializacion y orden de batches, no la composicion de train y test; por ello la
dispersion entre semillas estima variabilidad de optimizacion y no queda confundida con cambios de
particion. FastICA usa una seed fija derivada de `split_seed`; tampoco se mezcla la aleatoriedad de
la descomposicion ICA con la variabilidad del entrenamiento neural.

## Hipotesis y estadistica

Metrica primaria: accuracy. Metrica secundaria: Cohen's kappa.

Dentro de cada familia dataset, protocolo y metrica se comparan los diez pares de modelos mediante
Wilcoxon bilateral y correccion de Holm. Los intervalos del 95 por ciento para medias y diferencias
pareadas usan la distribucion t de Student, sin bootstrap. La variabilidad entre sujetos y la
variabilidad entre semillas se informan por separado. Cada contraste incluye la correlacion
rank-biserial pareada, cuyo signo sigue la diferencia indicada en la tabla.

La augmentacion usa Wilcoxon pareado sobre deltas por sujeto promediados entre semillas. La familia
de Holm contiene las comparaciones `nonoverlap-center_x2` y `overlap-center_x6` de los cinco
modelos.

## Criterios de cierre de feedback

- Multisemilla: cerrado cuando existen cinco semillas completas por celda primaria.
- Estadistica: cerrado cuando se generan tablas Wilcoxon-Holm e IC con sujeto como unidad.
- Fairness: cerrado cuando todos los modelos reciben identico soporte temporal y augmentacion.
- Cross-session: cerrado cuando Tavakolan2017 y Zhou2020 terminan leave-one-session-out.
- ICA: parcialmente cerrado. El primario no usa ICA; la sensibilidad por kurtosis es auditable,
  pero no identifica la fisiologia del componente.
- Latencia: cerrado cuando todos los modelos se perfilan en el mismo dispositivo, input y batch.
- Reproducibilidad: cerrado cuando manifest, versiones, configuracion y predicciones estan
  disponibles en el repositorio o release.
