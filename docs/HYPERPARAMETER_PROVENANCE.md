# Procedencia congelada de hiperparámetros

Esta tabla se fijó antes de ejecutar la matriz multisemilla. Ningún valor se selecciona usando los
resultados de test. El benchmark compara configuraciones reproducibles de frameworks, no el mejor
hiperparámetro alcanzable por cada arquitectura.

## Parámetros compartidos

| Parámetro | Valor | Procedencia y decisión |
|---|---:|---|
| Epochs | 300 fijas | Receta de convergencia transferida desde `HANDOFF_MICCAI_PARA_CLEI.md`; evita early stopping inestable en cohortes pequeñas |
| Batch size | 64 | Receta transferida; `drop_last=False` conserva el último batch |
| Scheduler | CosineAnnealingLR | Receta de convergencia transferida |
| Train split interno | Ninguno | Todo el training fold se usa para ajuste; test nunca selecciona epochs |
| Sampling rate | 128 Hz | Soporte común para los cinco modelos |
| Input primario | 4 s en MI-OpenBCI/Zhou2020; 3 s en Tavakolan2017 | Igual para todos los modelos dentro de cada dataset; respeta la duración disponible de Tavakolan2017 |
| Banda común | 8-30 Hz | Banda sensorimotora preespecificada |

## Parámetros por modelo

| Modelo | Implementación | Optimizador | LR | `provenance_id` | Lectura correcta |
|---|---|---|---:|---|---|
| EEGNet | Braindecode 1.5.2 `EEGNet` | AdamW | 6.25e-4 | `miccai_fixed_convergence_v1` | Receta que corrigió infraentrenamiento en el proyecto hermano |
| FBCNet | Braindecode 1.5.2 `FBCNet` | AdamW | 6.25e-4 | `shared_fixed_convergence_v1` | Presupuesto compartido, no valor reclamado como óptimo del paper original |
| ShallowConvNet | Braindecode 1.5.2 `ShallowFBCSPNet` | AdamW | 6.25e-4 | `shared_fixed_convergence_v1` | Presupuesto compartido, no búsqueda específica |
| EEGConformer | Braindecode 1.5.2 `EEGConformer` | Adam | 5e-5 | `project_predeclared_conformer_v1` | Configuración predeclarada del proyecto, sin test tuning |
| EEGInceptionMI | Braindecode 1.5.2 `EEGInceptionMI` | Adam | 1e-3 | `project_predeclared_inception_v1` | Configuración predeclarada del proyecto, sin test tuning |

FBCNet aplica internamente seis subbandas: 8-12, 12-16, 16-20, 20-24, 24-28 y 28-30 Hz. Se
restringieron a la banda común porque el preprocessing ya elimina frecuencias fuera de 8-30 Hz.

La receta completa se serializa dentro de cada celda. El fingerprint incluye la receta, el código,
las versiones del stack científico y el SHA-256 de las épocas cargadas, remuestreadas y recortadas
de cada sujeto o cohorte LOSO. Cada fold guarda además hashes de train y test después del
preprocesamiento, así como de las entradas finales entregadas al modelo. Para dependencias
instaladas desde un repositorio, también incorpora el
contenido de `direct_url.json`; por tanto, Tavakolan2017 registra el commit exacto de
BCI2kReader y no solo su versión declarada `0.31.dev0`. La revisión Git del benchmark se conserva
como metadato de auditoría, pero se excluye del hash operativo: un commit que solo cambie
documentación no invalida celdas cuyos código, datos, entorno y receta permanecen idénticos.
