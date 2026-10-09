# Diseño Peterson Diffusion V11

Fecha: 2026-10-08

Estado: aprobada por el autor el 2026-10-08

Rama: `exp/peterson-diffusion-v11`

Base congelada: `feat/peterson-journal-v10@6c116d5`

## 1. Objetivo

V11 seleccionará y ajustará un decoder de energía por difusión para la tarea Peterson
`motor_imagery_vs_rest`. La fase es exploratoria y usa exclusivamente `within_session`. Su propósito
es determinar si una formulación generativa condicionada por clase y máscara puede:

1. mejorar la accuracy limpia respecto de los controles comparables; o
2. conservar una accuracy limpia competitiva y mejorar la estabilidad ante canales ausentes o
   señales corruptas.

Peterson V10 y sus 3,250 celdas quedan congelados. V11 no sobrescribirá código, manifiestos,
estadísticas ni resultados V10. LOSO permanecerá fuera de V11 hasta que una configuración quede
congelada.

## 2. Evidencia que delimita la contribución

- V10 obtiene 83.24% de accuracy media con FBCNet en `within_session/overlap`, 80.98% con `full` y
  74.01% con `center`. V10 no contiene el control `within_session/center_x6`.
- DESAM y DDF-Net ya usan difusión para augmentación MI-EEG. La generación sintética no será la
  contribución principal de V11.
- Diffusion Classifier ya usa error de denoising como estimador condicional para clasificación. V11
  no reclamará como novedad la regla de energía por sí sola.
- La clasificación Riemanniana con datos ausentes proporciona un control explícito para canales
  faltantes.
- Los trabajos de calibración y rechazo en MI-EEG justifican medir selective risk y coverage, pero
  esa capa no constituye la novedad algorítmica.

La contribución candidata queda restringida a la combinación de clasificación por energía,
condicionamiento por máscara y consistencia de decisiones bajo degradaciones controladas en EEG
consumer-grade.

## 3. Hipótesis y criterios falsables

### H1. Diversidad temporal

Si las seis ventanas solapadas aportan información temporal útil, `overlap` superará a `center_x6`
con el mismo número de ejemplos, batches y actualizaciones.

H1 avanza si la diferencia media de accuracy por participante es al menos 0.02 y favorece a
`overlap` en al menos 7 de 10 participantes. Si no se cumple, `full` será la condición primaria de
las olas arquitectónicas y `overlap` quedará como sensibilidad.

El contraste primario de H1 promedia primero las cinco seeds dentro de cada combinación
participante-modelo y luego promedia EEGNet, FBCNet, ShallowConvNet y EEGConformer dentro de cada
participante. CSP+LDA se informa como sensibilidad secundaria porque sus repeticiones por seed son
deterministas y su sesgo inductivo de covarianza difiere del de los modelos deep. No se seleccionará
post hoc el modelo que produzca la mayor diferencia.

### H2. Base temporal suave

Si restringir los kernels temporales reduce sobreajuste con pocos datos, `MCDD-SmoothBasis`
superará a `MCDD-Res` con un presupuesto de parámetros comparable y reducirá la variabilidad entre
seeds. El efecto debería ser mayor en `full` y `center_x6` que en `overlap`.

H2 se rechaza si la variante suave queda dominada en accuracy, kappa y coste por `MCDD-Res` en las
tres condiciones.

### H3. Formulación generativa

Si el objetivo de difusión aprende estructura útil para señales degradadas, MCDD mejorará el área
bajo accuracy frente a severidad respecto del mismo backbone con cross-entropy.

H3 avanza por una de dos rutas:

- ruta de accuracy: mejora limpia de al menos 0.02 frente al mejor baseline comparable y dirección
  favorable en al menos 7 participantes;
- ruta de robustez: pérdida limpia no mayor de 0.02 y mejora de al menos 0.03 en el área
  accuracy-severidad frente al mejor control discriminativo.

### H4. Ranking y consistencia de máscara

Si el error generativo necesita supervisión discriminativa explícita, añadir ranking aumentará la
separación entre energías de clases. Si la consistencia de máscara captura estabilidad low-cost,
añadirla mejorará degradaciones sin perder más de 0.02 de accuracy limpia.

La ablación compara `noise_only`, `noise_rank` y `noise_rank_consistency`. Una variante que no
mejore su métrica objetivo respecto de la anterior no avanza.

### H5. Calibración y rechazo

Si las energías contienen información de confianza, temperature scaling ajustado solo en validación
reducirá ECE o Brier, y la selective prediction reducirá el riesgo al disminuir coverage. Este
análisis no cambia la accuracy cerrada ni puede rescatar una arquitectura que falle H3.

### H6. Número de evaluaciones de denoising

Si la estimación Monte Carlo se estabiliza rápidamente, `K=4` u `K=8` quedará a menos de 0.01 de
accuracy de `K=16` con menor latencia. Se evaluará `K ∈ {1, 4, 8, 16}` con ruido y timesteps comunes
para ambas clases.

## 4. Arquitecturas candidatas

Todas reciben tensores `C × T`, un timestep, una clase candidata y una máscara de canales. La salida
tiene la misma forma que la entrada. Ninguna variante reduce la resolución temporal.

1. `MCDD-Res`: stem depthwise-separable, mezclador espacial `1×1` y bloques residuales dilatados.
2. `MCDD-FilterBank`: ramas temporales asociadas a subbandas dentro de 8-30 Hz antes de la mezcla
   espacial.
3. `MCDD-SmoothBasis`: kernels temporales expresados como combinación de una base suave fija y
   coeficientes aprendibles.
4. `MCDD-Conformer`: embedding convolucional compacto y atención temporal con capacidad acotada.

Cada arquitectura tendrá una contraparte discriminativa con el mismo backbone. La diferencia de
parámetros entre cada par deberá quedar registrada y no superar 10%, salvo la cabeza estrictamente
necesaria para la tarea. En las pruebas de robustez, la contraparte discriminativa recibirá la misma
distribución de corrupciones de training que MCDD. Así se evita atribuir a difusión una ventaja que
proceda solo de haber visto señales corruptas.

## 5. Objetivo y clasificación

Para cada señal `x`, clase candidata `y`, máscara `m`, timestep `t` y ruido `epsilon`:

```text
x_t = sqrt(alpha_bar_t) * x + sqrt(1 - alpha_bar_t) * epsilon
E_y = mean((epsilon - epsilon_theta(x_t, t, y, m))^2 sobre elementos observables)

L_noise = E_true
L_rank  = max(0, margin + E_true - E_wrong)
L_cons  = JS(p(y|x,m_full), p(y|corrupt(x),m_corrupt))
L_total = L_noise + lambda_rank * L_rank + lambda_cons * L_cons

p(y|x) = softmax(-E_y / temperature)
```

Las clases candidatas reutilizarán exactamente los mismos `t` y `epsilon`. De este modo, la
diferencia de energía no incorpora ruido Monte Carlo distinto para cada clase. Temperature scaling
y cualquier umbral se ajustarán con predicciones de validación, nunca con test.

## 6. Condiciones y unidad experimental

Las tres condiciones son:

| Condición | Entrada | Propósito |
|---|---|---|
| `full` | trial completo de 4 s | Sin augmentación por ventanas |
| `center_x6` | misma ventana central de 2 s repetida seis veces | Control de cómputo y repetición |
| `overlap` | seis ventanas distintas de 2 s | Augmentación temporal |

El split se realiza por trial antes de generar ventanas. Todas las ventanas de un trial permanecen
en el mismo fold. Las probabilidades o energías se agregan a una predicción por trial. El
participante es la unidad estadística; folds, ventanas y seeds no se tratan como observaciones
independientes.

## 7. Programa por olas

```mermaid
flowchart LR
    V10[Peterson V10 congelado] --> W0[Ola 0: control de ventanas]
    W0 --> W1[Ola 1: arquitecturas]
    W1 --> W2[Ola 2: objetivos]
    W2 --> W3[Ola 3: cinco seeds]
    W3 --> W4[Ola 4: degradaciones]
    W4 --> W5[Ola 5: calibración y coste]
    W5 --> Freeze[Congelar ganador]
    Freeze --> LOSO[LOSO posterior, fuera de V11]
```

### Ola 0. Control de ventanas

Completar `within_session/center_x6` para CSP+LDA, EEGNet, FBCNet, ShallowConvNet y EEGConformer.
Usar seeds 0-4 y los splits V10. El auditor exigirá igualdad de ejemplos, batches y actualizaciones
entre `center_x6` y `overlap`.

### Ola 1. Screening arquitectónico

Evaluar las cuatro arquitecturas, sus contrapartes discriminativas y las tres condiciones con seed 0
en los diez participantes. Los smokes de forma, gradientes y determinismo no se usarán como evidencia
de rendimiento.

### Ola 2. Ablación del objetivo

Con la arquitectura seleccionada, comparar `noise_only`, `noise_rank` y
`noise_rank_consistency` en las tres condiciones y seeds 0-2.

### Ola 3. Evaluación within-session congelada

Ejecutar la configuración ganadora y su control discriminativo con seeds 0-4. Los baselines V10 se
referenciarán por hash. Solo se reentrenará un baseline cuando falte una condición exactamente
comparable.

### Ola 4. Degradaciones de test

Evaluar checkpoints congelados sin reentrenar:

- canales ausentes: 1, 3 y 5;
- dropout temporal por canal: 100, 250 y 500 ms;
- ruido por canal: SNR 20, 10 y 0 dB;
- clipping e impulsos.

Las máscaras y corrupciones aleatorias se precomputarán por trial, seed de corrupción y severidad.
Todos los modelos recibirán exactamente las mismas realizaciones. La métrica primaria de esta ola es
el área bajo accuracy frente a severidad.

### Ola 5. Calibración y coste

Generar ECE, Brier, NLL, risk-coverage, AURC, parámetros, latencia por trial, throughput, memoria GPU
máxima y número de forwards por predicción. También se probará `K ∈ {1, 4, 8, 16}`.

## 8. Métricas y estadística

La única métrica primaria de las olas 0-3 es accuracy por trial. Kappa, balanced accuracy, macro-F1,
AUROC, ECE y Brier son secundarias. La ola 4 usa el área accuracy-severidad como primaria.

Las seeds se promedian dentro de participante antes de cualquier prueba. Las comparaciones pareadas
usan Wilcoxon bilateral, intervalo del 95% de la diferencia, correlación rank-biserial y corrección
Holm dentro de cada familia predeclarada. Con diez participantes, la decisión no dependerá solo de
`p < 0.05`; también usará magnitud, intervalo y consistencia direccional.

Las comparaciones no previstas se marcarán como exploratorias. Un resultado observado no modificará
retroactivamente el umbral de éxito de su ola.

## 9. Integridad y reproducibilidad

Cada celda guardará:

- commit y estado limpio/sucio;
- hashes de datos, split y configuración;
- versiones de Python, PyTorch, CUDA, cuDNN, MNE y Braindecode;
- arquitectura, objetivo, condición, sujeto, fold y seed;
- conteos de trials, ventanas, batches y actualizaciones;
- curvas de entrenamiento y validación;
- predicciones por trial y métricas;
- parámetros, tiempo, memoria y estado de finalización.

Cada ola tendrá un manifiesto esperado antes de entrenar. El proceso de estadísticas y figuras se
detendrá si faltan celdas, existen duplicados, cambian los splits o una receta no coincide con su
hash congelado.

## 10. Organización del código y resultados

El desarrollo vive en el worktree `benchmark_deep_v2_diffusion_v11` y la rama
`exp/peterson-diffusion-v11`. El código experimental se aislará bajo:

```text
experiments/peterson_diffusion_v11/
├── configs/
├── manifests/
├── models/
├── runners/
├── statistics/
└── tests/
```

Los resultados de Cayetano no se versionarán:

```text
results_diffusion_v11/
├── wave_00_<commit>/
├── wave_01_<commit>/
├── wave_02_<commit>/
├── wave_03_<commit>/
├── wave_04_<commit>/
└── wave_05_<commit>/
```

Un único script de entrada aceptará la ola y repartirá shards entre `cuda:0` y `cuda:1`. Las celdas
válidas existentes se omitirán para permitir reanudación. Cada ola terminará con auditoría,
estadísticas, `run_complete.json`, archivo comprimido y SHA-256.

## 11. Ciclo entre resultados

1. Cayetano ejecuta una ola congelada.
2. El usuario transfiere el export y su SHA-256 al Mac.
3. Se valida checksum, manifiesto, completitud, leakage y determinismo.
4. Se interpreta la hipótesis con criterios predeclarados.
5. Se registra `advance`, `revise` o `stop` con justificación.
6. Solo entonces se genera y versiona el código de la siguiente ola.

El agente no puede observar la máquina Cayetano sin una ruta sincronizada o acceso remoto. El
archivo `run_complete.json` y el export son el contrato para iniciar la revisión en este chat.

## 12. Criterios de detención

Una variante se detiene si ocurre cualquiera de estas condiciones:

- leakage entre trials o ajuste con test;
- NaN, gradientes no finitos o colapso a una sola clase;
- receta, split o dataset con hash inesperado;
- pérdida limpia mayor de 0.02 sin compensación de robustez definida en H3;
- variante dominada en accuracy, kappa y coste por otra más simple;
- incumplimiento del manifiesto después de reanudación y auditoría.

V11 concluye al congelar una configuración o al rechazar H3. Solo una configuración congelada podrá
pasar a un futuro estudio LOSO confirmatorio.

## 13. Claims permitidos

Si los resultados los sostienen, V11 podrá afirmar mejora de accuracy dentro de participante o
mejora ante corrupciones simuladas y controladas. No podrá afirmar robustez a fallos reales de
hardware, superioridad clínica, generalización cross-session, novedad de augmentación por difusión ni
ser el primer clasificador de difusión para EEG.

## 14. Fuentes metodológicas

- Peterson et al., *A feasibility study of a complete low-cost consumer-grade brain-computer
  interface system*, Heliyon, 2020, DOI: 10.1016/j.heliyon.2020.e03425.
- Luo y Cai, *Diffusion models-based motor imagery EEG sample augmentation via mixup strategy*,
  Expert Systems with Applications, 2025, DOI: 10.1016/j.eswa.2024.125585.
- Li et al., *Your Diffusion Model is Secretly a Zero-Shot Classifier*, ICCV, 2023.
- Hippert-Ferrer et al., *Riemannian classification of EEG signals with missing values*, 2021,
  arXiv:2110.10011.
- Heim et al., *Real-Time EEG-Based BCI for Self-Paced Motor Imagery and Motor Execution Using
  Functional Neural Networks*, IEEE Access, 2025, DOI: 10.1109/ACCESS.2025.3569932.
- Ganeshkumar et al., *Reject option to reduce false prediction rates for EEG-motor imagery based
  BCI*, EMBC, 2017, DOI: 10.1109/EMBC.2017.8037479.
- Banville et al., *NeuralBench: A Unifying Framework to Benchmark NeuroAI Models*, arXiv:2605.08495.
