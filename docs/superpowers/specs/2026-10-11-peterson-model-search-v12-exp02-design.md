# Peterson V12 exp02: diseño factorial de topología, objetivo e inferencia

Fecha: 2026-10-11  
Rama: `exp/peterson-model-search-v12`  
Base ejecutable: `6a2d2e2`  
Entrada obligatoria: exp01 reparado completo, 216/216 celdas, sin acceso al holdout

## 1. Decisión que motiva exp02

Exp01 identificó una señal prometedora, pero todavía no una superioridad demostrada:

- la familia híbrida Inception compacta alcanzó `0.7700` de accuracy media y `0.5401` de kappa;
- la familia híbrida no presentó colapso en ninguna de sus 72 celdas;
- la energía de difusión pura quedó cerca del azar y colapsó en 30/72 celdas;
- el incremento híbrido frente al control discriminativo fue positivo, pero incierto con seis
  participantes de descubrimiento;
- la mejora apareció principalmente en la topología Inception, por lo que exp01 no permite separar
  un efecto del objetivo auxiliar de un efecto de capacidad o arquitectura.

Exp02 no ampliará la búsqueda de backbones. Se concentrará en una familia Inception temporal
multiescala y separará de forma factorial tres mecanismos: topología, objetivo de entrenamiento y
regla de inferencia. La difusión se conservará como objetivo auxiliar, no como generador de muestras
ni como clasificador energético independiente.

## 2. Pregunta e hipótesis falsables

### Pregunta primaria

¿Una red Inception temporal diseñada para EEG low-cost, regularizada mediante denoising condicional,
supera a FBCNet y CSP+LDA cuando todos usan los mismos participantes, trials, folds y ventanas?

### H1. Topología multiescala

Los kernels temporales largos, las ramas depthwise-separable o un pooling que retenga dispersión
mejorarán la accuracy por participante frente al Inception compacto original sin aumentar la tasa de
colapso.

**Predicción:** al menos una topología superará al control compacto en `>=1.0` punto porcentual de
accuracy media y mostrará dirección favorable en al menos 4/6 participantes.

**Refutación:** ninguna topología alcanza ese margen o cualquier ganancia exige colapso, peor kappa o
un crecimiento desproporcionado de latencia.

### H2. Denoising como regularizador auxiliar

Una contribución moderada de denoising mejorará la representación compartida frente al control de
clasificación de igual arquitectura, sin depender de la puntuación energética durante inferencia.

**Predicción:** al menos una receta auxiliar con inferencia `classifier-only` superará a su control
`classification-only` emparejado en topología y presupuesto.

**Refutación:** la ventaja desaparece con `alpha=1.0`, se limita a una sola persona o empeora la
calibración y el kappa de forma material.

### H3. Fusión de evidencia

Si la cabeza energética contiene información complementaria, una fusión predeclarada de
probabilidades superará al mismo checkpoint con inferencia solo discriminativa.

**Predicción:** una combinación con `alpha<1.0` mejorará la accuracy media y no dependerá de un único
número de muestras de ruido.

**Refutación:** `alpha=1.0` domina, los resultados cambian de signo con `K` o la energía produce
probabilidades degeneradas. En ese caso se retirará completamente la energía durante inferencia.

## 3. Alcance congelado

| Elemento | Valor exp02 |
|---|---|
| Dataset | MI-OpenBCI/Peterson |
| Tarea | motor imagery vs rest |
| Participantes | descubrimiento: `S09,S03,S02,S10,S12,S08` |
| Holdout | `S07,S04,S05,S06`, inaccesible |
| Protocolo | `within_session` |
| Ventanas | `overlap`, agregadas por trial |
| Folds internos | 0, 2 y 4 |
| Seed | 0 |
| Baselines | CSP+LDA y FBCNet |
| Épocas máximas | 150 |
| Paciencia | 35 |
| Métrica primaria | accuracy macro por participante |
| Guardrails | kappa, colapso de clase y latencia p95 por trial |
| Fuera de alcance | LOSO, Souza2023, test exterior, exp05 y nuevas fuentes de datos |

No se utilizará augmentación estocástica en exp02. El solapamiento de ventanas permanece activo,
pero channel dropout y temporal masking se reservarán para la siguiente etapa. Esto evita confundir
el efecto de la arquitectura y el objetivo auxiliar con el de una segunda regularización.

## 4. Factor A: seis topologías Inception y dos rutas FBCNet

Todas las topologías conservarán dos bloques residuales con dilataciones 1 y 2, normalización,
activación GELU y dropout congelados. Solo cambiará el factor declarado.

| ID | Canales ocultos | Kernels | Rama | Pooling |
|---|---:|---|---|---|
| T0 | 16 | 9,17,33,65 | convolución estándar | media |
| T1 | 32 | 9,17,33,65 | convolución estándar | media |
| T2 | 16 | 17,33,65,129 | convolución estándar | media |
| T3 | 16 | 9,17,33,65 | depthwise temporal + pointwise | media |
| T4 | 16 | 9,17,33,65 | convolución estándar | media + desviación estándar |
| T5 | 16 | 9,17,33,65 | convolución estándar | atención temporal de una cabeza |
| T6 | 96 | bandas FBCNet 8-30 Hz | FBCNet, 16 filtros espaciales por banda | log-varianza |
| T7 | 192 | bandas FBCNet 8-30 Hz | FBCNet, 32 filtros espaciales por banda | log-varianza |

El pooling por atención usará una proyección escalar `1x1`, softmax temporal y suma ponderada. El
pooling estadístico concatenará media y desviación estándar con corrección desactivada para que sea
estable en ventanas cortas. Cada celda registrará parámetros totales y parámetros activos.

T6 y T7 reutilizarán la capa de filter-bank, la convolución espacial y el pooling de log-varianza
de FBCNet. El objetivo auxiliar se conectará a la representación temporal previa al pooling. T7/O0
actuará como control arquitectónico del FBCNet exacto dentro del mismo entrenador; el baseline
externo seguirá usando la implementación y receta congeladas de V10. Esto permite distinguir un
beneficio de FBCNet, del denoising y de su interacción.

## 5. Factor B: seis objetivos de entrenamiento

La pérdida general será:

`L = L_ce + lambda_d L_denoise + lambda_r L_rank + lambda_c L_consistency`.

| ID | `lambda_d` | `lambda_r` | `lambda_c` | checkpoint |
|---|---:|---:|---:|---|
| O0 | 0.00 | 0.00 | 0.00 | CE de validación |
| O1 | 0.25 | 0.00 | 0.00 | CE de validación |
| O2 | 0.25 | 0.10 | 0.00 | CE de validación |
| O3 | 1.00 | 0.20 | 0.10 | pérdida total de validación |
| O4 | 1.00 | 0.20 | 0.10 | CE de validación |
| O5 | 0.25 | 0.10 | 0.05 | CE de validación |

O3 replica la receta completa de exp01. O4 aísla el criterio de checkpoint. O0 usará exactamente la
misma arquitectura capaz de denoising que los demás objetivos, pero omitirá el cálculo auxiliar y
evaluará solo el clasificador. Así se controla la capacidad del modelo sin pagar cómputo innecesario.
Los parámetros auxiliares sin gradiente se informarán como inactivos.

El resto del entrenamiento permanecerá congelado: AdamW, learning rate `6.25e-4`, weight decay
`1e-4`, batch 64, cosine decay, clipping 5.0 y el calendario DDPM lineal de 1,000 pasos usado en el
exp01 reparado.

## 6. Factor C: inferencia sin reentrenamiento

Cada checkpoint O1-O5 generará probabilidades discriminativas `p_cls` y energéticas `p_energy`. La
fusión será:

`p = alpha * p_cls + (1 - alpha) * p_energy`.

Se evaluarán `alpha={0,0.25,0.5,0.75,1}` y `K={1,4,8}` muestras de ruido. Cuando `alpha=1`, K no
cambia el resultado y se guardará una sola receta. Esto produce 13 recetas únicas por checkpoint
híbrido. Las ocho muestras se generarán una vez con seeds deterministas; K=1 y K=4 usarán prefijos
anidados de la misma secuencia, evitando repetir inferencia.

O0 solo admite `alpha=1`. Los resultados conservarán por separado `p_cls`, `p_energy`, la predicción
fusionada y los identificadores de trial. Ninguna variante se entrenará de nuevo.

## 7. Tamaño exacto

- 8 topologías x 6 objetivos = 48 configuraciones entrenadas;
- 48 configuraciones x 6 participantes = 288 celdas candidatas;
- 288 celdas x 3 folds = 864 entrenamientos de candidatos;
- 40 checkpoints auxiliares x 13 inferencias + 8 controles O0 = 528 configuraciones efectivas;
- 2 baselines x 6 participantes = 12 celdas baseline;
- total del manifiesto principal: 300 celdas;
- total de filas de evaluación candidatas: 528 x 6 = 3,168.

Las celdas baseline también incluirán tres folds. CSP+LDA y FBCNet usarán los mismos índices de
trial que cada candidato. FBCNet reutilizará la receta congelada V10, no un reentrenamiento con
hiperparámetros favorecidos por exp02.

## 8. Decisión y control de multiplicidad

Exp02 es una etapa de selección, no una prueba confirmatoria. La configuración primaria de cada
checkpoint será la receta de inferencia con mayor accuracy macro por participante. Todas las demás
se conservarán y se informarán; no se eliminarán ejecuciones desfavorables.

Se compararán diferencias emparejadas por participante contra CSP+LDA y FBCNet. Se producirán media,
mediana, bootstrap descriptivo e indicación direccional. Los p-valores exactos por cambio de signo y
la corrección Holm serán descriptivos debido a `n=6` y al sesgo de selección. No se afirmará
significancia estadística a partir de exp02.

Una configuración será elegible si:

1. no presenta colapso en ningún participante ni fold;
2. tiene todos los trials y clases esperados;
3. no contiene NaN ni errores de convergencia;
4. conserva kappa no inferior al baseline de referencia en más de 0.02;
5. conserva trazabilidad completa de datos, split, código y checkpoint.

La señal mínima para continuar hacia optimización será una configuración elegible que:

- supere a FBCNet en al menos `+1.0` punto porcentual de accuracy macro;
- mejore en al menos 4/6 participantes;
- no empeore el cuartil inferior en más de 1 punto porcentual.

Si no existe, exp02 terminará con estado `needs_redesign`; no se promoverá un ganador por obligación.
Si existe, se promoverán hasta ocho configuraciones, con máximo dos por topología y dos por objetivo.
Los empates menores de 0.5 puntos porcentuales se resolverán por cuartil inferior, kappa, menor
latencia p95 y menor número de parámetros, en ese orden.

## 9. Integridad, portabilidad y recuperación

El runner escribirá antes de entrenar:

- `experiment_manifest.json` con las 300 celdas esperadas;
- `catalog.json` con las 48 configuraciones completas;
- `inference_catalog.json` con las 528 recetas efectivas;
- `data_manifest.json` con SHA-256 de cada MAT y hashes de datos estandarizados;
- `split_manifest.json` con índices y hashes de folds;
- `environment.json` y estado Git limpio.

Cada celda se escribirá de forma atómica y podrá reanudarse solo si coincide su fingerprint completo.
Un fallo de configuración se registrará como celda firmada y no detendrá el shard. El auditor tendrá
dos modos:

- `strict`: recomputa hashes de los MAT, datos y splits en Cayetano;
- `portable`: verifica manifiestos, celdas, checkpoints y hashes sin exigir una copia local del
  dataset.

El reporte final distinguirá fallos técnicos, colapso real, configuraciones dominadas y evidencia
inconclusa. No mezclará celdas de exp01 ni de commits anteriores.

## 10. Paralelización y artefactos

Las 300 celdas se dividirán determinísticamente en dos shards balanceados por costo estimado. Cada
GPU ejecutará un shard. La asignación formará parte del manifiesto, y la identidad científica de una
celda no dependerá del dispositivo.

La ejecución producirá:

- checkpoints y celdas firmadas;
- tabla larga de las 3,168 evaluaciones;
- comparaciones emparejadas contra CSP+LDA y FBCNet;
- ablation readout de topología, objetivo, checkpoint e inferencia;
- fronteras accuracy-latencia y accuracy-parámetros;
- matriz de colapso y estabilidad por participante;
- `promotion_decision.json` reproducible;
- `run_complete.json` solo si el manifiesto y los auditores pasan.

## 11. Criterios de aceptación del código

Antes de ejecutar en Cayetano deben pasar:

1. tests unitarios de las seis topologías y seis objetivos;
2. prueba de equivalencia de la receta O3/T0 con exp01;
3. prueba de que O0 no ejecuta ni actualiza la cabeza auxiliar;
4. prueba de agregación por trial sin fuga entre folds;
5. prueba de prefijos deterministas K=1/4/8;
6. prueba de igualdad de splits entre candidatos y baselines;
7. smoke de una celda por GPU con auditoría strict y portable;
8. revisión de Python y revisión de código sin hallazgos críticos.

## 12. Anclajes metodológicos

El diseño multiescala y dilatado se apoya en EEG-ITNet
(`https://doi.org/10.1109/ACCESS.2022.3161489`); las ramas separables e Inception, en MI-EEGNet
(`https://doi.org/10.1016/j.jneumeth.2020.109037`); y la comparación profunda principal conserva
FBCNet como referencia de filter-bank y pooling de varianza. El objetivo de denoising deriva del
mecanismo DDPM validado en el proyecto MICCAI y del DDPM original
(`https://proceedings.neurips.cc/paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4-Abstract.html`).

Estos trabajos justifican los componentes que se prueban, pero no demuestran de antemano que la
combinación sea superior en Peterson. Esa afirmación queda condicionada a los criterios anteriores.
