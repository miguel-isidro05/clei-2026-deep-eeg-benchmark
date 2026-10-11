# Peterson V12 exp03: confirmación causal de FBCNet y adaptación a EEG low-cost

Fecha: 2026-10-11

Rama: `exp/peterson-model-search-v12`

Base ejecutable: `291c3f9`
Entrada obligatoria: exp02 completo, 300/300 celdas, sin acceso al holdout

## 1. Decisión que motiva exp03

Exp02 produjo un resultado útil pero no una arquitectura nueva:

- CSP+LDA alcanzó `0.764779` de accuracy media;
- el FBCNet congelado alcanzó `0.820513`;
- `T7/O0`, que reproduce el clasificador FBCNet completo dentro del entrenador de búsqueda,
  alcanzó `0.831553` y kappa `0.663105`;
- la diferencia emparejada frente al FBCNet congelado fue `+0.011040`, con 4/6 victorias,
  un empate, una derrota, IC bootstrap `[-0.010684, 0.028490]` y prueba exacta de signos
  `p=0.4375`;
- las rutas con denoising, ranking o inferencia energética empeoraron de forma material;
- 197/288 celdas candidatas presentaron al menos una ruta o fold con colapso de clase;
- el conteo de `83,063` parámetros de T7/O0 era engañoso: solo `4,994` participaban en el
  clasificador y el resto pertenecía a módulos auxiliares inactivos.

Por tanto, exp03 retirará la difusión de la ruta principal. Primero determinará si la mejora de
T7/O0 provino del criterio de checkpoint, del weight decay o de una diferencia accidental entre
entrenadores. Después evaluará una adaptación explícita a la degradación de canales en EEG low-cost,
manteniendo el backbone espectro-espacial de FBCNet.

## 2. Preguntas e hipótesis falsables

### H1. Selección del checkpoint

Seleccionar el checkpoint con menor CE de validación dentro de una trayectoria de 300 épocas
mejorará la generalización frente a usar la última época de esa misma trayectoria.

**Predicción:** el contraste emparejado `best_val - last_epoch`, agregado primero por
participante y seed, será positivo en accuracy y kappa sin aumentar el colapso.

**Refutación:** la diferencia media es menor de `+0.5` puntos porcentuales, cambia de signo entre
seeds o depende de un solo participante.

### H2. Weight decay

`weight_decay=1e-4` reducirá la varianza entre seeds o mejorará el cuartil inferior frente a
`weight_decay=0` con trayectoria, datos y checkpoint emparejados.

**Predicción:** el contraste marginal de weight decay será favorable en accuracy media o en el
cuartil inferior, sin una pérdida de kappa mayor de 0.02.

**Refutación:** no aporta mejora reproducible o perjudica sistemáticamente a los participantes de
menor rendimiento.

### H3. Gating condicionado por calidad

Una compuerta inicializada como identidad, alimentada solo por descriptores de calidad calculados
sin usar etiquetas de test, permitirá atenuar canales degradados sin destruir la representación de
bandas de FBCNet.

**Predicción:** al menos una variante superará en `>=1.0` punto porcentual al mejor control FBCNet
confirmado, mejorará en 4/6 participantes y conservará el cuartil inferior.

**Refutación:** ninguna variante alcanza el margen, el efecto desaparece al excluir amplitud del
descriptor o aparece colapso de clase.

### H4. Channel dropout

La eliminación aleatoria de canales completos durante entrenamiento aumentará robustez en datos
low-cost y complementará, o sustituirá, al gating aprendido.

**Predicción:** `p=0.1` o `p=0.2` mejorará el rendimiento de descubrimiento y una prueba de estrés
con canales ausentes frente al mismo modelo con `p=0`.

**Refutación:** solo mejora la prueba corrupta, reduce la accuracy limpia o su interacción con el
gating es negativa.

## 3. Alcance congelado

| Elemento | Valor exp03 |
|---|---|
| Dataset | MI-OpenBCI/Peterson |
| Tarea | motor imagery vs rest |
| Participantes de descubrimiento | `S09,S03,S02,S10,S12,S08` |
| Holdout | `S07,S04,S05,S06`, inaccesible |
| Protocolo | `within_session` |
| Ventanas | `overlap`, agregadas por trial |
| Métrica primaria | accuracy media por participante |
| Guardrails | kappa, cuartil inferior, colapso, parámetros activos y latencia |
| Baselines | CSP+LDA y FBCNet exacto |
| Fuera de alcance | LOSO, Souza2023, holdout, difusión, generación sintética y afirmaciones confirmatorias |

El experimento tendrá dos fases secuenciales. La fase B no puede comenzar sin una decisión firmada y
auditable de la fase A.

## 4. Fase A: confirmación causal de la receta

### 4.1 Arquitectura congelada

Se usará únicamente el FBCNet exacto de T7/O0:

- seis bandas: 8-12, 12-16, 16-20, 20-24, 24-28 y 28-30 Hz;
- 32 filtros espaciales por banda;
- pooling de log-varianza;
- cabeza binaria;
- `4,994` parámetros activos;
- ningún módulo de denoising, energía o difusión instanciado.

La implementación verificará en construcción que `total_params == active_classifier_params`.

### 4.2 Factorial 2 x 2

Para cada valor de weight decay se entrenará una única trayectoria completa de 300 épocas. De la
misma trayectoria se evaluarán dos checkpoints, eliminando diferencias estocásticas entre las
políticas de selección.

| Factor | Niveles |
|---|---|
| Checkpoint | `last_epoch`, `best_val_ce` |
| Weight decay | `0`, `1e-4` |

Se congelan AdamW, learning rate `6.25e-4`, batch 64, clipping 5.0, cosine decay y todos los demás
hiperparámetros. `best_val_ce` seleccionará solo con el fold de validación interno; nunca observará
el fold de test.

Como control de reproducibilidad se añadirá `exp02_replay`: máximo 150 épocas, paciencia 35,
`weight_decay=1e-4` y checkpoint por CE de validación. Esta receta no participa en los contrastes
factoriales, pero debe reproducir T7/O0 dentro de la variación entre seeds.

### 4.3 Repeticiones y tamaño

- participantes: 6;
- seeds: 0-4;
- folds: los cinco folds internos congelados;
- trayectorias factoriales: `2 x 6 x 5 x 5 = 300` entrenamientos;
- replay exp02: `1 x 6 x 5 x 5 = 150` entrenamientos;
- celdas de recetas: `5 x 6 x 5 = 150`;
- CSP+LDA: 6 celdas deterministas;
- FBCNet V10 congelado: `6 x 5 = 30` celdas;
- manifiesto de fase A: 186 celdas.

El FBCNet V10 se recalculará con los mismos trials y folds. Si el control `last_epoch/wd0` no es
equivalente al baseline V10 dentro de tolerancia numérica, la fase se detendrá para investigar la
diferencia entre entrenadores.

### 4.4 Decisión de fase A

El modelo de referencia para la fase B será elegido por una regla predeclarada:

1. cero colapsos del clasificador en cualquier fold;
2. mayor accuracy media agregando primero folds, luego seeds y finalmente participantes;
3. si la diferencia es menor de 0.5 puntos porcentuales, mayor cuartil inferior;
4. después, mayor kappa y menor varianza entre seeds;
5. en empate restante, se prefiere `weight_decay=0` y `last_epoch` por simplicidad.

Se informarán contrastes emparejados por participante y seed, IC bootstrap jerárquico y prueba de
permutación por cambio de signo. Con seis participantes, estos resultados siguen siendo evidencia de
selección y no confirmación poblacional.

## 5. Fase B: FBCNet condicionado por calidad

### 5.1 Descriptores por canal

Para cada ventana y canal se calcularán descriptores diferenciables o deterministas:

- log-RMS de amplitud;
- log-RMS de la primera diferencia;
- fracción de diferencias próximas a cero;
- fracción de muestras extremas respecto a mediana y MAD del entrenamiento.

Toda mediana, MAD, umbral y estandarización se ajustará exclusivamente con el fold de entrenamiento.
El test solo aplicará transformaciones congeladas. Se evaluarán tres gates:

| ID | Descriptores | Propósito |
|---|---|---|
| G0 | ninguno | control FBCNet |
| G1 | derivada, flatline, extremos | calidad sin amplitud |
| G2 | G1 + log-RMS | calidad completa; detecta si la amplitud aporta señal adicional |

El gate será un MLP compartido `d -> 8 -> 1`. Su última capa se inicializará en cero. La salida será
`g = 0.5 + 0.5*sigmoid(z)` y se normalizará por la media de gates del trial, de forma que la
inicialización sea exactamente identidad y no pueda mejorar solo reduciendo la escala global.

### 5.2 Channel dropout

Durante entrenamiento se eliminarán canales completos con `p={0,0.1,0.2}`. La máscara será común a
todas las muestras temporales de un canal, tendrá seed registrado, conservará al menos un canal y
no se aplicará en validación ni test. La prueba limpia seguirá siendo primaria.

Además se generará una prueba de estrés secundaria, sin ajuste ni selección, con uno y dos canales
ausentes siguiendo una lista de máscaras deterministas. Esta prueba nunca decidirá por sí sola una
promoción.

### 5.3 Factores de búsqueda

| Factor | Niveles |
|---|---|
| Gate | G0, G1, G2 |
| Channel dropout | 0, 0.1, 0.2 |
| Filtros espaciales por banda | 16, 32, 48 |
| `stride_factor` de varianza | 2, 4, 8 |

Esto produce 81 configuraciones. La fase B usa seed 0, los folds 0, 2 y 4 y los seis participantes:

- celdas candidatas: `81 x 6 = 486`;
- entrenamientos: `486 x 3 = 1,458`;
- el control primario es G0, dropout 0, 32 filtros y `stride_factor=4` con la receta elegida en A.

El screening es deliberadamente amplio. Hasta ocho configuraciones diversas podrán pasar a exp04,
donde se confirmarán con cinco seeds y cinco folds. Exp03 no tocará el holdout.

## 6. Criterios de promoción de fase B

Una configuración será técnicamente elegible si:

1. completa los 18 fits esperados;
2. no presenta NaN, error de convergencia ni colapso del clasificador;
3. contiene todos los trials y ambas clases en predicción y referencia;
4. preserva hashes de datos, folds, receta y código;
5. informa parámetros totales y activos sin módulos inactivos.

La señal mínima de utilidad será:

- `>=+1.0` punto porcentual frente al control primario;
- dirección favorable en al menos 4/6 participantes;
- cuartil inferior no peor en más de 1 punto porcentual;
- kappa no peor en más de 0.02;
- ninguna pérdida limpia superior a 1 punto porcentual a cambio de robustez corrupta.

Se promoverán hasta ocho configuraciones, con máximo dos por tipo de gate y dos por nivel de
dropout. Los empates menores de 0.5 puntos se resolverán por cuartil inferior, kappa, estabilidad
ante canales ausentes, parámetros activos y latencia, en ese orden.

No se declarará una configuración superior por significancia después de seleccionar entre 81
candidatos. Los p-valores y Holm se conservarán como diagnóstico; la prueba confirmatoria se
reserva para exp04/exp05.

## 7. Integridad y recuperación

Cada fase escribirá su propio manifiesto antes de ejecutar:

- `recipe_manifest.json` y `quality_manifest.json`;
- catálogo completo de configuraciones;
- SHA-256 de cada MAT y hashes de tensores estandarizados;
- índices y hashes de folds;
- ambiente, revisión Git y diff limpio;
- decisión firmada de fase A, incluida en el fingerprint de fase B.

Las celdas se escribirán atómicamente. Una celda solo podrá reanudarse si coincide el fingerprint
completo. Un fallo se registrará como resultado firmado y no detendrá el shard. El auditor tendrá
modos `strict` y `portable` y distinguirá:

- colapso del clasificador;
- degeneración de una ruta diagnóstica;
- fallo técnico;
- configuración dominada;
- evidencia inconclusa.

## 8. Paralelización y artefactos

Dos GPUs ejecutarán shards deterministas balanceados por costo. La asignación de GPU no formará
parte de la identidad científica. El lanzador será reanudable y tendrá fases `recipe`, `quality` y
`all`.

Salidas mínimas:

- celdas y checkpoints de ambas fases;
- contrastes factoriales de checkpoint y weight decay;
- curvas de época y distribución de `best_epoch`;
- ranking completo de las 81 configuraciones;
- mapas de gates por canal y participante;
- ablation limpia y bajo canales ausentes;
- fronteras accuracy-parámetros y accuracy-latencia;
- matriz de estabilidad por participante y seed;
- `recipe_decision.json`, `promotion_decision.json` y `run_complete.json`;
- archivos largos CSV/JSON para auditoría y figuras, sin escoger visualizaciones manualmente.

## 9. Criterios de aceptación del código

Antes de ejecutar en Cayetano deben pasar:

1. test unitario que demuestra que G0 es identidad exacta;
2. test que los estadísticos de calidad se ajustan solo con train;
3. test que channel dropout actúa solo en entrenamiento y conserva al menos un canal;
4. test que los checkpoints `last` y `best` provienen de la misma trayectoria;
5. test que O0 no instancia módulos auxiliares;
6. test de paridad del control `last/wd0` con FBCNet V10;
7. test de partición determinista entre dos shards;
8. smoke CPU/MPS local de ambas fases;
9. auditor portable sobre el smoke;
10. suite completa existente sin regresiones.

## 10. Interpretación permitida

Exp03 podrá concluir que una receta o mecanismo es prometedor en la cohorte de descubrimiento. No
podrá afirmar superioridad general, robustez clínica ni novedad algorítmica confirmada. El nombre
provisional será **quality-conditioned FBCNet candidate**. Un nombre propio y cualquier afirmación
de novedad se reservarán hasta completar la búsqueda bibliográfica y la confirmación de exp04/exp05.

## 11. Anclajes bibliográficos

- Mane et al., *FBCNet: A Multi-view Convolutional Neural Network for Brain-Computer Interface*,
  arXiv:2104.01233. Justifica el backbone de filter bank, filtrado espacial y varianza.
- Wimpff et al., *EEG motor imagery decoding: A framework for comparative analysis with channel
  attention mechanisms*, arXiv:2310.11198. Motiva comparar mecanismos de atención bajo un
  backbone común, no asumir su beneficio.

Estos trabajos motivan el diseño, pero no sustituyen la validación emparejada en MI-OpenBCI.
