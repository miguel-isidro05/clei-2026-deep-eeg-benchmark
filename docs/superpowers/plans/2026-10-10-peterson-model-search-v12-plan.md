# Peterson Model Search V12: plan de implementación TDD

Fecha: 2026-10-10  
Rama: `exp/peterson-model-search-v12`  
Worktree: `benchmark_deep_v2_search_v12`  
Especificación: `docs/superpowers/specs/2026-10-10-peterson-model-search-v12-design.md`  
Entorno local: `eeg-diffusion`  
Entorno Cayetano: `deep-eeg-clei`

## 1. Resultado de esta implementación

Esta fase entregará la infraestructura común V12 y un `exp01` ejecutable en dos GPU. No generará
`exp02-exp05` antes de recibir, auditar y aprobar el resultado de `exp01`.

El entregable debe producir:

- catálogo inmutable de 36 configuraciones `exp01`;
- 216 celdas esperadas, correspondientes a 36 configuraciones y seis sujetos de descubrimiento;
- entrenamiento sobre folds 0 y 2, seed 0, hasta 80 épocas;
- selección basada solo en validación interna;
- ausencia material y de esquema de predicciones del test exterior;
- smoke dual-GPU separado;
- ejecución reanudable con fingerprints;
- auditor estricto y readout de promoción de 12 configuraciones;
- bloqueo explícito de `exp02-exp05` sin una decisión válida de la etapa anterior.

## 2. Restricciones

1. No modificar `src/deepbench`, Peterson V10 ni `experiments/peterson_diffusion_v11`.
2. No usar Souza2023, LOSO, cross-session ni otras condiciones durante V12.
3. No usar métricas del test para entrenar, detener, ordenar o promover candidatos.
4. No presentar los smokes como evidencia.
5. No introducir dependencias nuevas si PyTorch, NumPy, SciPy o la biblioteca estándar cubren la
   función.
6. No descargar datasets.
7. No crear `exp02` hasta que `exp01` tenga export, checksum, auditoría y decisión firmada.

## 3. Arquitectura de archivos

```text
experiments/peterson_search_v12/
├── __init__.py
├── common/
│   ├── __init__.py
│   ├── audit.py
│   ├── catalogs.py
│   ├── config.py
│   ├── identity.py
│   ├── io.py
│   ├── manifests.py
│   ├── models.py
│   ├── objectives.py
│   ├── partitions.py
│   ├── promotion.py
│   ├── runner.py
│   ├── splits.py
│   └── training.py
└── exp01_architecture_formulation/
    ├── README.md
    └── config.json

scripts/
├── analyze_peterson_search_v12.py
├── check_peterson_search_v12.py
└── run_peterson_search_v12.py

run_peterson_search_cayetano.sh

tests/peterson_search_v12/
├── test_audit.py
├── test_catalogs.py
├── test_cli_and_launcher.py
├── test_config.py
├── test_manifests.py
├── test_models.py
├── test_objectives.py
├── test_partitions.py
├── test_promotion.py
├── test_runner.py
├── test_splits.py
└── test_training.py
```

Los módulos `identity.py` e `io.py` podrán envolver funciones V11 sin modificar V11. Si una función
compartida no cumple el contrato V12, se implementará localmente en V12.

## 4. Contratos congelados de exp01

### Sujetos

El hash `SHA256("peterson-v12-subject-partition-20261010:<subject>")` debe producir:

- descubrimiento: `S09`, `S03`, `S02`, `S10`, `S12`, `S08`;
- holdout: `S07`, `S04`, `S05`, `S06`.

El runner `exp01` rechazará sujetos holdout aunque el usuario los pase por CLI.

### Catálogo

El orden será determinista:

1. backbone: `residual`, `tcn`, `inception`, `filterbank`, `smooth_basis`, `conformer_lite`;
2. formulación: `discriminative`, `diffusion_energy`, `hybrid`;
3. anchura: `compact`, `wide`.

Cada combinación tendrá un `config_id` estable `exp01-cfg-001` a `exp01-cfg-036` y un hash del JSON
canónico. Cambiar cualquier parámetro debe cambiar el hash.

### Presupuesto

- subjects: seis de descubrimiento;
- folds: 0 y 2;
- seed: 0;
- condición: `overlap`;
- epochs: 80;
- patience: 20;
- batch size: 64;
- split seed: 2026.

El manifiesto tendrá 216 IDs únicos. Cada celda representa una configuración y un sujeto; los dos
folds se ejecutan dentro de la celda.

## 5. Secuencia TDD

### Tarea 1. Salvaguardas, configuración y partición de sujetos

Crear:

- `experiments/peterson_search_v12/__init__.py`;
- `experiments/peterson_search_v12/common/__init__.py`;
- `experiments/peterson_search_v12/common/config.py`;
- `experiments/peterson_search_v12/common/partitions.py`;
- `tests/peterson_search_v12/test_config.py`;
- `tests/peterson_search_v12/test_partitions.py`.

Pruebas RED:

1. aceptar solo Peterson, MI vs rest, `within_session` y `overlap`;
2. rechazar LOSO, Souza2023, sujetos desconocidos y sujetos holdout en `exp01`;
3. reproducir exactamente la partición 6/4 predeclarada;
4. rechazar solapamiento o ausencia de sujetos entre discovery y holdout;
5. impedir que `exp02-exp05` carguen sin un archivo de decisión anterior.

Implementación GREEN:

- dataclasses congeladas para programa, etapa y presupuesto;
- función pura de partición por SHA-256;
- validación de campos y rutas de decisión;
- `.gitignore` para `results_peterson_search_v12/` si no existe una regla equivalente.

Commits:

```text
test(v12): define search configuration and subject partition
feat(v12): establish search contracts and holdout boundary
```

### Tarea 2. Catálogo exp01 y manifiesto

Crear:

- `experiments/peterson_search_v12/common/catalogs.py`;
- `experiments/peterson_search_v12/common/manifests.py`;
- `experiments/peterson_search_v12/exp01_architecture_formulation/config.json`;
- `experiments/peterson_search_v12/exp01_architecture_formulation/README.md`;
- `tests/peterson_search_v12/test_catalogs.py`;
- `tests/peterson_search_v12/test_manifests.py`.

Pruebas RED:

1. generar exactamente 36 configuraciones y 216 celdas;
2. cubrir el producto 6 × 3 × 2 sin duplicados;
3. producir IDs y hashes idénticos sin depender del orden de diccionarios;
4. cambiar el hash cuando cambia un hiperparámetro;
5. dividir dos shards disjuntos de 108 celdas cuya unión sea el manifiesto completo;
6. registrar únicamente folds 0 y 2 y seed 0;
7. impedir que una celda `exp01` incluya un sujeto holdout.

Implementación GREEN:

- catálogo materializado como JSON ordenado;
- IDs legibles y hash canónico;
- manifiesto esperado con hash propio, revisión base y fingerprints;
- función de sharding determinista.

Commits:

```text
test(v12): freeze exp01 catalog and manifest matrix
feat(v12): generate immutable exp01 search catalog
```

### Tarea 3. Splits de validación sin acceso al test

Crear:

- `experiments/peterson_search_v12/common/splits.py`;
- `tests/peterson_search_v12/test_splits.py`.

Pruebas RED:

1. reproducir los splits exteriores V10 para folds 0 y 2;
2. derivar inner-train e inner-validation solo de los índices outer-train;
3. demostrar disjunción entre inner-train, inner-validation y outer-test;
4. mantener todas las ventanas de un trial en el mismo conjunto;
5. no devolver arrays del outer-test a las funciones de cribado;
6. generar el mismo hash ante el mismo sujeto, fold y split seed;
7. cambiar el hash cuando cambia cualquier índice.

Implementación GREEN:

- objeto `SearchSplit` sin datos ni etiquetas de test;
- objeto privado de auditoría con el hash del outer-test, pero sin exponer sus valores al trainer;
- estratificación determinista del inner-validation;
- validación explícita de índices y trial provenance.

Commits:

```text
test(v12): specify validation-only search splits
feat(v12): isolate inner validation from outer test
```

### Tarea 4. Registro de backbones

Crear:

- `experiments/peterson_search_v12/common/models.py`;
- `tests/peterson_search_v12/test_models.py`.

Backbones:

- residual temporal;
- TCN dilatada;
- inception multikernel;
- filter-bank temporal-espacial;
- SmoothBasis;
- Conformer ligero.

Pruebas RED:

1. construir las seis familias en anchuras compacta y amplia;
2. aceptar `(batch, channels, time)` y conservar el batch;
3. producir embeddings de dimensión declarada;
4. comprobar gradientes finitos en CPU para cada backbone;
5. comprobar que compacta tiene menos parámetros que amplia;
6. prohibir nombres o anchuras no registrados;
7. mantener los parámetros dentro de límites predeclarados para low-cost;
8. ejecutar dos forwards idénticos bajo seed y modo evaluación.

Implementación GREEN:

- interfaz común `encode(x, channel_mask)`;
- bloques pequeños y aislados;
- inicialización explícita;
- `parameter_report` por configuración;
- sin importar modelos V10 salvo FBCNet en las pruebas baseline.

Commits:

```text
test(v12): define six low-cost EEG backbone contracts
feat(v12): implement exp01 backbone registry
```

### Tarea 5. Formulaciones y objetivos

Crear:

- `experiments/peterson_search_v12/common/objectives.py`;
- ampliar `models.py` con heads discriminativo, diffusion-energy e híbrido;
- `tests/peterson_search_v12/test_objectives.py`.

Pruebas RED:

1. entropía cruzada discriminativa finita;
2. energía de difusión finita para ambas clases;
3. pérdida híbrida con términos clasificación, denoising, ranking y consistencia;
4. lambda cero elimina exactamente el término correspondiente;
5. aumentar energía de la clase incorrecta mejora el margen en la dirección prevista;
6. channel masks no alteran dimensiones ni generan NaN;
7. cada formulación usa el mismo backbone y presupuesto cuando se compara emparejada;
8. las tres formulaciones producen probabilidades binarias normalizadas.

Implementación GREEN:

- heads separados sobre un backbone común;
- schedule de difusión reutilizable de V11 cuando conserve el mismo contrato;
- registro explícito de todos los términos de pérdida;
- inferencia energética determinista bajo seed fijo.

Commits:

```text
test(v12): define discriminative diffusion and hybrid objectives
feat(v12): implement paired exp01 formulations
```

### Tarea 6. Entrenamiento con selección por validación

Crear:

- `experiments/peterson_search_v12/common/training.py`;
- `tests/peterson_search_v12/test_training.py`.

Pruebas RED:

1. early stopping observa solo inner-validation;
2. restaurar el checkpoint de menor pérdida de validación;
3. no aceptar argumentos o callbacks de outer-test;
4. contar épocas, batches y actualizaciones;
5. detenerse ante pérdidas o gradientes no finitos;
6. reproducir métricas y pesos con el mismo seed;
7. cambiar resultados al cambiar seed en un modelo estocástico;
8. escribir checkpoints atómicamente;
9. un checkpoint recargado reproduce logits de validación;
10. eliminar objetos CUDA y liberar cache al terminar una celda.

Implementación GREEN:

- loop PyTorch explícito;
- AdamW con receta exp01 fija;
- cosine scheduler, clipping y patience;
- AMP solo cuando el dispositivo lo soporte y quede registrado;
- historial completo de entrenamiento y validación.

Commits:

```text
test(v12): specify validation-only training lifecycle
feat(v12): implement reproducible exp01 training
```

### Tarea 7. Runner reanudable y esquema de celda

Crear:

- `experiments/peterson_search_v12/common/identity.py`;
- `experiments/peterson_search_v12/common/io.py`;
- `experiments/peterson_search_v12/common/runner.py`;
- `scripts/run_peterson_search_v12.py`;
- `tests/peterson_search_v12/test_runner.py`.

Pruebas RED:

1. plan-only crea manifiesto sin cargar EEG;
2. celda válida requiere ID, hashes, estado y fingerprint exactos;
3. una celda parcial, corrupta o de código anterior se recalcula;
4. el resultado `exp01` contiene métricas de validación y no contiene `y_test`, `test_accuracy`,
   `test_metrics` ni predicciones exteriores;
5. el runner usa solo sujetos discovery y folds 0 y 2;
6. dos shards son disjuntos y reanudables;
7. escritura JSON y Torch es atómica;
8. el error de una celda termina el shard con código distinto de cero;
9. el resultado registra versiones, hardware, memoria, tiempos y conteos.

Implementación GREEN:

- fingerprint de configuración, catálogo, código, datos, split, entorno y dispositivo;
- payload validable por esquema;
- progreso por celda en stdout;
- `--plan-only`, `--num-shards`, `--shard-index` y `--max-cells-per-shard` solo para smoke.

Commits:

```text
test(v12): define resumable validation-only cells
feat(v12): implement exp01 runner and fingerprints
```

### Tarea 8. Auditor y promoción

Crear:

- `experiments/peterson_search_v12/common/audit.py`;
- `experiments/peterson_search_v12/common/promotion.py`;
- `scripts/check_peterson_search_v12.py`;
- `scripts/analyze_peterson_search_v12.py`;
- `tests/peterson_search_v12/test_audit.py`;
- `tests/peterson_search_v12/test_promotion.py`.

Pruebas RED:

1. rechazar celdas faltantes, extra, duplicadas o inválidas;
2. rechazar sujetos holdout y folds fuera de `{0, 2}`;
3. rechazar cualquier métrica o predicción del outer-test;
4. rechazar hashes, recetas o splits inesperados;
5. rechazar colapso, NaN o historial incompleto;
6. ordenar configuraciones por accuracy media de validación por sujeto;
7. usar cuartil inferior, parámetros y latencia como desempates predeclarados;
8. promover 12 configuraciones, máximo tres por backbone y al menos dos formulaciones cuando sean
   viables;
9. generar decisión con hashes del manifiesto, resultados y código;
10. impedir que un readout parcial produzca una decisión final.

Implementación GREEN:

- auditor de esquema e integridad;
- tabla completa de 36 configuraciones;
- ranking y Pareto accuracy-coste;
- archivo `decisions/exp01.json` solo después de 216/216 celdas válidas;
- no generar aún catálogo exp02.

Commits:

```text
test(v12): define exp01 audit and promotion gates
feat(v12): implement exp01 audit and readout
```

### Tarea 9. Launcher dual-GPU y smoke

Crear:

- `run_peterson_search_cayetano.sh`;
- `tests/peterson_search_v12/test_cli_and_launcher.py`;
- actualizar `docs/CAYETANO_RUNBOOK.md`.

Pruebas RED:

1. exigir exactamente dos dispositivos CUDA;
2. exigir `PETERSON_SEARCH_EXPERIMENT=exp01`;
3. lanzar shards 0 y 1 y propagar el fallo de cualquiera;
4. smoke ejecuta una celda de modelo profundo por GPU y una época;
5. smoke usa manifiesto propio y auditor completo;
6. smoke no puede escribir `run_complete.json`;
7. corrida completa exige árbol Git limpio;
8. auditoría completa produce `run_complete.json` solo con 216/216 celdas;
9. no encadenar automáticamente `exp02`.

Implementación GREEN:

- launcher resumable con logs por shard;
- preflight de datos y CUDA;
- `PETERSON_SEARCH_SMOKE=1` como único perfil reducido;
- directorios diferentes para smoke y resultado real;
- cierre atómico y timestamp UTC.

Commits:

```text
test(v12): define dual-GPU exp01 launcher contract
feat(v12): add audited Cayetano exp01 launcher
```

### Tarea 10. Verificación final y publicación

Ejecutar:

```bash
conda run -n eeg-diffusion pytest -q tests/peterson_search_v12
conda run -n eeg-diffusion pytest -q tests
conda run -n eeg-diffusion ruff check \
  experiments/peterson_search_v12 \
  scripts/run_peterson_search_v12.py \
  scripts/check_peterson_search_v12.py \
  scripts/analyze_peterson_search_v12.py \
  tests/peterson_search_v12
python -m compileall -q experiments/peterson_search_v12 scripts
bash -n run_peterson_search_cayetano.sh
git diff --check de0813d...HEAD
git diff --exit-code de0813d...HEAD -- src/deepbench experiments/peterson_diffusion_v11
```

Verificaciones adicionales:

1. plan-only declara 36 configuraciones y 216 celdas;
2. smoke CPU sintético cubre las 36 configuraciones;
3. no existe ninguna clave de test exterior en un payload de cribado;
4. todos los IDs, catálogos y manifiestos son deterministas;
5. revisión separada de especificación y estándares;
6. `pip-audit` informativo, sin cambiar el entorno científico durante la auditoría;
7. árbol Git limpio;
8. push solo después de revisar el diff y confirmar el commit remoto.

Commit de cierre si la verificación exige ajustes documentales:

```text
docs(v12): finalize exp01 Cayetano runbook
```

## 6. Criterios de aceptación

La implementación estará lista para Cayetano cuando:

- todas las pruebas estén verdes;
- el plan-only produzca 36 configuraciones y 216 celdas;
- las 36 variantes pasen forward, backward y determinismo sintético;
- ningún camino de `exp01` pueda leer o escribir predicciones del outer-test;
- el smoke dual-GPU tenga manifiesto y auditor propios;
- V10, V11 y `src/deepbench` permanezcan sin cambios;
- la rama remota apunte al mismo commit verificado localmente.

## 7. Trabajo posterior bloqueado

`exp02` se diseñará y versionará únicamente después de:

1. completar `exp01` en Cayetano;
2. transferir `.tar.gz` y `.sha256` al Mac;
3. validar checksum, manifiesto, hashes, splits y ausencia de test;
4. ejecutar el ranking predeclarado;
5. revisar y aprobar `decisions/exp01.json`.

La misma regla se repetirá para `exp03`, `exp04` y `exp05`.
