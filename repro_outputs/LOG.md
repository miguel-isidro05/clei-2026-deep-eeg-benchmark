# Log de auditoría

## 2026-09-27

1. Se leyó el tracker canónico de 35 ejes, el feedback textual más reciente, el handoff MICCAI,
   la documentación del benchmark, el código, los tests y el manuscrito TeX.
2. Se verificó que el cambio solicitado elimina CSP y conserva cinco redes deep.
3. Se ejecutaron tests, lint, preflight, plan-only y un smoke real de MI-OpenBCI.
4. Se comprobó el gate de resultados incompletos en dos rutas: fallo confirmatorio y descriptivos
   exploratorios sin inferencia.
5. Se comparó el plan real con el tracker. Se detectaron tres decisiones P0 previas a la corrida:
   5-fold, definición binaria/cuatro clases de BNCI2014_001 y trazabilidad de recetas/entorno/datos.
6. No se modificó código científico ni se generaron resultados para el paper.

Artefactos temporales de ejecución: `/tmp/clei-scientific-audit.BVbjNz`.

## 2026-09-28

1. Se implementaron controles de augmentación emparejados por ejemplos y actualizaciones.
2. Se congeló y auditó el perfil de 300 épocas, con rechazo explícito de ejecuciones piloto.
3. Se añadieron trazabilidad por fold, calidad de señal, figuras, tablas LaTeX y registro durable.
4. `pytest` terminó con 60 pruebas aprobadas y Ruff sin hallazgos.
5. El preflight local terminó en `preflight=OK` para los cinco modelos y auditó los diez sujetos
   MI-OpenBCI sin valores no finitos.
6. El plan confirmatorio produjo `planned_cells=3600` y `--epochs 3` fue rechazado.
7. No se ejecutó la matriz completa localmente. Los resultados, ganadores y significancia siguen
   pendientes de la corrida en las dos RTX A6000.
8. `pip-audit` encontró avisos en dependencias del entorno científico, incluidas versiones de
   `aiohttp`, Pillow, pip, setuptools y PyTorch. No se cambiaron silenciosamente porque actualizar
   PyTorch modificaría el entorno numérico congelado; deben revisarse en una actualización de
   entorno separada antes de distribuir una imagen de ejecución a terceros.

## 2026-09-29

1. La revision independiente final detecto cuatro riesgos: decision de modelos no registrada como
   fuente, salida primaria usando `within_split`, preflight sin comprobar dos GPU y matriz de
   confusion generada pero no incluida en el bloque LaTeX.
2. Se documento que el conjunto EEGNet/FBCNet/ShallowConvNet/EEGConformer/EEGInceptionMI es una
   decision explicita del autor para esta revision.
3. `generate_manuscript_results.py` ahora toma `within_session` cinco-fold como tabla pareada
   primaria y marca las columnas como held-out test.
4. La matriz de confusion primaria usa celdas `within_session` y queda incluida en el bloque LaTeX.
5. `scripts/run_cayetano.sh` exige `--min-cuda-devices 2` en el preflight.
6. Verificacion posterior: 60 pruebas aprobadas, Ruff aprobado, `bash -n`, `git diff --check` y
   plan confirmatorio `planned_cells=3600`.
7. Las pruebas de integridad alteraron de forma intencional un CSV y un PDF; ambos intentos fueron
   bloqueados antes de generar el TeX.
