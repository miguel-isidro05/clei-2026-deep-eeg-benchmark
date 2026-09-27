# Resumen de auditoría científica

## Dictamen

El pipeline es ejecutable y cubre correctamente el núcleo solicitado: cinco decoders deep,
cinco semillas, particiones fijas, Wilcoxon pareado por sujeto, corrección de Holm, intervalos
de confianza sin bootstrap, evaluación cross-session, ICA train-only como sensibilidad y
latencia de inferencia controlada.

Todavía no debe usarse para cerrar todas las respuestas de los revisores ni para producir la
versión final del paper. Antes de gastar el presupuesto completo de GPU deben resolverse tres
decisiones que cambian el diseño o su trazabilidad:

1. El perfil del paper no ejecuta el protocolo `within_session` de cinco folds que los revisores
   reconocieron como una fortaleza y que el manuscrito aún reporta.
2. La procedencia de las recetas e hiperparámetros no está congelada en una tabla auditable, y la
   compatibilidad de celdas no incorpora versiones de dependencias ni identidad de los datos.
3. El tracker canónico pide BNCI2014_001 de cuatro clases, mientras el código implementa una tarea
   binaria left-hand versus right-hand. Esta desviación puede ser válida, pero debe decidirse y
   preespecificarse antes de ejecutar.

Los resultados y las afirmaciones de superioridad permanecen abiertos hasta completar la matriz
multisemilla. El manuscrito activo no es compatible con el benchmark nuevo: todavía contiene CSP,
DeepConvNet, una seed, Friedman, bootstrap y resultados históricos.

## Evidencia de ejecución

- 26 pruebas aprobadas.
- `ruff check .` aprobado.
- Preflight aprobado para EEGNet, FBCNet, ShallowConvNet, EEGConformer y EEGInceptionMI.
- Smoke real MI-OpenBCI/S02/EEGNet, una época, aprobado.
- El plan completo genera 2,950 celdas esperadas.
- El análisis confirmatorio se bloquea cuando faltan celdas.
- Una ejecución incompleta solo genera descriptivos marcados como exploratorios.

La matriz completa está en `SCIENTIFIC_AUDIT.md`.
