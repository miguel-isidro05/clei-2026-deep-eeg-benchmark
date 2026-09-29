# Scientific changelog

## 2026-09-27

- Se auditó el diseño contra 35 ejes de feedback.
- No se modificaron datos, modelos, preprocessing, splits, seeds, métricas ni estadística.
- Se creó evidencia documental de cumplimiento, faltantes y decisiones pendientes.
- No se aplicaron parches al código científico.

## 2026-09-28

- Se reemplazó la comparación de ventanas confundida por dos contrastes con presupuesto igualado:
  `nonoverlap-center_x2` y `overlap-center_x6`.
- El perfil confirmatorio quedó bloqueado a 300 épocas y 3.600 celdas. El auditor verifica receta,
  identidad, semillas, historial completo por fold y hashes de resultados derivados.
- Se evitó repetir la banda 1-40 Hz en datos que MOABB ya entrega filtrados.
- Se añadieron hashes posteriores al preprocesamiento y de las entradas finales del modelo.
- Se añadió control no destructivo de calidad de señal, figuras PNG/PDF, generación de tablas LaTeX
  y logs de consola con resumen Markdown.
- Estos cambios modifican el diseño de augmentación respecto de la versión anterior, pero conservan
  los modelos, datasets, splits, métricas y unidad estadística predeclarados.
