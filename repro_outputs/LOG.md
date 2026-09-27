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
