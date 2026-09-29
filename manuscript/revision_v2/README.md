# Fuente canónica revision_v2

Esta carpeta reemplaza como fuente activa al TeX histórico ubicado en `Paper_final/`. El archivo
histórico permanece inmutable porque contiene el envío y resultados anteriores.

`paper_CLEI2026.tex` describe el diseño multisemilla actual. No contiene CSP, DeepConvNet,
Friedman, bootstrap ni resultados de seed 0. Mientras `generated_results.tex` conserve el aviso de
pendiente, esta fuente es un borrador metodológico y no debe enviarse.

Los valores numéricos solo se incorporarán después de que `scripts/run_statistics.py` confirme:

- ninguna celda esperada faltante;
- cinco seeds por sujeto, modelo y condición;
- cohortes idénticas para cada contraste pareado;
- tablas Wilcoxon-Holm, intervalos y tamaños de efecto completas.

Tras una corrida válida, `scripts/generate_manuscript_results.py` sustituye automáticamente el
aviso de pendiente por tablas y referencias a figuras generadas desde los mismos resultados. El
gate exige los manifiestos con hash de estadísticas, figuras y latencia. El TeX incluye la tabla
principal de Wilcoxon-Holm con IC, tamaño de efecto y decisión corregida.

Compilación de control:

```bash
cd manuscript/revision_v2
pdflatex -interaction=nonstopmode -halt-on-error paper_CLEI2026.tex
pdflatex -interaction=nonstopmode -halt-on-error paper_CLEI2026.tex
```
