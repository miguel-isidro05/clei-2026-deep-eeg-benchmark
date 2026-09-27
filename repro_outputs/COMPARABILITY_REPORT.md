# Reporte de comparabilidad

## Dentro de cada dataset

La comparación entre modelos es estructuralmente válida: mismo split, input temporal, condiciones,
seed de partición, seed ICA y presupuesto de epochs. La seed de optimización cambia entre
repeticiones y es compartida entre modelos para cada celda lógica.

## Entre datasets

MI-OpenBCI, Zhou2020 y BNCI2014_001 no son una comparación causal de hardware. Cambian sujetos,
tarea, montaje, número de sesiones y protocolo de adquisición. Zhou2020 aproxima mejor MI versus
rest; BNCI2014_001 aporta cross-session canónico, pero el código usa left versus right binario.
Cada dataset debe tener su propia tabla e inferencia.

## Contra el estudio histórico

El nuevo benchmark no es directamente comparable con las tablas actuales del TeX. Cambian el
conjunto de modelos, las seeds, el soporte temporal, el uso primario de ICA, las condiciones de
augmentación y el plan estadístico. Los números viejos no deben trasladarse a la revisión.

## Desviaciones registradas

- CSP y DeepConvNet fueron retirados por decisión del autor.
- EEGInceptionMI fue añadido.
- El primario usa trials completos para las cinco redes.
- ICA por kurtosis es sensibilidad exploratoria, no preprocessing primario.
- No se usan Friedman ni bootstrap; se usa Wilcoxon bilateral con Holm e IC t.
- BNCI2014_001 se redujo a left-versus-right binario, mientras el tracker histórico pedía cuatro
  clases. Esta desviación requiere aprobación metodológica explícita antes de la corrida completa.
