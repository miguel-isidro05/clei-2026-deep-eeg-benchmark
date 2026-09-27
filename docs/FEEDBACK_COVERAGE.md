# Cobertura honesta de la retroalimentacion

| Observacion | Solucion implementada | Evidencia generada | Estado antes de ejecutar el perfil paper |
|---|---|---|---|
| Cohorte low-cost pequena | Se evita pseudorreplicacion, se informa cada sujeto y se añade validacion externa task-matched en 20 sujetos Zhou2020 y 12 sujetos Tavakolan2017 | Tablas separadas por dataset/protocolo | Mitigado, no cerrado: MI-OpenBCI sigue teniendo 10 sujetos y las seeds no aumentan el n biologico |
| Una sola semilla limita la robustez | Cinco semillas independientes por modelo, sujeto y protocolo | JSON por celda, tabla de variabilidad entre sujetos y entre semillas | Implementado; pendiente completar computo GPU |
| Falta inferencia pareada y correccion multiple | Wilcoxon bilateral por sujeto y Holm por familia predeclarada | `paired_wilcoxon_holm.csv` | Implementado; pendiente resultados completos |
| Faltan intervalos de confianza | IC 95% t de Student para medias y diferencias pareadas | Tabla descriptiva y tabla pareada | Implementado; pendiente resultados completos |
| Sliding windows puede favorecer modelos o inflar n | Mismas condiciones para todos los modelos; test conserva un voto por trial; sujeto es la unidad estadistica | Predicciones y hashes de indices por fold | Cerrado en codigo |
| Falta cross-session | Leave-one-session-out para Zhou2020 y Tavakolan2017 | Celdas `cross_session` | Implementado; pendiente descarga y computo |
| ICA no explica como selecciona componentes | Se elimina ICA del analisis primario. La sensibilidad ajusta FastICA solo con training y usa kurtosis > 10, maximo dos, sin seleccion manual | Convergencia, iteraciones y kurtosis por fold | Parcial: auditable, pero no identifica ocular/muscular; no se hace ese claim |
| Falta sensibilidad a ICA | El perfil primario `ica_policy=none` se contrasta con una sensibilidad exploratoria `kurtosis` | Resultados separados por politica ICA | Implementado; pendiente computo |
| Latencia poco clara | Perfil model-only con mismo dispositivo, forma, batch, warm-up e iteraciones | `latency.csv` y `latency.json` | Cerrado para latencia computacional; no mide latencia BCI end-to-end |
| Comparacion con datasets MOABB | Zhou2020 y Tavakolan2017 para right-hand vs rest y cross-session | Resultados por dataset y protocolo, sin mezclar inferencia | Implementado; no permite atribuir diferencias causalmente al costo del hardware |
| Reproducibilidad insuficiente | Configuracion unica, manifiestos, versions, seeds separadas, predicciones, hashes de codigo/configuracion y reanudacion por fold | `results/manifests`, `results/cells`, `results/fold_cache` | Cerrado en infraestructura; requiere publicar los artefactos finales |

## Lo que el codigo no puede cerrar por si solo

Los claims numericos permanecen abiertos hasta completar todas las celdas con cinco semillas. El
benchmark externo no convierte datasets heterogeneos en una comparacion causal de hardware. La
latencia informada es de inferencia del modelo y no incluye adquisicion EEG, espera para acumular
la ventana, transmision, interfaz ni actuacion. Finalmente, la seleccion de componentes por
kurtosis es auditable y reproducible, pero no demuestra que todo componente excluido sea ocular o
muscular; por ello se incluye la sensibilidad sin ICA y no se describe como identificacion clinica
del artefacto.
