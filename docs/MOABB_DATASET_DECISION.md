# Decision sobre datasets y utilidades MOABB

MOABB ya ofrece `CrossSessionEvaluation`, cuyo splitter deja una sesion completa para test y usa
las restantes para training. El benchmark adopta exactamente esa semantica para Zhou2020 y
BNCI2014_001. Se conserva un evaluador propio porque el experimento necesita cinco seeds,
preprocesamiento e ICA ajustados dentro de cada fold, predicciones por trial, cache por fold e
informes que la interfaz estandar de MOABB no conserva con este nivel de detalle.

## Seleccion final

| Dataset | Sujetos/sesiones | Clases usadas | Funcion en el paper |
|---|---:|---|---|
| MI-OpenBCI | 10/1 | motor imagery vs rest | Resultado low-cost principal |
| Zhou2020 | 20/7 | right hand vs rest | Validacion externa task-matched y cross-session |
| BNCI2014_001 | 9/2 | left hand vs right hand | Benchmark canonico BCI Competition IV 2a cross-session |
| AlexMI | 8/1 | right hand vs rest | Smoke test, no tabla principal |

Zhou2020 es la comparacion externa mas cercana a la tarea MI frente a reposo. BNCI2014_001 aporta
el protocolo ampliamente reconocible de BCI IV 2a, pero su tarea binaria no es la misma. Por ello
ninguna diferencia entre estos datasets se atribuye causalmente al costo del hardware.

Fuentes oficiales:

- MOABB y ejemplo oficial de `CrossSessionEvaluation`: https://github.com/NeuroTechX/moabb
- Implementacion de Zhou2020: https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/zhou2020.py
- Documentacion de AlexMI: https://moabb.neurotechx.com/docs/generated/moabb.datasets.AlexMI.html
- EEGInceptionMI en Braindecode: https://github.com/braindecode/braindecode/blob/master/braindecode/models/eeginception_mi.py
