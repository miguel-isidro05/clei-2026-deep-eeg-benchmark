# Decision sobre datasets y utilidades MOABB

MOABB ya ofrece `CrossSessionEvaluation`, cuyo splitter deja una sesion completa para test y usa
las restantes para training. El benchmark adopta exactamente esa semantica para Zhou2020 y
Tavakolan2017. Se conserva un evaluador propio porque el experimento necesita cinco seeds,
preprocesamiento e ICA ajustados dentro de cada fold, predicciones por trial, cache por fold e
informes que la interfaz estandar de MOABB no conserva con este nivel de detalle.

## Seleccion final

| Dataset | Sujetos/sesiones | Clases usadas | Funcion en el paper |
|---|---:|---|---|
| MI-OpenBCI | 10/1 | motor imagery vs rest | Resultado low-cost principal |
| Zhou2020 | 20/7 | right hand vs rest | Validacion externa task-matched y cross-session |
| Tavakolan2017 | 12/4 | right hand vs rest | Validacion externa task-matched y computacionalmente manejable |
| AlexMI | 8/1 | right hand vs rest | Smoke test, no tabla principal |

Zhou2020 y Tavakolan2017 contienen exactamente los eventos `right_hand` y `rest`, ademas de varias
sesiones por sujeto. BNCI2014_001 permanece soportado como dataset opcional, pero se retiro del
perfil confirmatorio porque no contiene `rest`; reducir sus cuatro clases a left-versus-right
contestaria otra pregunta. Ninguna diferencia entre datasets se atribuye causalmente al costo del
hardware.

Fuentes oficiales:

- MOABB y ejemplo oficial de `CrossSessionEvaluation`: https://github.com/NeuroTechX/moabb
- Implementacion de Zhou2020: https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/zhou2020.py
- Documentacion de AlexMI: https://moabb.neurotechx.com/docs/generated/moabb.datasets.AlexMI.html
- EEGInceptionMI en Braindecode: https://github.com/braindecode/braindecode/blob/master/braindecode/models/eeginception_mi.py
