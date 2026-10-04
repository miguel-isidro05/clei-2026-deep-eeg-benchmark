# Auditoría técnica de transferencia v8 y plan v9

## Veredicto

La v8 está completa y sus métricas almacenadas coinciden con las predicciones, pero debe
tratarse como evidencia diagnóstica, no como resultado final del journal. La inicialización
de los checkpoints fuente no quedó determinada por la semilla declarada, la partición interna
de Souza cambió entre semillas, EEGConformer usó atención CUDA no determinista y la sexta
ventana Peterson terminaba una muestra antes del final del trial.

La v9 repara estos cuatro puntos y conserva la v8 sin sobrescribirla.

## Hechos observados en v8

- Integridad: 25 checkpoints fuente, 2,000 celdas Peterson y 750 celdas Souza; ninguna
  discrepancia al recalcular accuracy, kappa, F1 macro, precision macro, recall macro y AUC.
- Peterson, régimen principal overlap:
  - within-session: FBCNet 83.30%, ShallowConvNet 79.66%, EEGConformer 78.72%, EEGNet
    78.22% y EEGInceptionMI 73.21%;
  - within-split: FBCNet 82.38%, ShallowConvNet 80.79%, EEGNet 80.02%, EEGConformer
    79.31% y EEGInceptionMI 75.24%.
- El overlap supera al full trial en within-session para los cinco modelos. La ganancia media
  es +8.24 puntos para EEGConformer, +4.63 para EEGNet, +2.32 para FBCNet, +2.28 para
  ShallowConvNet y +0.67 para EEGInceptionMI.
- ICA por kurtosis cambia menos de un punto porcentual en todos los modelos. No aporta una
  mejora consistente.
- Souza, EEGNet:
  - within-session: scratch 50.25%, linear probe 55.40% y full fine-tune 56.38%;
  - cross-session: scratch 51.15%, linear probe 55.05% y full fine-tune 56.30%.
- La mejora within-session de EEGNet ocurre en 5/5 sujetos. En cross-session ocurre en 3/5
  y el sujeto 003 aporta la mayor parte de la media.
- FBCNet scratch es el mejor resultado cross-session de Souza, 59.13%; su transferencia
  supervisada no mejora esa base.
- Ningún contraste sobre cinco sujetos sobrevive Holm. Con n=5, el mínimo p bilateral exacto
  es 0.0625; la evidencia debe expresarse mediante tamaño de efecto, intervalo, consistencia
  direccional y replicación entre protocolos.

## Interpretación

La señal actual es compatible con transferencia útil para EEGNet, no con una ventaja general
de preentrenar cualquier CNN o transformer. EEGInceptionMI seleccionó entre una y tres épocas
en casi todos los checkpoints fuente, por lo que su representación fuente probablemente fue
débil. Esta interpretación debe verificarse con métricas del sujeto fuente retenido, añadidas
en v9.

El fenómeno fisiológico relevante es ERD/ERS de ritmos sensorimotores mu y beta, no un ERP
clásico bloqueado a estímulo. La hipótesis mecanística se formulará como transferencia de una
representación espectro-espacial sensorimotora compartida entre MI-versus-rest y
izquierda-versus-derecha.

## Reparación confirmatoria v9

La v9 repite la matriz completa de 2,775 artefactos en una carpeta nueva y exige:

1. Semilla fijada antes de construir cada modelo fuente y objetivo.
2. Partición interna fija por fold, independiente de la semilla de optimización.
3. Algoritmos Torch deterministas estrictos y atención matemática para EEGConformer.
4. Seis ventanas que cubren exactamente el trial completo.
5. Checkpoint de mejor validación restaurado y métricas del sujeto fuente retenido registradas.
6. Hash del entorno, política de ventanas, receta y código ejecutor en cada manifiesto.
7. Tres contrastes pareados, IC95%, mediana, rank-biserial y valores por sujeto.
8. ERD/ERS con log-potencia, conteos, intervalos, runs y lateralización contralateral.
9. Rechazo de un worktree sucio, artefactos obsoletos o entornos numéricos mezclados.

## Experimentos exploratorios posteriores a v9

Estos bloques no se mezclarán con el confirmatorio. Toda elección se hará con el inner train y
validation; el outer test permanecerá cerrado.

### H1: el beneficio procede de fisiología aprendida, no solo de inicialización

- Preentrenamiento Peterson con etiquetas reales.
- Control negativo con etiquetas Peterson permutadas dentro de sujeto.
- Backbone aleatorio congelado, igualando arquitectura y cabeza.
- Comparar linear probe, última etapa más cabeza, full fine-tune y L2-SP hacia los pesos fuente.
- Criterio: la fuente real debe superar a los controles en ambos protocolos y mostrar una
  relación dosis-respuesta al descongelar capas.

### H2: la transferencia reduce la cantidad de Souza necesaria

- Curvas con 10%, 25%, 50% y 100% del outer-train, estratificadas por run y clase.
- Misma partición para todos los brazos.
- Criterio: alcanzar una accuracy o kappa dada con menos datos, no solo mejorar la media final.

### H3: la información transferible está en el sistema sensorimotor

- Mu, beta y mu+beta.
- C3/Cz/C4 frente a los 15 canales comunes.
- Medir cambio de accuracy, kappa y calibración junto con ERD/ERS.
- Control: ninguna selección de banda o canal puede mirar el outer test.

### H4: el fallo cross-session es principalmente desplazamiento de dominio

- Normalización por run aprendida solo en outer-train.
- Adaptive BatchNorm sin etiquetas del test y CORAL ajustado solo con datos permitidos.
- Comparar con full fine-tune y scratch en los mismos folds.

### H5: la arquitectura determina qué se transfiere

- Representaciones congeladas de los cinco modelos evaluadas con una cabeza lineal común.
- Similaridad de representaciones entre Peterson y Souza y separabilidad de MI/rest en el
  sujeto Peterson retenido.
- Prestar especial atención a EEGNet, EEGConformer y EEGInceptionMI, porque v8 muestra
  comportamientos opuestos.

La cantidad de GPU no resuelve el límite biológico de cinco sujetos Souza. Más celdas sirven
para aislar mecanismos y estabilidad; no convierten seeds o ventanas en sujetos adicionales.
