# Integracion de Souza2023

## Alcance

El benchmark trata los dos datasets low-cost como experimentos independientes:

- `MI-OpenBCI`: imaginacion motora frente a reposo, 10 participantes y 15 canales.
- `Souza2023`: mano izquierda frente a mano derecha, 6 participantes, 16 canales y 4 corridas por participante.

Las metricas, familias estadisticas, figuras y tablas deben conservar las columnas `dataset` y
`task`. No se combinan observaciones de ambos datasets ni se interpreta su diferencia como efecto
del hardware.

## Epocado de Souza2023

Los eventos `LeftExec` y `RightExec` definen el inicio de cada ensayo. La etiqueta binaria es
izquierda `0` y derecha `1`. Cada `NewRun` inicia una sesion (`run_1` a `run_4`).

El articulo describe una fase de ejecucion de 4 s. Los EDF publicados sitúan `Resting` entre 2,97
y 3,16 s despues de `LeftExec` o `RightExec`. Para evitar incorporar reposo a la ventana de
imaginacion, el cargador extrae 3,0 s desde el evento de ejecucion. Los datos se remuestrean de
125 Hz a 128 Hz. El filtrado 1-40 Hz y 8-30 Hz, ICA opcional y normalizacion se mantienen dentro
del pipeline existente y se ajustan solo con la particion de entrenamiento.

Cada archivo completo debe contener 160 ensayos, 80 por clase, distribuidos en cuatro corridas de
40 ensayos balanceados. El cargador falla si cambia cualquiera de estos conteos.

## Protocolos

`Souza2023` ejecuta:

- `within_split`: division estratificada 70/30 dentro de cada corrida.
- `within_session`: validacion cruzada estratificada de cinco folds dentro de cada corrida.
- `cross_session`: leave-one-run-out.
- `loso`: leave-one-subject-out.
- Las mismas condiciones de ventanas y la misma sensibilidad ICA del perfil de `MI-OpenBCI`.

La condicion `overlap` conserva seis ventanas de 2 s y las distribuye entre el inicio y el final
de la epoca de 3 s. `center_x6` conserva seis replicas como control de computo. `nonoverlap` usa dos
ventanas y se compara con `center_x2`.

Los cinco modelos, cinco semillas y 300 epocas permanecen congelados para ambos datasets.

## Integridad de las descargas

El sitio del articulo ofrece seis enlaces, pero los adjuntos `42` y `45` son copias byte a byte del
sujeto `004` y tienen la misma huella SHA-256. El enlace `42` no puede usarse como sujeto `001`.

`setup.sh` conserva los enlaces suministrados, descarga los archivos y ejecuta una validacion de
identidad, estructura y duplicados. La corrida confirmatoria exige `001.edf` a `006.edf` unicos.
Mientras el editor no corrija el adjunto, pueden ejecutarse pilotos con `002` a `006`; el modo
completo termina con error. La variable `SOUZA_001_URL` permite proporcionar una URL corregida sin
editar el codigo.

## Operacion en Cayetano

`setup.sh` crea o actualiza el entorno, obtiene MI-OpenBCI, descarga Souza2023 y los datasets
MOABB, y ejecuta el preflight. `run_cayetano.sh` lanza los dos datasets low-cost por separado en
las dos GPU, luego completa validaciones externas, sensibilidad ICA, estadistica, figuras y tablas.
Todas las etapas son reanudables.
