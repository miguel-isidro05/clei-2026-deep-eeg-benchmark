# Inventario reproducible de datasets Motor Imagery en MOABB

Fecha de consulta: 2026-09-27. Entorno fijado: MOABB 1.5.0.

## Utilidad usada

MOABB incluye `moabb.datasets.utils.dataset_search`, que filtra por paradigma, eventos, número de
sujetos, canales y presencia de varias sesiones. El filtro exacto fue:

```python
from moabb.datasets.utils import dataset_search

candidates = dataset_search(
    paradigm="imagery",
    multi_session=True,
    events=["right_hand", "rest"],
    has_all_events=True,
)
```

La instalación fijada contiene 54 datasets con `paradigm="imagery"`, 25 multisesión y solo tres
que cumplen a la vez `right_hand`, `rest` y más de una sesión.

| Dataset | Sujetos | Sesiones | Canales | Duración | Ventaja | Decisión |
|---|---:|---:|---:|---:|---|---|
| Zhou2020 | 20 | 7 | 26 o 41 | 5 s | Más sujetos, longitudinal y task-matched | Incluir |
| Tavakolan2017 | 12 | 4 | 32 | 3 s | Task-matched, CC0 y costo manejable | Incluir |
| Stieger2021 | 62 | hasta 11 | 64 | 3 s | Cohorte longitudinal grande | Excluir del perfil: costo y licencia no comercial |

Zhou2020 registra left hand, right hand, feet y rest. Tavakolan2017 registra rest, right hand y
right elbow flexion. El paradigma selecciona solo `right_hand` y `rest`, sin cambiar las etiquetas
restantes. Stieger2021 tiene cerca de 250.000 trials; cinco modelos, cinco seeds y leave-one-session-
out multiplicarían demasiado el costo del paper actual.

BNCI2014_001 no aparece en el filtro porque sus clases son left hand, right hand, feet y tongue, sin
rest. Es adecuado para un benchmark BCI IV 2a de cuatro clases, pero no es task-matched con el
contraste MI-versus-rest del paper.

Fuentes:

- https://moabb.neurotechx.com/docs/generated/moabb.datasets.utils.dataset_search.html
- https://moabb.neurotechx.com/docs/dataset_summary.html
- https://moabb.neurotechx.com/docs/generated/moabb.datasets.Tavakolan2017.html
- https://moabb.neurotechx.com/docs/generated/moabb.datasets.Stieger2021.html
- https://github.com/NeuroTechX/moabb/blob/develop/moabb/datasets/zhou2020.py
