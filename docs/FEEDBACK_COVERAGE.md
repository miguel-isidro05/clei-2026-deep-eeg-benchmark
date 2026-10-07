# Cobertura de la retroalimentación: Peterson V10

Este archivo resume el perfil confirmatorio vigente. Los diseños Souza/transferencia conservados
en `docs/superpowers/` son históricos y no redefinen esta matriz.

| Observación | Solución V10 | Estado tras la corrida completa |
|---|---|---|
| Alcance de modelos | CSP+LDA, EEGNet, FBCNet, ShallowConvNet y EEGConformer bajo la misma matriz | Cerrado |
| Una sola semilla | Cinco semillas deep; CSP determinista identificado y agregado por participante | Cerrado |
| Inferencia y multiplicidad | Wilcoxon bilateral pareado, Holm, IC Student-t y rank-biserial | Cerrado |
| Sliding windows puede inflar el n | Una predicción por trial; sujeto como unidad; controles `center_x2` y `center_x6` emparejados en cómputo | Cerrado |
| Falta generalización | `within_split`, `within_session` y LOSO separados | Cerrado dentro de Peterson; no demuestra generalización externa |
| Selección ICA ambigua | Sin ICA como principal; FastICA train-only con regla congelada solo como sensibilidad | Cerrado con claim limitado |
| Convergencia no demostrada | 300 épocas fijas y curvas de training exportadas; no se afirma convergencia por validación | Cerrado por diseño, con limitación explícita |
| Latencia poco clara | Forward pass deep con input y dispositivo documentados; se excluye latencia BCI end-to-end | Cerrado con claim limitado |
| Reproducibilidad | Manifest exacto, predicciones, recetas, hashes, auditor de 3,250 celdas y exportación verificada | Cerrado |
| Cohorte pequeña / un dataset | Diez participantes Peterson | Limitación inherente |
| Comparación causal del hardware | No hay brazo research-grade controlado | No resuelto por diseño; no hacer ese claim |
| Fuentes de métodos y estadística | Auditor bibliográfico con DOI/identificadores para dataset, arquitecturas, CSP, Wilcoxon, Holm y kappa | Cerrado en documentación; pendiente insertar las citas al redactar el manuscrito |

La pregunta respaldada es: bajo un protocolo congelado dentro de Peterson MI-vs-rest, ¿cómo se
comparan CSP+LDA y cuatro decoders deep, y cuánto rendimiento aporta la augmentación temporal?
