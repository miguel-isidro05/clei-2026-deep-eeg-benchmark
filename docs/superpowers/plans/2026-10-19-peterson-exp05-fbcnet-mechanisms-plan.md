# Plan de implementación Exp05

1. Definir catálogo y configuración congelada de 20 mecanismos.
2. Implementar modelos con una interfaz común, inicialización residual/identidad
   y hooks train-only para CSP, mu y calidad.
3. Implementar partición externa + validación interna, con hashes e invariantes
   de no solapamiento.
4. Implementar selección de época interna y refit desde cero sobre todo el train
   externo.
5. Implementar runner reanudable, sharding por sujeto y celdas auditables.
6. Implementar análisis screen/confirm, decisión firmada y promoción de máximo
   cuatro mecanismos.
7. Implementar auditor y launcher para dos GPU.
8. Añadir pruebas de catálogo, modelos, fuga, determinismo, auditoría y CLI.
9. Ejecutar smoke local, suite completa, lint y revisión independiente.
10. Commit, push y entregar comandos `tmux` para Cayetano.
