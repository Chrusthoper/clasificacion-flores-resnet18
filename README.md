# Evaluación y demostración (Persona 4)

Rama de la parte de evaluación del proyecto de clasificación de flores con ResNet18. El
notebook `05_flores.ipynb` está ejecutado y trae, además de las ocho secciones respondidas,
una sección 9 con la ruta completa de evaluación y demo: matriz de confusión, métricas por
clase, tres errores comentados, guardado con metadatos, recarga estricta e inferencia Top-3.

## Dependencia con los experimentos

La evaluación final se calcula con la configuración que gane en validación, así que esta
parte depende de Persona 3. Para no quedarse esperando, la sección 9 tiene un interruptor:

```python
CONFIG = 'base'   # 'base' mientras se espera; 'final' cuando se decida
```

Con `'base'` entrena solo la capa `fc` durante 5 épocas y deja funcionando toda la tubería
de evaluación. Con `'final'` descongela `layer4` (8,446,054 parámetros entrenables frente a
52,326) y usa 10 épocas, que es la configuración que ganó en la comparación de validación.
El código de las celdas no cambia entre ambos casos: cambia el punto de partida del modelo.

## Cómo ejecutarlo

1. Abrir `05_flores.ipynb` en Colab con GPU, o en local con `torch`, `torchvision`,
   `scikit-learn` y `matplotlib`.
2. Ejecutar todo con `CONFIG = 'base'`. La primera vez descarga Oxford Flowers 102 (unos
   345 MB) y los pesos de ResNet18.
3. Cuando Persona 3 cierre los experimentos, cambiar `CONFIG` a `'final'` y volver a correr
   solo la sección 9. Los números, la matriz y las capturas se actualizan solos.
4. Si ya existe `evaluacion_<config>.pth`, la sección 9 carga el checkpoint y no reentrena.

## Resultados de la corrida base (registrada)

Entorno: CPU, PyTorch 2.14.0+cu130, torchvision 0.29.0, scikit-learn 1.9.1. Sin GPU
disponible en la máquina donde se ejecutó.

| Medida | Resultado |
|---|---:|
| Exactitud de validación (época 5) | 78.33 % |
| Prueba Top-1 | 75.49 % (4,642 de 6,149) |
| Prueba Top-3 | 87.67 % |
| Pérdida en prueba | 1.5577 |
| Parámetros entrenables | 52,326 de 11,228,838 |
| Tiempo de entrenamiento | ~16 min (5 épocas, CPU) |

Confusiones más frecuentes: Flor 74 → Flor 96 (28 casos), Flor 51 → Flor 76 (26) y
Flor 51 → Flor 72 (22). La clase 51 tiene 238 imágenes de prueba y su recall es de 0.46,
así que concentra buena parte del error.

La recarga del modelo guardado reproduce exactamente las predicciones del modelo en memoria
(`Predicciones idénticas tras recargar: True`), y las cinco imágenes de la demo aparecen con
su Top-3 y su confianza. La confianza de esas cinco va de 0.17 a 0.91, lo que ya muestra que
el modelo no está calibrado de forma uniforme.

## Archivos que genera

- `evaluacion_base.pth` / `evaluacion_final.pth`: pesos, historial y configuración.
- `modelo_demo_base.pth` / `modelo_demo_final.pth`: pesos con clases, transformación y métricas.
- `modelo_entrenado.pth`: lo guarda la sección 7 de la plantilla.

Los `.pth` están en `.gitignore` porque pesan unos 45 MB cada uno; se regeneran al ejecutar.

## Qué queda pendiente

- Cambiar `CONFIG` a `'final'` y volver a correr la sección 9 cuando Persona 3 elija.
- Rellenar la segunda fila de la tabla de la sección 10 con esas cifras.
- Repetir la demo con imágenes propias si el profesor pide una entrada externa al dataset.

## Límites que conviene decir en voz alta

El modelo solo elige entre las 102 clases del dataset; una flor de otra especie recibirá
alguna de esas etiquetas, a veces con confianza alta. Las probabilidades de softmax no
acreditan calibración y Top-3 alto no significa que el modelo "dude" donde debería. La
evaluación con 6,149 imágenes es la que sostiene las conclusiones; cinco predicciones de
demo no sustituyen esa medición.
