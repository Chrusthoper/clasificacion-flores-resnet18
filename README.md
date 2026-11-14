# Arquitectura ResNet18 para clasificación de flores

Se implementó ResNet18 con transferencia de aprendizaje para clasificar las 102 categorías de Oxford Flowers102. Se verificaron dimensiones y parámetros, se compararon seis configuraciones, se seleccionó E5_LAYER4 mediante validación y se evaluó el modelo final en prueba. El checkpoint distribuido corresponde a la ejecución registrada en GPU Tesla T4.

## Resultados registrados

| Medida | Resultado |
|---|---:|
| Validación | 91.9608% |
| Prueba Top-1 | 88.8600% |
| Prueba Top-3 | 94.9748% |
| Imágenes de prueba | 6,149 |
| Parámetros totales | 11,228,838 |
| Parámetros entrenables en E5 | 8,446,054 |

Se sustituyó la capa final por Linear(512, 102) y se habilitó el entrenamiento de layer4 y fc. E5 mejoró la exactitud de validación en 16.2745 puntos porcentuales respecto a la configuración base.

## Archivos

- `docs/Arquitectura_ResNet18.pdf` y `.docx`: arquitectura, metodología, resultados, análisis visual, hallazgos, reproducibilidad y referencias.
- `notebooks/Arquitectura_ResNet18_GPU_ejecutado.ipynb`: salidas GPU conservadas y respuestas interpretadas.
- `notebooks/Arquitectura_ResNet18_GPU.ipynb`: código sin salidas para reproducir el protocolo; las interpretaciones identifican la ejecución de referencia.
- `notebooks/Arquitectura_ResNet18_CPU_historico.ipynb`: registro CPU identificado como histórico; sus métricas son distintas de las del modelo final GPU.
- `modelo/modelo_entrenado.pth`: checkpoint final con state_dict y metadatos.
- `modelo/metadata.json`: arquitectura, clases, transformaciones, configuración y versiones registradas.
- `modelo/resultados_experimentos.json`: historiales de seis configuraciones. Las rutas `checkpoint_origen_colab` documentan el origen; `checkpoint_final_exportado` identifica el modelo final distribuido.
- `evidencias/`: curvas, matriz de confusión, tres errores, reporte por clase y resumen GPU.
- `docs/VERIFICACION_GPU.md`: identidad del modelo y comprobaciones registradas.
- `requirements_registrados.txt`: versiones de bibliotecas registradas y variantes por entorno.
- `MANIFIESTO_SHA256.json`: tamaños y hashes de todos los demás archivos.

## Reproducción

Abrir `notebooks/Arquitectura_ResNet18_GPU.ipynb` en Google Colab, activar GPU y ejecutar en orden. El código descarga Flowers102 y los pesos preentrenados, entrena las seis configuraciones, selecciona por validación y evalúa prueba al final. La salida se genera en la carpeta indicada por ROOT. El notebook ejecutado permite consultar resultados sin repetir el entrenamiento.

Para reconstruir el checkpoint exportado, crear `resnet18(weights=None)`, sustituir `fc` por `Linear(512,102)` y cargar `checkpoint['state_dict']` con coincidencia estricta de claves. Aplicar el preprocesamiento de `modelo/metadata.json`, usar `eval()` y desactivar gradientes durante inferencia. Las etiquetas Flor 1 a Flor 102 conservan el orden de clases; no son nombres botánicos.

## Interpretación y límites

Las métricas proceden de las salidas GPU conservadas; la comprobación sintética de carga no equivale a recalcularlas sobre el dataset. Se conocían resultados históricos de prueba, aunque la selección dentro de la ejecución GPU utilizó validación. No se afirma igualdad bit a bit entre entrenamientos o dispositivos. Las probabilidades softmax no prueban calibración. El análisis y los hallazgos están desarrollados en el informe y los notebooks.

## Referencias

- https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- https://arxiv.org/abs/1512.03385
- https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
