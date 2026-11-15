# Clasificación de flores con ResNet18 — entrega final

Proyecto 5 de Deep Learning: adaptar ResNet18 preentrenada para clasificar las 102 clases de
Oxford Flowers 102 con transferencia de aprendizaje.

Esta rama reúne el trabajo de las cuatro ramas del equipo en un solo notebook,
`05_flores_final.ipynb`, ejecutado y con las ocho secciones respondidas más dos secciones
nuevas: la comparación de experimentos y la evaluación final.

## Qué aporta cada rama

| Rama | Persona | Aporte | Dónde queda |
|---|---|---|---|
| `Datos` | Laura (1) | Exploración del dataset, respuestas de las secciones 1 y 2, pruebas de semilla y tamaño de lote en CPU | `05_flores_datos.ipynb` |
| `Arquitectura` | Migue (2) | Auditoría de la red, seis configuraciones entrenadas en GPU, selección por validación, evaluación final, matriz, métricas por clase y checkpoint | `notebooks/`, `modelo/`, `evidencias/`, `docs/` |
| `Entrenamiento_y_experimentos` | Emi (3) | No tenía commits propios: apuntaba al mismo commit base que `main`. Su contenido (la comparación de experimentos) vive en la rama `Arquitectura` y se integra en la sección 9 del notebook final | sección 9 de `05_flores_final.ipynb` |
| `Evaluación_demostración` | Chris (4) | Ruta de evaluación y demo: matriz de confusión, errores comentados, guardado con metadatos, recarga estricta e inferencia Top-3 | sección 10 de `05_flores_final.ipynb` |

## Configuración final

La comparación se hizo solo con validación, y ganó `E5_LAYER4`: ResNet18 con `layer4` y `fc`
entrenables, lote 32, tasa de aprendizaje 0.001 y **5 épocas**. Deja 8,446,054 parámetros
entrenables de 11,228,838.

Conviene decirlo porque hubo una divergencia: una versión anterior del README de evaluación
describía la configuración final como `layer4` con 10 épocas. Los registros de
`modelo/resultados_experimentos.json` muestran que el ganador por validación es E5 con 5
épocas (91.9608 %); las 10 épocas corresponden a E4, que mantiene la red congelada y llega a
83.4314 %. El notebook final usa E5.

## Resultados

Los dos entornos se reportan por separado porque no son la misma corrida y no deben
promediarse.

**GPU (Tesla T4, Colab) — modelo final E5, evaluación única sobre prueba**

| Medida | Resultado |
|---|---:|
| Exactitud de validación | 91.9608 % |
| Prueba Top-1 | 88.8600 % (5,464 de 6,149) |
| Prueba Top-3 | 94.9748 % |
| Pérdida en prueba | 0.5492 |
| Parámetros entrenables | 8,446,054 de 11,228,838 |

**CPU (local, PyTorch 2.14.0) — corrida base con `fc` solamente, usada para validar la tubería**

| Medida | Resultado |
|---|---:|
| Exactitud de validación (época 5) | 78.33 % |
| Prueba Top-1 | 75.49 % (4,642 de 6,149) |
| Prueba Top-3 | 87.67 % |
| Pérdida en prueba | 1.5577 |
| Parámetros entrenables | 52,326 de 11,228,838 |

## Los seis experimentos

Todos con semilla 42 salvo donde se indica, y todos evaluados en validación antes de tocar
prueba.

| Id | Cambio | Val. exactitud | Val. pérdida | Segundos | Entrenables |
|---|---|---:|---:|---:|---:|
| B0 | Base: solo `fc`, 5 épocas | 75.6863 % | 1.7359 | 65.29 | 52,326 |
| E1 | Semilla 7 | 76.3725 % | 1.7016 | 62.91 | 52,326 |
| E2 | Lote 64 | 73.4314 % | 2.2156 | 60.63 | 52,326 |
| E3 | Tasa 0.0001 | 18.3333 % | 4.1334 | 63.62 | 52,326 |
| E4 | 10 épocas | 83.4314 % | 1.1090 | 128.24 | 52,326 |
| E5 | `layer4` descongelado | 91.9608 % | 0.4631 | 63.84 | 8,446,054 |

Regla de selección aplicada: mayor exactitud de validación, luego menor pérdida de validación,
luego menos parámetros entrenables.

## Cómo ejecutarlo

1. Abrir `05_flores_final.ipynb` en Colab con GPU, o en local con `torch`, `torchvision`,
   `scikit-learn` y `matplotlib`.
2. Ejecutar en orden. La primera vez descarga Oxford Flowers 102 (unos 345 MB) y los pesos
   de ResNet18.
3. La sección 9 lee `modelo/resultados_experimentos.json` si está presente; si no, muestra la
   tabla registrada en el propio notebook.
4. La sección 10 tiene un interruptor `CONFIG = 'base' | 'final'`. Con `'base'` entrena solo
   `fc` y sirve para probar la tubería rápido; con `'final'` reproduce E5.
5. Si existe el checkpoint de la configuración elegida, la sección 10 lo carga y no reentrena.

## Archivos

- `05_flores_final.ipynb`: notebook integrado y ejecutado (entrega principal).
- `05_flores_datos.ipynb`: evidencia de la rama `Datos`, con la exploración y las pruebas de
  semilla y lote en CPU.
- `notebooks/Arquitectura_ResNet18_GPU_ejecutado.ipynb`: salidas GPU conservadas.
- `notebooks/Arquitectura_ResNet18_GPU.ipynb`: código sin salidas para reproducir.
- `notebooks/Arquitectura_ResNet18_CPU_historico.ipynb`: registro CPU histórico, con métricas
  distintas de las del modelo final.
- `modelo/modelo_entrenado.pth`: checkpoint final con `state_dict` y metadatos.
- `modelo/metadata.json`: arquitectura, clases, transformación, configuración y versiones.
- `modelo/resultados_experimentos.json`: historiales de las seis configuraciones.
- `evidencias/`: curvas, matriz de confusión, tres errores, métricas por clase y resumen GPU.
- `docs/`: informe de arquitectura en PDF y DOCX, y verificación del entorno GPU.
- `requirements_registrados.txt`: versiones por entorno.

## Límites que conviene decir en voz alta

El modelo solo elige entre las 102 clases del dataset: una flor de otra especie recibirá
alguna de esas etiquetas, a veces con confianza alta. Las probabilidades de softmax no
acreditan calibración, y un Top-3 alto no significa que el modelo dude donde debería. La
evaluación con 6,149 imágenes es la que sostiene las conclusiones; cinco predicciones de demo
no sustituyen esa medición.

Las diferencias de semilla rondan 0.69 puntos en GPU y 1.57 en CPU, así que mejoras menores a
ese orden no deben presentarse como hallazgos.

## Referencias

- https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- https://arxiv.org/abs/1512.03385
- https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
