# Infraestructura de entrenamiento y experimentos

Autor de esta parte: **Emiliano Villalobos Luevano** (rama `Emiliano_VL`, commit `791e22b`,
"Preparar entorno de entrenamiento y experimentación"). Esta nota documenta lo que entregó y
lo que se verificó al integrarlo; no reescribe su trabajo.

## Qué entregó

1. `05_flores_experimentos.ipynb`: copia de la plantilla con el entorno de entrenamiento
   preparado. Tiene 26 celdas y solo dos ejecutadas (importaciones y carga de datos); las
   celdas de respuesta siguen con el texto "Respuestas del equipo: escriban aquí".
2. Cinco checkpoints, ahora en `modelo/experimentos/`, cada uno con pesos, configuración,
   arquitectura, número de clases y la lista de las 102 etiquetas:

   | Archivo | Experimento | Semilla | Épocas | Tasa | Lote | Fine-tuning | Entrenables |
   |---|---|---:|---:|---:|---:|---|---:|
   | `BASE.pth` | BASE | 42 | 5 | 0.001 | 32 | No | 52,326 |
   | `EXP-01.pth` | EXP-01 | 42 | 10 | 0.001 | 32 | No | 52,326 |
   | `EXP-02.pth` | EXP-02 | 42 | 5 | 0.0005 | 32 | No | 52,326 |
   | `modelo_final.pth` | EXP-03 | 42 | 5 | 0.001 | 32 | Sí (`layer4`) | 8,446,054 |
   | `EXP-04.pth` | EXP-04 | 7 | 5 | 0.001 | 32 | No | 52,326 |

## Sus resultados, tal como quedaron registrados

Cada checkpoint guarda su propia configuración y su mejor época. Estos son los valores del
archivo, sin reinterpretarlos:

| Experimento | Mejor época | Exactitud de validación | Pérdida de validación | Tiempo (s) |
|---|---:|---:|---:|---:|
| BASE | 5 | 0.7569 | 1.7359 | 61.76 |
| EXP-01 | 10 | 0.8343 | 1.1091 | 114.46 |
| EXP-02 | 5 | 0.6853 | 2.5648 | 60.94 |
| EXP-03 | 5 | 0.9216 | 0.4620 | 65.49 |
| EXP-04 | 5 | 0.7637 | 1.7016 | 56.07 |

Los cinco registros declaran `dispositivo: mps`, es decir, GPU de Apple Silicon. Es un tercer
entorno, distinto de la Tesla T4 de la rama `Arquitectura` y de la CPU de la réplica local, así
que sus números no se promedian con los otros dos.

Lo que sostienen sus datos, en una lectura directa: duplicar las épocas sube la validación de
0.7569 a 0.8343; bajar la tasa a 0.0005 la hunde a 0.6853; cambiar la semilla mueve 0.68 puntos
(0.7569 a 0.7637); y descongelar `layer4` con las mismas 5 épocas la lleva a 0.9216. Ese último
es el modelo que él marcó como final, y coincide en dirección y magnitud con la configuración
E5 de la rama `Arquitectura`.

## Verificación independiente de los pesos

No basta con leer los metadatos: comparé tensor por tensor contra `BASE.pth`.

| Archivo | Tensores distintos de BASE | Bloques que cambian |
|---|---:|---|
| `EXP-01.pth` | 2 de 122 | `fc` |
| `EXP-02.pth` | 2 de 122 | `fc` |
| `EXP-04.pth` | 2 de 122 | `fc` |
| `modelo_final.pth` | 32 de 122 | `fc` y `layer4` |

Esto confirma lo que dice la configuración: en EXP-01, EXP-02 y EXP-04 el cuerpo de la red
quedó congelado y solo se movió el clasificador; en EXP-03 sí se ajustó `layer4`, con
diferencias de hasta 0.82 en las medias móviles de BatchNorm y 0.042 en las convoluciones.
La comparación entre experimentos también cuadra: los tres congelados comparten 120 de 122
tensores entre sí y solo 90 con EXP-03.

## Cómo se cargan

```python
import torch
estado = torch.load('modelo/experimentos/modelo_final.pth', map_location='cpu', weights_only=True)
print(estado['configuracion'])
pesos = estado['model_state_dict']  # 122 tensores, fc de (102, 512)
```

El archivo no es un `state_dict` suelto: es un diccionario con `model_state_dict`,
`configuracion`, `arquitectura` (`resnet18`), `num_clases` y `clases`. Para reutilizarlo hay que
reconstruir ResNet18, sustituir `fc` por `Linear(512, 102)` y respetar el orden de `clases`.

## Nota sobre el formato de los archivos

Los checkpoints llegaron como carpetas extraídas (`BASE.pth/BASE/data/...`), que es el
contenido interno del zip de PyTorch. Los volví a empaquetar como `.pth` para que
`torch.load` funcione y para versionarlos con Git LFS. Los tensores y los metadatos son
idénticos a los de las carpetas; el contenedor zip no. Huellas de los archivos publicados
(SHA-256, primeros 12 caracteres):

| Archivo | SHA-256 |
|---|---|
| `BASE.pth` | `a40af982cf0d` |
| `EXP-01.pth` | `07690afd5edb` |
| `EXP-02.pth` | `5dcfaf433930` |
| `EXP-04.pth` | `e619b1ab8427` |
| `modelo_final.pth` | `29ecef4d1439` |

## Lo que queda pendiente de su parte

- Las respuestas del notebook `05_flores_experimentos.ipynb` siguen vacías: sus conclusiones
  están en los metadatos de los checkpoints, no en el texto del notebook. Completarlas es
  trabajo suyo, no de quien integra.
- El notebook quedó con dos celdas ejecutadas de veintiséis; para la entrega conviene
  reiniciar el entorno y ejecutarlo completo, o dejar constancia de que solo se preparó el
  entorno.
