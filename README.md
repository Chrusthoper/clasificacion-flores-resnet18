# Clasificación de flores con ResNet18

Notebook didáctico en español para clasificar Oxford Flowers 102 mediante transferencia de aprendizaje con ResNet18 preentrenada de torchvision.

## Contenido

- `05_flores.ipynb`: flujo guiado de preparación de datos, ajuste de la capa clasificadora, evaluación, matriz de confusión y predicción Top-3.

## Requisitos

Python 3.10+ y Jupyter. Dependencias: PyTorch, torchvision, NumPy, Matplotlib, scikit-learn y pandas (si se amplía el análisis). La primera ejecución descarga Oxford Flowers 102 y los pesos preentrenados; se requiere conexión a internet y espacio en disco.

## Uso

Abre `05_flores.ipynb` en Jupyter o Google Colab y ejecuta las celdas en orden. El notebook es una plantilla de actividad: las preguntas y espacios de respuesta están pendientes de completar. Los tiempos y métricas dependen del hardware, las versiones y la ejecución.

El conjunto de prueba debe reservarse para la evaluación final; no se debe usar para elegir hiperparámetros.
