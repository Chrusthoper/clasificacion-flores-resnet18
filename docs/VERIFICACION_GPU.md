# Identidad y verificación del modelo final

El paquete contiene `modelo/modelo_entrenado.pth`, con 44,998,667 bytes y SHA-256 `92d799f5448ac120e73d2a03f49af9176a49951710264724a1b516e29ccf137c`. Los metadatos internos identifican CUDA, E5_LAYER4, ResNet18 y 102 clases, coherentes con `modelo/metadata.json` y el registro GPU.

## Comprobaciones registradas

El notebook GPU conserva la reconstrucción de ResNet18, carga estricta del state_dict y comparación de la predicción antes y después de exportar: las salidas coinciden para el ejemplo comprobado. La verificación externa documentada registró carga con weights_only=True, coincidencia estricta de claves e inferencia sintética de forma (1,3,224,224) a (1,102), con valores finitos.

La auditoría de arquitectura registra 11,228,838 parámetros totales; E5 habilita 8,446,054 y congela 2,782,784. La selección se realizó con validación (91.9608%) y la prueba registró Top-1 de 88.8600% y Top-3 de 94.9748% sobre 6,149 imágenes.

## Alcance de las comprobaciones

La verificación de integridad y la lectura estática de metadatos identifican el archivo distribuido; no recalculan sus métricas. La comprobación sintética acredita carga y dimensiones, no exactitud sobre Flowers102. Las métricas proceden de las salidas GPU conservadas. La celda auxiliar de empaquetado del notebook ejecutado no registra ejecución; esto no altera las salidas conservadas de entrenamiento, evaluación y exportación.
