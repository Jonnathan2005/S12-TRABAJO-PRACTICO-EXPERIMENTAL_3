"""
Paquete de visión por computadora de VisionAsistencia.

Es independiente de Django (solo usa OpenCV y NumPy), por lo que puede
reutilizarse en otros proyectos o probarse de forma aislada:

- ``procesamiento``: preprocesamiento de imágenes (gris, CLAHE, nitidez, filtros).
- ``detector``: detección de rostros y ojos con clasificadores Haar.
- ``reconocedor``: reconocimiento facial con LBPH (Local Binary Patterns Histograms).
- ``dibujo``: utilidades para dibujar resultados sobre el video.
- ``procesador``: tubería que une detección + reconocimiento + registro por cuadro.
- ``camara``: hilo de captura de video en tiempo real y emisión MJPEG.
"""
