"""Funciones de procesamiento de imágenes con OpenCV."""

import cv2
import numpy as np

# Tamaño estándar al que se normalizan los rostros antes de entrenar/reconocer.
TAMANO_ROSTRO = (200, 200)


_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def a_gris(imagen):
    """Convierte una imagen BGR a escala de grises (si ya es gris la devuelve igual)."""
    if imagen.ndim == 2:
        return imagen
    return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)


def ecualizar(gris):
    """Mejora el contraste local con CLAHE; reduce el efecto de la iluminación desigual."""
    return _clahe.apply(gris)


def normalizar_rostro(imagen, rect):
    """
    Recorta el rostro indicado por ``rect`` (x, y, w, h), lo pasa a gris,
    lo ecualiza y lo redimensiona a ``TAMANO_ROSTRO``. Es el formato que usa LBPH.
    """
    x, y, w, h = rect
    recorte = imagen[y:y + h, x:x + w]
    gris = a_gris(recorte)
    gris = cv2.resize(gris, TAMANO_ROSTRO, interpolation=cv2.INTER_AREA)
    return ecualizar(gris)


def nitidez(gris):
    """Varianza del Laplaciano: valores bajos indican imagen borrosa."""
    return float(cv2.Laplacian(gris, cv2.CV_64F).var())


def brillo(gris):
    """Brillo medio (0-255) de la imagen."""
    return float(np.mean(gris))


def evaluar_iluminacion(valor_brillo):
    """Clasifica el brillo medio en una etiqueta legible."""
    if valor_brillo < 60:
        return 'baja'
    if valor_brillo > 200:
        return 'excesiva'
    return 'adecuada'

