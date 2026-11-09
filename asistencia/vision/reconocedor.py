"""Reconocimiento facial con LBPH (Local Binary Patterns Histograms) de OpenCV."""

import os
import threading
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Prediccion:
    etiqueta: int | None   # id de la persona o None si es desconocida
    distancia: float       # distancia LBPH (menor = más parecido)
    confianza: float       # porcentaje 0-100 derivado de la distancia

    @property
    def reconocido(self):
        return self.etiqueta is not None


def distancia_a_confianza(distancia, umbral):
    """Convierte la distancia LBPH en un porcentaje intuitivo (100 = idéntico)."""
    limite = umbral * 1.6
    return round(max(0.0, min(100.0, 100.0 * (1 - distancia / limite))), 1)


class ReconocedorFacial:
    """
    Envoltura segura para hilos sobre ``cv2.face.LBPHFaceRecognizer``.

    El entrenamiento crea un modelo nuevo y lo intercambia de forma atómica,
    así el hilo de video puede seguir prediciendo mientras se reentrena.
    """

    def __init__(self, umbral=70.0, ruta_modelo=None):
        self.umbral = umbral
        self.ruta_modelo = str(ruta_modelo) if ruta_modelo else None
        self._modelo = None
        self._lock = threading.Lock()

    @staticmethod
    def _nuevo_modelo():
        if not hasattr(cv2, 'face'):
            raise RuntimeError('cv2.face no disponible: instale opencv-contrib-python.')
        return cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)

    @property
    def entrenado(self):
        return self._modelo is not None

    def entrenar(self, rostros, etiquetas):
        """Entrena con imágenes en gris del mismo tamaño y sus etiquetas (ids enteros)."""
        if len(rostros) == 0:
            with self._lock:
                self._modelo = None
            return
        if len(rostros) != len(etiquetas):
            raise ValueError('La cantidad de rostros y etiquetas debe coincidir.')
        modelo = self._nuevo_modelo()
        modelo.train(list(rostros), np.array(etiquetas, dtype=np.int32))
        with self._lock:
            self._modelo = modelo

    def predecir(self, rostro_gris):
        """Devuelve una :class:`Prediccion`. Si no hay modelo, la persona es desconocida."""
        with self._lock:
            modelo = self._modelo
        if modelo is None:
            return Prediccion(None, float('inf'), 0.0)
        etiqueta, distancia = modelo.predict(rostro_gris)
        confianza = distancia_a_confianza(distancia, self.umbral)
        if distancia > self.umbral:
            return Prediccion(None, float(distancia), confianza)
        return Prediccion(int(etiqueta), float(distancia), confianza)

    def guardar(self):
        if not self.ruta_modelo or self._modelo is None:
            return False
        os.makedirs(os.path.dirname(self.ruta_modelo), exist_ok=True)
        with self._lock:
            self._modelo.write(self.ruta_modelo)
        return True

    def cargar(self):
        if not self.ruta_modelo or not os.path.exists(self.ruta_modelo):
            return False
        modelo = self._nuevo_modelo()
        modelo.read(self.ruta_modelo)
        with self._lock:
            self._modelo = modelo
        return True

    def olvidar(self):
        with self._lock:
            self._modelo = None
        if self.ruta_modelo and os.path.exists(self.ruta_modelo):
            os.remove(self.ruta_modelo)
