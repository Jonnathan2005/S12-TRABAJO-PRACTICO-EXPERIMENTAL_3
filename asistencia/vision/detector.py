"""Detección de rostros y ojos con clasificadores en cascada de Haar (OpenCV)."""

import os
from dataclasses import dataclass, field

import cv2

from .procesamiento import a_gris, ecualizar


def _ruta_cascada(nombre):
    ruta = os.path.join(cv2.data.haarcascades, nombre)
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f'No se encontró {nombre}. Instale opencv-contrib-python 4.x (ver requirements.txt).'
        )
    return ruta


@dataclass
class Rostro:
    """Rostro detectado en un cuadro."""

    x: int
    y: int
    w: int
    h: int
    ojos: list = field(default_factory=list)  # [(x, y, w, h)] en coordenadas absolutas

    @property
    def rect(self):
        return (self.x, self.y, self.w, self.h)

    @property
    def area(self):
        return self.w * self.h


class DetectorRostros:
    """
    Detecta rostros frontales y, opcionalmente, ojos dentro de cada rostro.

    Para ganar velocidad, la detección se hace sobre una copia reducida del
    cuadro (``ancho_proceso``) y luego las coordenadas se escalan al tamaño real.
    """

    def __init__(self, ancho_proceso=480, escala=1.1, vecinos=5, tam_minimo=60, detectar_ojos=True):
        self.cascada_rostro = cv2.CascadeClassifier(_ruta_cascada('haarcascade_frontalface_default.xml'))
        self.cascada_ojos = cv2.CascadeClassifier(_ruta_cascada('haarcascade_eye_tree_eyeglasses.xml'))
        self.ancho_proceso = ancho_proceso
        self.escala = escala
        self.vecinos = vecinos
        self.tam_minimo = tam_minimo
        self.detectar_ojos = detectar_ojos

    def detectar(self, imagen):
        """Devuelve una lista de :class:`Rostro` ordenada de mayor a menor tamaño."""
        gris = ecualizar(a_gris(imagen))
        alto, ancho = gris.shape[:2]

        factor = 1.0
        if ancho > self.ancho_proceso:
            factor = self.ancho_proceso / ancho
            pequena = cv2.resize(gris, (self.ancho_proceso, int(alto * factor)), interpolation=cv2.INTER_AREA)
        else:
            pequena = gris

        minimo = max(20, int(self.tam_minimo * factor))
        rects = self.cascada_rostro.detectMultiScale(
            pequena,
            scaleFactor=self.escala,
            minNeighbors=self.vecinos,
            minSize=(minimo, minimo),
        )

        rostros = []
        for (x, y, w, h) in rects:
            x, y, w, h = (int(round(v / factor)) for v in (x, y, w, h))
            rostro = Rostro(x, y, w, h)
            if self.detectar_ojos:
                rostro.ojos = self._detectar_ojos(gris, rostro)
            rostros.append(rostro)

        rostros.sort(key=lambda r: r.area, reverse=True)
        return rostros

    def _detectar_ojos(self, gris, rostro):
        # Los ojos solo se buscan en la mitad superior del rostro.
        mitad = gris[rostro.y:rostro.y + rostro.h // 2, rostro.x:rostro.x + rostro.w]
        if mitad.size == 0:
            return []
        minimo = max(10, rostro.w // 8)
        ojos = self.cascada_ojos.detectMultiScale(mitad, scaleFactor=1.1, minNeighbors=6, minSize=(minimo, minimo))
        return [(rostro.x + ex, rostro.y + ey, ew, eh) for (ex, ey, ew, eh) in ojos[:2]]
