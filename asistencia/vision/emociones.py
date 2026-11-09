"""
Detección de emociones mediante expresiones faciales.

Se usa la red FER+ (Microsoft, formato ONNX) ejecutada con el módulo
``cv2.dnn`` de OpenCV: recibe el rostro en gris a 64x64 y devuelve una
puntuación para 8 expresiones. No requiere TensorFlow ni PyTorch.

Si el archivo del modelo no está disponible (sin internet en la primera
ejecución), se usa un *modo básico* con el clasificador Haar de sonrisas,
que solo distingue «Feliz» de «Neutral».
"""

import logging
import os
import tempfile
import threading
import urllib.request

import cv2
import numpy as np

from .procesamiento import a_gris

log = logging.getLogger(__name__)

URL_MODELO = ('https://github.com/onnx/models/raw/main/validated/vision/body_analysis/'
              'emotion_ferplus/model/emotion-ferplus-8.onnx')
TAMANO_ENTRADA = (64, 64)

# Orden de salida de la red FER+.
EMOCIONES = ('neutral', 'feliz', 'sorpresa', 'tristeza', 'enojo', 'asco', 'miedo', 'desprecio')
NOMBRES = {
    'neutral': 'Neutral', 'feliz': 'Feliz', 'sorpresa': 'Sorpresa', 'tristeza': 'Tristeza',
    'enojo': 'Enojo', 'asco': 'Asco', 'miedo': 'Miedo', 'desprecio': 'Desprecio',
}
# Colores BGR para dibujar cada emoción sobre el video.
COLORES = {
    'neutral': (180, 180, 180), 'feliz': (21, 204, 250), 'sorpresa': (235, 160, 59),
    'tristeza': (200, 110, 40), 'enojo': (68, 68, 239), 'asco': (60, 160, 60),
    'miedo': (200, 90, 170), 'desprecio': (90, 120, 200),
}


def softmax(valores):
    v = np.asarray(valores, dtype=np.float64).ravel()
    e = np.exp(v - v.max())
    return e / e.sum()


def preparar_rostro(imagen, rect, margen=0.1):
    """Recorta el rostro con un pequeño margen, lo pasa a gris y lo lleva a 64x64."""
    x, y, w, h = rect
    alto, ancho = imagen.shape[:2]
    dx, dy = int(w * margen), int(h * margen)
    x0, y0 = max(0, x - dx), max(0, y - dy)
    x1, y1 = min(ancho, x + w + dx), min(alto, y + h + dy)
    recorte = imagen[y0:y1, x0:x1]
    if recorte.size == 0:
        return None
    return cv2.resize(a_gris(recorte), TAMANO_ENTRADA, interpolation=cv2.INTER_AREA)


class ClasificadorEmociones:
    """Clasifica la expresión facial de un rostro. Seguro para usar desde el hilo de video."""

    def __init__(self, ruta_modelo=None, descargar=True):
        self.ruta_modelo = str(ruta_modelo) if ruta_modelo else None
        self.descargar = descargar
        self._red = None
        self._sonrisas = None
        self._lock = threading.Lock()
        self._cargado = False

    # ----------------------------------------------------------- modelo
    @property
    def modo(self):
        self._asegurar_cargado()
        return 'FER+ (red neuronal ONNX)' if self._red is not None else 'basico (detector de sonrisa)'

    @property
    def completo(self):
        self._asegurar_cargado()
        return self._red is not None

    def descargar_modelo(self):
        """Descarga el modelo FER+ (~34 MB) a ``ruta_modelo``. Devuelve True si quedó disponible."""
        if not self.ruta_modelo:
            return False
        if os.path.exists(self.ruta_modelo):
            return True
        carpeta = os.path.dirname(self.ruta_modelo)
        os.makedirs(carpeta, exist_ok=True)
        log.info('Descargando modelo de emociones desde %s', URL_MODELO)
        fd, temporal = tempfile.mkstemp(dir=carpeta, suffix='.part')
        os.close(fd)
        try:
            urllib.request.urlretrieve(URL_MODELO, temporal)
            os.replace(temporal, self.ruta_modelo)
            return True
        finally:
            if os.path.exists(temporal):
                os.remove(temporal)

    def _asegurar_cargado(self):
        if self._cargado:
            return
        with self._lock:
            if self._cargado:
                return
            try:
                if self.ruta_modelo and not os.path.exists(self.ruta_modelo) and self.descargar:
                    self.descargar_modelo()
                if self.ruta_modelo and os.path.exists(self.ruta_modelo):
                    self._red = cv2.dnn.readNetFromONNX(self.ruta_modelo)
            except Exception:
                log.exception('No se pudo cargar el modelo FER+; se usará el modo básico.')
                self._red = None
            if self._red is None:
                ruta = os.path.join(cv2.data.haarcascades, 'haarcascade_smile.xml')
                self._sonrisas = cv2.CascadeClassifier(ruta)
            self._cargado = True

    # ------------------------------------------------------- clasificar
    def clasificar(self, imagen, rect):
        """
        Devuelve un vector de probabilidades (suma 1) en el orden de ``EMOCIONES``
        para el rostro ``rect`` de ``imagen``, o ``None`` si el recorte no es válido.
        """
        self._asegurar_cargado()
        cara = preparar_rostro(imagen, rect)
        if cara is None:
            return None
        if self._red is not None:
            blob = cv2.dnn.blobFromImage(cara, 1.0, TAMANO_ENTRADA)
            with self._lock:  # cv2.dnn.Net no es seguro entre hilos
                self._red.setInput(blob)
                salida = self._red.forward()
            return softmax(salida)
        return self._clasificar_basico(cara)

    def _clasificar_basico(self, cara):
        """Modo de respaldo: sonrisa en la mitad inferior del rostro -> Feliz; si no, Neutral."""
        grande = cv2.resize(cara, (160, 160))
        boca = cv2.equalizeHist(grande[90:, :])
        sonrisas = self._sonrisas.detectMultiScale(boca, scaleFactor=1.6, minNeighbors=18, minSize=(40, 15))
        probs = np.full(len(EMOCIONES), 0.02)
        probs[EMOCIONES.index('feliz' if len(sonrisas) else 'neutral')] = 0.86
        return probs / probs.sum()


class SuavizadorEmociones:
    """
    Promedia las probabilidades de cada rostro entre cuadros (media móvil
    exponencial) para que la etiqueta no parpadee. Los rostros se asocian
    entre cuadros por la cercanía de sus centros.
    """

    def __init__(self, alfa=0.35, max_perdidos=5):
        self.alfa = alfa
        self.max_perdidos = max_perdidos
        self._pistas = []  # [{'centro': (cx, cy), 'tam': w, 'probs': array, 'perdidos': n}]

    def reiniciar(self):
        self._pistas = []

    def actualizar(self, rects, lista_probs):
        """Recibe los rostros del cuadro y sus probabilidades; devuelve las suavizadas."""
        usadas = set()
        resultado = []
        for (x, y, w, h), probs in zip(rects, lista_probs):
            if probs is None:
                resultado.append(None)
                continue
            centro = (x + w / 2, y + h / 2)
            mejor, mejor_dist = None, None
            for i, pista in enumerate(self._pistas):
                if i in usadas:
                    continue
                dist = np.hypot(centro[0] - pista['centro'][0], centro[1] - pista['centro'][1])
                if dist < 0.6 * max(w, pista['tam']) and (mejor_dist is None or dist < mejor_dist):
                    mejor, mejor_dist = i, dist
            if mejor is None:
                self._pistas.append({'centro': centro, 'tam': w, 'probs': probs, 'perdidos': 0})
                usadas.add(len(self._pistas) - 1)
                resultado.append(probs)
            else:
                pista = self._pistas[mejor]
                pista['probs'] = self.alfa * probs + (1 - self.alfa) * pista['probs']
                pista.update(centro=centro, tam=w, perdidos=0)
                usadas.add(mejor)
                resultado.append(pista['probs'])

        for i, pista in enumerate(self._pistas):
            if i not in usadas:
                pista['perdidos'] += 1
        self._pistas = [p for p in self._pistas if p['perdidos'] <= self.max_perdidos]
        return resultado
