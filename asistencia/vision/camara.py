"""
Captura de video en tiempo real con ``cv2.VideoCapture`` en un hilo dedicado.

Un único hilo lee la cámara, procesa cada cuadro y publica el último JPEG.
Todos los clientes web consumen ese mismo JPEG (MJPEG), de modo que abrir
varias pestañas no multiplica el trabajo ni compite por la cámara.
"""

import logging
import sys
import threading
import time

import cv2
import numpy as np

from . import dibujo

log = logging.getLogger(__name__)


def interpretar_fuente(fuente):
    """'0' -> 0 (cámara local); cualquier otra cadena es ruta de archivo o URL."""
    fuente = str(fuente).strip()
    return int(fuente) if fuente.isdigit() else fuente


def abrir_captura(fuente, ancho, alto):
    fuente = interpretar_fuente(fuente)
    if isinstance(fuente, int) and sys.platform.startswith('win'):
        captura = cv2.VideoCapture(fuente, cv2.CAP_DSHOW)  # abre más rápido en Windows
        if not captura.isOpened():
            captura = cv2.VideoCapture(fuente)
    else:
        captura = cv2.VideoCapture(fuente)
    if captura.isOpened() and isinstance(fuente, int):
        captura.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
        captura.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
        captura.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return captura


def cuadro_mensaje(ancho, alto, titulo, detalle=''):
    """Imagen que se muestra cuando la cámara no está disponible."""
    img = np.full((alto, ancho, 3), (40, 30, 25), dtype=np.uint8)
    cv2.putText(img, dibujo.texto_ascii(titulo), (24, alto // 2 - 10), dibujo.FUENTE, 0.8, dibujo.BLANCO, 2, cv2.LINE_AA)
    if detalle:
        cv2.putText(img, dibujo.texto_ascii(detalle), (24, alto // 2 + 24), dibujo.FUENTE, 0.5, (180, 180, 180), 1,
                    cv2.LINE_AA)
    return img


class ServicioVideo:
    """Hilo de captura + procesamiento con apagado automático por inactividad."""

    def __init__(self, procesador, fuente=0, ancho=640, alto=480, calidad_jpeg=80, inactividad=30):
        self.procesador = procesador
        self.fuente = fuente
        self.ancho = ancho
        self.alto = alto
        self.calidad_jpeg = calidad_jpeg
        self.inactividad = inactividad

        self._hilo = None
        self._activo = threading.Event()
        self._cond = threading.Condition()
        self._jpeg = None
        self._id_cuadro = 0
        self._ultimo_acceso = time.monotonic()
        self._generacion = 0  # se incrementa al detener: invalida los streams abiertos
        self.camara_ok = False
        self.error = ''
        self.info = {}

    # --------------------------------------------------------------- control
    @property
    def en_ejecucion(self):
        return self._hilo is not None and self._hilo.is_alive()

    def tocar(self):
        """Registra que hay un espectador; arranca el hilo si estaba detenido."""
        self._ultimo_acceso = time.monotonic()
        if not self.en_ejecucion:
            self.iniciar()

    def iniciar(self):
        if self.en_ejecucion:
            return
        self._activo.set()
        self._hilo = threading.Thread(target=self._bucle, name='ServicioVideo', daemon=True)
        self._hilo.start()

    def detener(self):
        self._generacion += 1
        self._activo.clear()
        if self._hilo is not None:
            self._hilo.join(timeout=3)
        self._hilo = None
        self.camara_ok = False

    # ----------------------------------------------------------- consumidores
    def esperar_cuadro(self, ultimo_id=0, timeout=2.0, generacion=None):
        """
        Bloquea hasta que haya un cuadro más nuevo que ``ultimo_id``.
        Si ``generacion`` ya no es la vigente (se pulsó «Detener»), no reabre la cámara.
        """
        if generacion is not None and generacion != self._generacion:
            return ultimo_id, None
        self.tocar()
        with self._cond:
            self._cond.wait_for(lambda: self._id_cuadro != ultimo_id or not self._activo.is_set(), timeout)
            return self._id_cuadro, self._jpeg

    def generar_mjpeg(self):
        """Generador para ``StreamingHttpResponse`` (multipart/x-mixed-replace)."""
        ultimo = 0
        generacion = self._generacion
        while generacion == self._generacion:
            ultimo, jpeg = self.esperar_cuadro(ultimo, generacion=generacion)
            if generacion != self._generacion:
                break
            if jpeg is None:
                time.sleep(0.1)
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(jpeg)).encode() +
                   b'\r\n\r\n' + jpeg + b'\r\n')

    def estado(self):
        return {
            'en_ejecucion': self.en_ejecucion,
            'camara_ok': self.camara_ok,
            'error': self.error,
            'fuente': str(self.fuente),
            **self.info,
        }

    # ----------------------------------------------------------------- hilo
    def _publicar(self, imagen):
        ok, buffer = cv2.imencode('.jpg', imagen, [cv2.IMWRITE_JPEG_QUALITY, self.calidad_jpeg])
        if not ok:
            return
        with self._cond:
            self._jpeg = buffer.tobytes()
            self._id_cuadro += 1
            self._cond.notify_all()

    def _bucle(self):
        captura = None
        es_archivo = not isinstance(interpretar_fuente(self.fuente), int)
        try:
            while self._activo.is_set():
                if time.monotonic() - self._ultimo_acceso > self.inactividad:
                    log.info('Sin espectadores: se libera la cámara.')
                    break

                if captura is None or not captura.isOpened():
                    captura = abrir_captura(self.fuente, self.ancho, self.alto)
                    if not captura.isOpened():
                        self.camara_ok = False
                        self.error = f'No se pudo abrir la cámara/fuente "{self.fuente}".'
                        self._publicar(cuadro_mensaje(self.ancho, self.alto, 'Camara no disponible',
                                                      'Verifique la conexion o CAMERA_SOURCE. Reintentando...'))
                        captura.release()
                        captura = None
                        time.sleep(2)
                        continue
                    self.camara_ok, self.error = True, ''
                    fps_fuente = captura.get(cv2.CAP_PROP_FPS) or 30

                ok, cuadro = captura.read()
                if not ok or cuadro is None:
                    if es_archivo:  # los videos de prueba se reproducen en bucle
                        captura.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self.camara_ok = False
                    self.error = 'Se perdió la señal de la cámara. Reintentando...'
                    captura.release()
                    captura = None
                    time.sleep(1)
                    continue

                if cuadro.shape[1] > self.ancho:  # mantiene un tamaño de trabajo estable
                    factor = self.ancho / cuadro.shape[1]
                    cuadro = cv2.resize(cuadro, (self.ancho, int(cuadro.shape[0] * factor)))
                cuadro = cv2.flip(cuadro, 1) if not es_archivo else cuadro  # efecto espejo

                try:
                    anotado, self.info = self.procesador.procesar(cuadro)
                except Exception:  # un error en un cuadro no debe tumbar el video
                    log.exception('Error procesando cuadro')
                    anotado = cuadro
                self._publicar(anotado)

                if es_archivo:
                    time.sleep(1.0 / min(max(fps_fuente, 1), 60))
        finally:
            if captura is not None:
                captura.release()
            self.camara_ok = False
            self._activo.clear()
            with self._cond:
                self._cond.notify_all()
