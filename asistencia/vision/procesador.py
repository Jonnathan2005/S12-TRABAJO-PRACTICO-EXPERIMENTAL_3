"""
Tubería de procesamiento por cuadro:

    cuadro BGR ─► detección de rostros (Haar) ─► recorte gris 64x64
               ─► clasificación de la expresión (FER+ con cv2.dnn)
               ─► suavizado entre cuadros ─► dibujo de la emoción

También gestiona la sesión de *registro guiado*, en la que se capturan
muestras del rostro de una persona para entrenar el modelo.
"""

import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import dibujo
from .emociones import COLORES, EMOCIONES, NOMBRES, SuavizadorEmociones
from .procesamiento import a_gris, brillo, evaluar_iluminacion, nitidez, normalizar_rostro

NITIDEZ_MINIMA_MUESTRA = 25.0


def guardar_imagen(ruta, imagen):
    """``cv2.imwrite`` falla con rutas con caracteres no ASCII en Windows; se usa imencode."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    ok, buffer = cv2.imencode('.png', imagen)
    if ok:
        buffer.tofile(ruta)
    return ok


def leer_imagen(ruta, flags=cv2.IMREAD_GRAYSCALE):
    datos = np.fromfile(ruta, dtype=np.uint8)
    if datos.size == 0:
        return None
    return cv2.imdecode(datos, flags)


@dataclass
class SesionRegistro:
    """Estado de la captura guiada de muestras de rostro de una persona."""

    persona_id: int
    nombre: str
    carpeta: str
    objetivo: int = 30
    guardadas: int = 0
    descartadas: int = 0
    terminada: bool = False
    cancelada: bool = False
    mensaje: str = 'Mire a la cámara y mueva ligeramente la cabeza.'
    _cuadro: int = field(default=0, repr=False)

    @property
    def progreso(self):
        return min(1.0, self.guardadas / self.objetivo) if self.objetivo else 1.0

    def como_dict(self):
        return {
            'persona_id': self.persona_id,
            'nombre': self.nombre,
            'objetivo': self.objetivo,
            'guardadas': self.guardadas,
            'descartadas': self.descartadas,
            'progreso': round(self.progreso * 100),
            'terminada': self.terminada,
            'cancelada': self.cancelada,
            'mensaje': self.mensaje,
        }


class ProcesadorCuadros:
    """
    Procesa cuadros de video: detecta los rostros y clasifica la emoción de
    cada uno a partir de su expresión facial. Es independiente de Django.

    También atiende la sesión de *registro guiado* de muestras de rostro; al
    terminar llama a ``al_completar_registro(sesion)``.
    """

    def __init__(self, detector, emociones=None, al_completar_registro=None):
        self.detector = detector
        self.emociones = emociones
        self.al_completar_registro = al_completar_registro

        self.registro = None       # SesionRegistro activa
        self.suavizador = SuavizadorEmociones()
        self.conteo_emociones = {e: 0 for e in EMOCIONES}  # cuadros por emoción dominante
        self._lock = threading.Lock()

        self._t_anterior = time.perf_counter()
        self.fps = 0.0
        self.ultimo_resultado = {}

    # ----------------------------------------------------------------- config
    def reiniciar_resumen(self):
        with self._lock:
            self.conteo_emociones = {e: 0 for e in EMOCIONES}
            self.suavizador.reiniciar()

    def iniciar_registro(self, sesion):
        with self._lock:
            self.registro = sesion

    def cancelar_registro(self):
        with self._lock:
            if self.registro and not self.registro.terminada:
                self.registro.cancelada = True
                self.registro.terminada = True
                self.registro.mensaje = 'Registro cancelado.'

    # -------------------------------------------------------------- principal
    def procesar(self, cuadro):
        """Procesa un cuadro BGR y devuelve ``(cuadro_anotado, info)``."""
        self._actualizar_fps()

        rostros = self.detector.detectar(cuadro)
        gris = a_gris(cuadro)
        valor_brillo = brillo(gris)

        salida = cuadro.copy()
        sesion = self.registro if (self.registro and not self.registro.terminada) else None

        if sesion:
            caras = []
            self._procesar_registro(cuadro, rostros, salida, sesion)
        else:
            caras = self._detectar_emociones(cuadro, rostros, salida)

        dominante = caras[0]['emocion'] if caras else None
        dibujo.hud(salida, [
            f'FPS {self.fps:4.1f} | Rostros: {len(rostros)}',
            f'Iluminacion: {evaluar_iluminacion(valor_brillo)} ({valor_brillo:.0f})',
        ])

        total = sum(self.conteo_emociones.values())
        info = {
            'fps': round(self.fps, 1),
            'rostros': len(rostros),
            'caras': caras,
            'emocion_principal': NOMBRES[dominante] if dominante else None,
            'resumen': [
                {'emocion': e, 'nombre': NOMBRES[e], 'cuadros': n,
                 'porcentaje': round(100 * n / total, 1) if total else 0.0}
                for e, n in self.conteo_emociones.items()
            ],
            'brillo': round(valor_brillo, 1),
            'iluminacion': evaluar_iluminacion(valor_brillo),
            'modo_emociones': self.emociones.modo if self.emociones else 'no disponible',
            'registro': self.registro.como_dict() if self.registro else None,
        }
        self.ultimo_resultado = info
        return salida, info

    # ------------------------------------------------------------ internos
    def _actualizar_fps(self):
        ahora = time.perf_counter()
        dt = ahora - self._t_anterior
        self._t_anterior = ahora
        if dt > 0:
            instantaneo = 1.0 / dt
            self.fps = instantaneo if self.fps == 0 else 0.9 * self.fps + 0.1 * instantaneo

    def _detectar_emociones(self, cuadro, rostros, salida):
        """Clasifica la expresión de cada rostro, la dibuja y devuelve el detalle por rostro."""
        if self.emociones is None:
            crudas = [None] * len(rostros)
        else:
            crudas = [self.emociones.clasificar(cuadro, r.rect) for r in rostros]
        suavizadas = self.suavizador.actualizar([r.rect for r in rostros], crudas)

        caras = []
        for numero, (rostro, probs) in enumerate(zip(rostros, suavizadas), start=1):
            if probs is None:
                dibujo.esquinas(salida, rostro.rect, dibujo.BLANCO)
                dibujo.etiqueta(salida, 'Emocion no disponible', rostro.x, rostro.y - 4, dibujo.NEGRO)
                continue
            orden = np.argsort(probs)[::-1]
            emocion = EMOCIONES[orden[0]]
            confianza = float(probs[orden[0]]) * 100
            with self._lock:
                self.conteo_emociones[emocion] += 1

            color = COLORES[emocion]
            dibujo.esquinas(salida, rostro.rect, color)
            dibujo.etiqueta(salida, f'{NOMBRES[emocion]}  {confianza:.0f}%', rostro.x, rostro.y - 4, color)
            dibujo.barras_emociones(salida, rostro.rect,
                                    [(NOMBRES[EMOCIONES[i]], float(probs[i]), COLORES[EMOCIONES[i]])
                                     for i in orden[:3]])
            caras.append({
                'numero': numero,
                'emocion': emocion,
                'nombre': NOMBRES[emocion],
                'confianza': round(confianza, 1),
                'probabilidades': {EMOCIONES[i]: round(float(probs[i]) * 100, 1) for i in orden},
            })
        return caras

    def _procesar_registro(self, cuadro, rostros, salida, sesion):
        sesion._cuadro += 1
        if len(rostros) == 0:
            sesion.mensaje = 'No se detecta ningún rostro. Acérquese a la cámara.'
        elif len(rostros) > 1:
            sesion.mensaje = 'Hay más de un rostro: solo debe aparecer la persona a registrar.'
        else:
            rostro = rostros[0]
            cara = normalizar_rostro(cuadro, rostro.rect)
            # Se guarda una muestra cada 2 cuadros para tener variedad de poses.
            if sesion._cuadro % 2 == 0:
                if nitidez(cara) < NITIDEZ_MINIMA_MUESTRA:
                    sesion.descartadas += 1
                    sesion.mensaje = 'Imagen borrosa descartada; quédese quieto un momento.'
                else:
                    nombre_archivo = f'{int(time.time() * 1000)}_{sesion.guardadas:03d}.png'
                    if guardar_imagen(os.path.join(sesion.carpeta, nombre_archivo), cara):
                        sesion.guardadas += 1
                        sesion.mensaje = 'Capturando muestras... mueva ligeramente la cabeza.'

        for rostro in rostros:
            dibujo.esquinas(salida, rostro.rect, dibujo.AZUL)
            dibujo.ojos(salida, rostro.ojos)
        dibujo.barra_progreso(salida, sesion.progreso,
                              f'Registrando a {sesion.nombre}: {sesion.guardadas}/{sesion.objetivo}')

        if sesion.guardadas >= sesion.objetivo:
            sesion.terminada = True
            sesion.mensaje = 'Muestras completas. Entrenando el modelo...'
            if self.al_completar_registro:
                self.al_completar_registro(sesion)
