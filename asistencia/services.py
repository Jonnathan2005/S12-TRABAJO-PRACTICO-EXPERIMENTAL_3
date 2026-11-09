"""
Capa de servicios: conecta el paquete ``vision`` (OpenCV puro) con Django
(base de datos PostgreSQL, configuración y archivos).
"""

import logging
import shutil
import threading
import time
from datetime import datetime

import cv2
from django.conf import settings
from django.db import close_old_connections, transaction
from django.utils import timezone

from .models import Asistencia, EntrenamientoModelo, Persona
from .vision.camara import ServicioVideo
from .vision.detector import DetectorRostros
from .vision.emociones import ClasificadorEmociones
from .vision.procesador import ProcesadorCuadros, SesionRegistro, guardar_imagen, leer_imagen
from .vision.procesamiento import TAMANO_ROSTRO, nitidez, normalizar_rostro
from .vision.reconocedor import ReconocedorFacial

log = logging.getLogger(__name__)

_lock = threading.RLock()
_lock_entrenamiento = threading.Lock()
_lock_imagenes = threading.Lock()
_reconocedor = None
_procesador = None
_servicio = None
_detector_imagenes = None


# --------------------------------------------------------------------------
# Instancias compartidas (se crean bajo demanda, una sola vez por proceso)
# --------------------------------------------------------------------------
def obtener_reconocedor():
    global _reconocedor
    with _lock:
        if _reconocedor is None:
            vision = settings.VISION
            _reconocedor = ReconocedorFacial(vision['RECOGNITION_THRESHOLD'], vision['MODEL_PATH'])
            try:
                _reconocedor.cargar()
            except Exception:
                log.exception('No se pudo cargar el modelo guardado; se entrenará de nuevo.')
        return _reconocedor


def obtener_procesador():
    global _procesador
    with _lock:
        if _procesador is None:
            # La cámara solo detecta emociones: los ojos no se usan y se omiten por velocidad.
            _procesador = ProcesadorCuadros(
                detector=DetectorRostros(detectar_ojos=False),
                emociones=ClasificadorEmociones(settings.VISION['EMOTION_MODEL_PATH'],
                                                descargar=settings.VISION['EMOTION_MODEL_DOWNLOAD']),
                al_completar_registro=_al_completar_registro,
            )
        return _procesador


def obtener_servicio_video():
    global _servicio
    with _lock:
        if _servicio is None:
            vision = settings.VISION
            _servicio = ServicioVideo(
                obtener_procesador(),
                fuente=vision['CAMERA_SOURCE'],
                ancho=vision['FRAME_WIDTH'],
                alto=vision['FRAME_HEIGHT'],
                calidad_jpeg=vision['JPEG_QUALITY'],
                inactividad=vision['IDLE_TIMEOUT'],
            )
        return _servicio


def reiniciar_instancias():
    """Detiene la cámara y descarta las instancias compartidas (útil en pruebas)."""
    global _reconocedor, _procesador, _servicio, _detector_imagenes
    with _lock:
        if _servicio is not None:
            _servicio.detener()
        _reconocedor = _procesador = _servicio = _detector_imagenes = None


def procesador_existente():
    """Devuelve el procesador solo si ya fue creado (no carga modelos)."""
    return _procesador


def servicio_video_existente():
    """Devuelve el servicio solo si ya fue creado (no abre la cámara)."""
    return _servicio


# --------------------------------------------------------------------------
# Asistencia
# --------------------------------------------------------------------------
def hora_limite():
    return datetime.strptime(settings.HORA_LIMITE_PUNTUAL, '%H:%M').time()


def calcular_estado(hora):
    return Asistencia.Estado.PUNTUAL if hora <= hora_limite() else Asistencia.Estado.ATRASO


def registrar_asistencia(persona_id, confianza=0.0, origen=Asistencia.Origen.CAMARA, momento=None):
    """
    Registra la asistencia de hoy si aún no existe.
    Devuelve ``{'registrado': bool, 'mensaje': str, 'asistencia': Asistencia|None}``.
    """
    momento = timezone.localtime(momento or timezone.now())
    try:
        persona = Persona.objects.get(pk=persona_id, activo=True)
    except Persona.DoesNotExist:
        return {'registrado': False, 'mensaje': None, 'asistencia': None}

    hora = momento.time().replace(microsecond=0)
    with transaction.atomic():
        asistencia, creada = Asistencia.objects.get_or_create(
            persona=persona,
            fecha=momento.date(),
            defaults={
                'hora': hora,
                'confianza': round(float(confianza), 1),
                'estado': calcular_estado(hora),
                'origen': origen,
            },
        )
    mensaje = f'{asistencia.hora:%H:%M} ({asistencia.get_estado_display().lower()})'
    if creada:
        log.info('Asistencia registrada: %s %s', persona, mensaje)
    return {'registrado': creada, 'mensaje': mensaje, 'asistencia': asistencia}


# --------------------------------------------------------------------------
# Entrenamiento del modelo
# --------------------------------------------------------------------------
def cargar_muestras():
    rostros, etiquetas, personas = [], [], set()
    for persona in Persona.objects.filter(activo=True):
        for ruta in persona.archivos_muestras():
            img = leer_imagen(ruta)
            if img is None:
                continue
            if img.shape[:2] != TAMANO_ROSTRO[::-1]:
                img = cv2.resize(img, TAMANO_ROSTRO)
            rostros.append(img)
            etiquetas.append(persona.pk)
            personas.add(persona.pk)
    return rostros, etiquetas, personas


def entrenar_modelo():
    """Entrena el LBPH con todas las muestras de las personas activas y lo guarda en disco."""
    with _lock_entrenamiento:
        inicio = time.perf_counter()
        reconocedor = obtener_reconocedor()
        rostros, etiquetas, personas = cargar_muestras()
        try:
            if rostros:
                reconocedor.entrenar(rostros, etiquetas)
                reconocedor.guardar()
                detalle = 'Modelo entrenado correctamente.'
            else:
                reconocedor.olvidar()
                detalle = 'No hay muestras: el modelo quedó vacío.'
            exito = True
        except Exception as exc:
            log.exception('Fallo el entrenamiento')
            exito, detalle = False, f'Error: {exc}'[:250]

        registro = EntrenamientoModelo.objects.create(
            personas=len(personas),
            muestras=len(rostros),
            duracion_ms=int((time.perf_counter() - inicio) * 1000),
            exito=exito,
            detalle=detalle,
        )
        return registro


def _al_completar_registro(sesion):
    def tarea():
        close_old_connections()
        try:
            resultado = entrenar_modelo()
            sesion.mensaje = (f'¡Listo! {sesion.guardadas} muestras guardadas y modelo entrenado '
                              f'({resultado.duracion_ms} ms).')
        except Exception as exc:
            sesion.mensaje = f'Las muestras se guardaron, pero falló el entrenamiento: {exc}'
        finally:
            close_old_connections()

    threading.Thread(target=tarea, name='EntrenamientoLBPH', daemon=True).start()


# --------------------------------------------------------------------------
# Registro de rostros (muestras)
# --------------------------------------------------------------------------
def iniciar_registro(persona, objetivo=None):
    sesion = SesionRegistro(
        persona_id=persona.pk,
        nombre=persona.nombre_completo,
        carpeta=persona.carpeta_muestras,
        objetivo=objetivo or settings.VISION['ENROLL_SAMPLES'],
    )
    obtener_procesador().iniciar_registro(sesion)
    obtener_servicio_video().tocar()
    return sesion


def _detector_para_imagenes():
    global _detector_imagenes
    if _detector_imagenes is None:
        _detector_imagenes = DetectorRostros(ancho_proceso=800, vecinos=5, tam_minimo=40)
    return _detector_imagenes


def agregar_muestras_desde_imagen(persona, imagen):
    """Detecta el rostro más grande de una foto subida y lo guarda como muestra."""
    with _lock_imagenes:
        rostros = _detector_para_imagenes().detectar(imagen)
    if not rostros:
        return 0
    cara = normalizar_rostro(imagen, rostros[0].rect)
    if nitidez(cara) < 10:
        return 0
    nombre = f'foto_{int(time.time() * 1000)}.png'
    guardado = guardar_imagen(f'{persona.carpeta_muestras}/{nombre}', cara)
    # Se agrega también la versión espejada para enriquecer el entrenamiento.
    guardar_imagen(f'{persona.carpeta_muestras}/espejo_{nombre}', cv2.flip(cara, 1))
    return 2 if guardado else 0


def eliminar_muestras(persona):
    shutil.rmtree(persona.carpeta_muestras, ignore_errors=True)
