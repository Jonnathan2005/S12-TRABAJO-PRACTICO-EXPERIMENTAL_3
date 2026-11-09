"""Utilidades compartidas por las pruebas."""

import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings
from django.test import override_settings

from asistencia import services


def textura(semilla, ruido=0, tam=200):
    """Imagen en gris con un patrón reproducible (simula un 'rostro' para LBPH)."""
    rng = np.random.default_rng(semilla)
    base = cv2.resize(rng.integers(0, 256, (20, 20), dtype=np.uint8), (tam, tam), interpolation=cv2.INTER_CUBIC)
    if ruido:
        extra = np.random.default_rng(semilla * 1000 + ruido).normal(0, 6, base.shape)
        base = np.clip(base.astype(np.float32) + extra, 0, 255).astype(np.uint8)
    return base


def imagen_png(color=(200, 200, 200), tam=(320, 240)):
    img = np.full((tam[1], tam[0], 3), color, dtype=np.uint8)
    ok, buf = cv2.imencode('.png', img)
    return buf.tobytes()


class EntornoVisionTemporal:
    """
    Mixin para TestCase: redirige muestras y modelo a una carpeta temporal y
    reinicia las instancias compartidas para no tocar los datos reales.
    """

    def setUp(self):
        super().setUp()
        self._tmp = Path(tempfile.mkdtemp(prefix='va_test_'))
        vision = dict(settings.VISION, SAMPLES_DIR=self._tmp / 'rostros',
                      MODEL_PATH=self._tmp / 'modelo' / 'lbph.yml', CAMERA_SOURCE='99',
                      EMOTION_MODEL_PATH=self._tmp / 'modelo' / 'emociones.onnx', EMOTION_MODEL_DOWNLOAD=False)
        self._override = override_settings(VISION=vision, MEDIA_ROOT=self._tmp)
        self._override.enable()
        services.reiniciar_instancias()

    def tearDown(self):
        services.reiniciar_instancias()
        self._override.disable()
        shutil.rmtree(self._tmp, ignore_errors=True)
        super().tearDown()
