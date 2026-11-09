from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from asistencia.vision.emociones import URL_MODELO, ClasificadorEmociones


class Command(BaseCommand):
    help = 'Descarga la red FER+ (ONNX, ~34 MB) usada para detectar emociones en la cámara.'

    def handle(self, *args, **options):
        ruta = settings.VISION['EMOTION_MODEL_PATH']
        if ruta.exists():
            self.stdout.write(self.style.SUCCESS(f'El modelo ya existe en {ruta}'))
            return
        self.stdout.write(f'Descargando {URL_MODELO} ...')
        try:
            ClasificadorEmociones(ruta).descargar_modelo()
        except Exception as exc:
            raise CommandError(f'No se pudo descargar el modelo: {exc}')
        self.stdout.write(self.style.SUCCESS(f'Modelo guardado en {ruta}'))
