from django.core.management.base import BaseCommand

from asistencia import services


class Command(BaseCommand):
    help = 'Entrena el reconocedor LBPH con las muestras de rostro de todas las personas activas.'

    def handle(self, *args, **options):
        resultado = services.entrenar_modelo()
        estilo = self.style.SUCCESS if resultado.exito else self.style.ERROR
        self.stdout.write(estilo(
            f'{resultado.detalle} Personas: {resultado.personas}, muestras: {resultado.muestras}, '
            f'tiempo: {resultado.duracion_ms} ms.'
        ))
