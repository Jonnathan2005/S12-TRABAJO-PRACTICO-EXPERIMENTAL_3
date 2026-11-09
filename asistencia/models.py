import os

from django.conf import settings
from django.core.validators import RegexValidator
from django.db import models

validar_cedula = RegexValidator(r'^\d{10}$', 'La cédula debe tener exactamente 10 dígitos.')


class Persona(models.Model):
    """Estudiante o asistente que puede ser reconocido por la cámara."""

    nombres = models.CharField(max_length=80)
    apellidos = models.CharField(max_length=80)
    cedula = models.CharField(max_length=10, unique=True, validators=[validar_cedula])
    correo = models.EmailField(blank=True)
    curso = models.CharField('Curso / paralelo', max_length=60, blank=True)
    activo = models.BooleanField(default=True)
    creado = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['apellidos', 'nombres']

    def __str__(self):
        return self.nombre_completo

    @property
    def nombre_completo(self):
        return f'{self.nombres} {self.apellidos}'.strip()

    @property
    def iniciales(self):
        return (self.nombres[:1] + self.apellidos[:1]).upper()

    @property
    def carpeta_muestras(self):
        return os.path.join(settings.VISION['SAMPLES_DIR'], f'persona_{self.pk}')

    def archivos_muestras(self):
        carpeta = self.carpeta_muestras
        if not os.path.isdir(carpeta):
            return []
        return sorted(os.path.join(carpeta, f) for f in os.listdir(carpeta) if f.lower().endswith('.png'))

    @property
    def total_muestras(self):
        return len(self.archivos_muestras())


class Asistencia(models.Model):
    """Registro de asistencia diario (máximo uno por persona y día)."""

    class Estado(models.TextChoices):
        PUNTUAL = 'puntual', 'Puntual'
        ATRASO = 'atraso', 'Atraso'

    class Origen(models.TextChoices):
        CAMARA = 'camara', 'Cámara en vivo'
        MANUAL = 'manual', 'Manual'

    persona = models.ForeignKey(Persona, on_delete=models.CASCADE, related_name='asistencias')
    fecha = models.DateField()
    hora = models.TimeField()
    confianza = models.FloatField(default=0)
    estado = models.CharField(max_length=10, choices=Estado.choices, default=Estado.PUNTUAL)
    origen = models.CharField(max_length=10, choices=Origen.choices, default=Origen.CAMARA)
    creado = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-fecha', '-hora']
        constraints = [
            models.UniqueConstraint(fields=['persona', 'fecha'], name='una_asistencia_por_dia'),
        ]

    def __str__(self):
        return f'{self.persona} - {self.fecha} {self.hora:%H:%M}'


class EntrenamientoModelo(models.Model):
    """Historial de entrenamientos del reconocedor LBPH."""

    fecha = models.DateTimeField(auto_now_add=True)
    personas = models.PositiveIntegerField(default=0)
    muestras = models.PositiveIntegerField(default=0)
    duracion_ms = models.PositiveIntegerField(default=0)
    exito = models.BooleanField(default=True)
    detalle = models.CharField(max_length=255, blank=True)

    class Meta:
        ordering = ['-fecha']
        verbose_name = 'entrenamiento del modelo'
        verbose_name_plural = 'entrenamientos del modelo'

    def __str__(self):
        return f'Entrenamiento {self.fecha:%Y-%m-%d %H:%M} ({self.personas} personas)'
