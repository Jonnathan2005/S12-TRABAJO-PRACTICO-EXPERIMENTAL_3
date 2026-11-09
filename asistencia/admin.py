from django.contrib import admin

from .models import Asistencia, EntrenamientoModelo, Persona


@admin.register(Persona)
class PersonaAdmin(admin.ModelAdmin):
    list_display = ('apellidos', 'nombres', 'cedula', 'curso', 'activo', 'creado')
    list_filter = ('activo', 'curso')
    search_fields = ('nombres', 'apellidos', 'cedula')


@admin.register(Asistencia)
class AsistenciaAdmin(admin.ModelAdmin):
    list_display = ('persona', 'fecha', 'hora', 'estado', 'origen', 'confianza')
    list_filter = ('estado', 'origen', 'fecha')
    search_fields = ('persona__nombres', 'persona__apellidos', 'persona__cedula')
    date_hierarchy = 'fecha'


@admin.register(EntrenamientoModelo)
class EntrenamientoModeloAdmin(admin.ModelAdmin):
    list_display = ('fecha', 'personas', 'muestras', 'duracion_ms', 'exito')
    readonly_fields = ('fecha', 'personas', 'muestras', 'duracion_ms', 'exito', 'detalle')
