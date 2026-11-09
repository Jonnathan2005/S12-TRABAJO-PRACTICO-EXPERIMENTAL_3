from django.urls import path

from . import views

app_name = 'asistencia'

urlpatterns = [
    path('', views.panel, name='panel'),
    path('en-vivo/', views.en_vivo, name='en_vivo'),
    path('stream/', views.stream_video, name='stream'),

    path('personas/', views.personas_lista, name='personas'),
    path('personas/nueva/', views.persona_crear, name='persona_crear'),
    path('personas/<int:pk>/', views.persona_detalle, name='persona_detalle'),
    path('personas/<int:pk>/editar/', views.persona_editar, name='persona_editar'),
    path('personas/<int:pk>/eliminar/', views.persona_eliminar, name='persona_eliminar'),
    path('personas/<int:pk>/fotos/', views.persona_subir_fotos, name='persona_subir_fotos'),
    path('personas/<int:pk>/muestras/borrar/', views.persona_borrar_muestras, name='persona_borrar_muestras'),

    path('asistencias/', views.asistencias_lista, name='asistencias'),
    path('modelo/entrenar/', views.entrenar, name='entrenar'),

    # API JSON consumida por el JavaScript de la interfaz
    path('api/estado/', views.api_estado, name='api_estado'),
    path('api/camara/detener/', views.api_detener_camara, name='api_detener'),
    path('api/emociones/reiniciar/', views.api_reiniciar_emociones, name='api_reiniciar_emociones'),
    path('api/asistencias/hoy/', views.api_asistencias_hoy, name='api_asistencias_hoy'),
    path('api/personas/<int:pk>/registro/', views.api_registro_iniciar, name='api_registro_iniciar'),
    path('api/registro/estado/', views.api_registro_estado, name='api_registro_estado'),
    path('api/registro/cancelar/', views.api_registro_cancelar, name='api_registro_cancelar'),
]
