import os

import cv2
import numpy as np
from django.conf import settings
from django.contrib import messages
from django.db.models import Q
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from . import services
from .forms import FiltroAsistenciaForm, PersonaForm, SubirMuestrasForm
from .models import Asistencia, EntrenamientoModelo, Persona


def decodificar_imagen(archivo):
    """Convierte un archivo subido en una imagen BGR de OpenCV (o None si no es válida)."""
    datos = np.frombuffer(archivo.read(), dtype=np.uint8)
    if datos.size == 0:
        return None
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def asistencia_a_dict(a):
    return {
        'id': a.pk,
        'persona_id': a.persona_id,
        'persona': a.persona.nombre_completo,
        'iniciales': a.persona.iniciales,
        'curso': a.persona.curso,
        'fecha': a.fecha.isoformat(),
        'hora': a.hora.strftime('%H:%M:%S'),
        'confianza': a.confianza,
        'estado': a.estado,
        'estado_display': a.get_estado_display(),
        'origen': a.get_origen_display(),
    }


# --------------------------------------------------------------------------
# Panel principal
# --------------------------------------------------------------------------
def panel(request):
    hoy = timezone.localdate()
    personas_activas = Persona.objects.filter(activo=True).count()
    hoy_qs = Asistencia.objects.filter(fecha=hoy)
    presentes = hoy_qs.count()

    contexto = {
        'personas_activas': personas_activas,
        'presentes': presentes,
        'puntuales': hoy_qs.filter(estado='puntual').count(),
        'atrasos': hoy_qs.filter(estado='atraso').count(),
        'ausentes': max(0, personas_activas - presentes),
        'porcentaje': round(100 * presentes / personas_activas) if personas_activas else 0,
        'recientes': Asistencia.objects.select_related('persona')[:8],
        'ultimo_entrenamiento': EntrenamientoModelo.objects.first(),
        'modelo_entrenado': services.obtener_reconocedor().entrenado,
        'sin_muestras': [p for p in Persona.objects.filter(activo=True) if p.total_muestras == 0][:5],
        'hora_limite': settings.HORA_LIMITE_PUNTUAL,
    }
    return render(request, 'asistencia/panel.html', contexto)


# --------------------------------------------------------------------------
# Video en vivo
# --------------------------------------------------------------------------
def en_vivo(request):
    return render(request, 'asistencia/en_vivo.html')


def stream_video(request):
    """Transmisión MJPEG del video procesado por OpenCV."""
    servicio = services.obtener_servicio_video()
    respuesta = StreamingHttpResponse(servicio.generar_mjpeg(),
                                      content_type='multipart/x-mixed-replace; boundary=frame')
    respuesta['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return respuesta


@require_GET
def api_estado(request):
    servicio = services.servicio_video_existente()
    datos = servicio.estado() if servicio else {'en_ejecucion': False, 'camara_ok': False, 'error': ''}
    return JsonResponse(datos)


@require_POST
def api_reiniciar_emociones(request):
    """Pone en cero el resumen de emociones acumulado durante la sesión de cámara."""
    procesador = services.procesador_existente()
    if procesador:
        procesador.reiniciar_resumen()
    return JsonResponse({'ok': True})


@require_POST
def api_detener_camara(request):
    servicio = services.servicio_video_existente()
    if servicio:
        servicio.detener()
    return JsonResponse({'ok': True})


@require_GET
def api_asistencias_hoy(request):
    hoy = timezone.localdate()
    qs = Asistencia.objects.filter(fecha=hoy).select_related('persona').order_by('-hora')
    return JsonResponse({
        'fecha': hoy.isoformat(),
        'total': qs.count(),
        'activos': Persona.objects.filter(activo=True).count(),
        'asistencias': [asistencia_a_dict(a) for a in qs],
    })


# --------------------------------------------------------------------------
# Personas
# --------------------------------------------------------------------------
def personas_lista(request):
    q = request.GET.get('q', '').strip()
    personas = Persona.objects.all()
    if q:
        personas = personas.filter(
            Q(nombres__icontains=q) | Q(apellidos__icontains=q) | Q(cedula__icontains=q) | Q(curso__icontains=q)
        )
    hoy = timezone.localdate()
    presentes = set(Asistencia.objects.filter(fecha=hoy).values_list('persona_id', flat=True))
    filas = [{'persona': p, 'muestras': p.total_muestras, 'presente': p.pk in presentes} for p in personas]
    return render(request, 'asistencia/personas_lista.html', {'filas': filas, 'q': q})


def persona_crear(request):
    form = PersonaForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        persona = form.save()
        messages.success(request, f'{persona} fue registrado. Ahora capture su rostro.')
        return redirect('asistencia:persona_detalle', pk=persona.pk)
    return render(request, 'asistencia/persona_form.html', {'form': form, 'titulo': 'Nueva persona'})


def persona_editar(request, pk):
    persona = get_object_or_404(Persona, pk=pk)
    form = PersonaForm(request.POST or None, instance=persona)
    if request.method == 'POST' and form.is_valid():
        form.save()
        messages.success(request, 'Datos actualizados.')
        return redirect('asistencia:persona_detalle', pk=persona.pk)
    return render(request, 'asistencia/persona_form.html', {'form': form, 'titulo': f'Editar {persona}',
                                                            'persona': persona})


@require_POST
def persona_eliminar(request, pk):
    persona = get_object_or_404(Persona, pk=pk)
    nombre = str(persona)
    services.eliminar_muestras(persona)
    persona.delete()
    services.entrenar_modelo()
    messages.success(request, f'{nombre} fue eliminado y el modelo se reentrenó.')
    return redirect('asistencia:personas')


def persona_detalle(request, pk):
    persona = get_object_or_404(Persona, pk=pk)
    archivos = persona.archivos_muestras()
    miniaturas = [
        settings.MEDIA_URL + os.path.relpath(r, settings.MEDIA_ROOT).replace(os.sep, '/') for r in archivos[-18:]
    ]
    return render(request, 'asistencia/persona_detalle.html', {
        'persona': persona,
        'total_muestras': len(archivos),
        'miniaturas': miniaturas,
        'asistencias': persona.asistencias.all()[:15],
        'form_fotos': SubirMuestrasForm(),
        'objetivo': settings.VISION['ENROLL_SAMPLES'],
    })


@require_POST
def persona_subir_fotos(request, pk):
    persona = get_object_or_404(Persona, pk=pk)
    form = SubirMuestrasForm(request.POST, request.FILES)
    if not form.is_valid():
        for error in form.errors.get('fotos', []):
            messages.error(request, error)
        return redirect('asistencia:persona_detalle', pk=pk)

    agregadas, sin_rostro = 0, []
    for archivo in form.cleaned_data['fotos']:
        imagen = decodificar_imagen(archivo)
        n = services.agregar_muestras_desde_imagen(persona, imagen) if imagen is not None else 0
        if n:
            agregadas += n
        else:
            sin_rostro.append(archivo.name)
    if agregadas:
        services.entrenar_modelo()
        messages.success(request, f'Se agregaron {agregadas} muestras y se reentrenó el modelo.')
    if sin_rostro:
        messages.warning(request, 'No se detectó un rostro claro en: ' + ', '.join(sin_rostro))
    return redirect('asistencia:persona_detalle', pk=pk)


@require_POST
def persona_borrar_muestras(request, pk):
    persona = get_object_or_404(Persona, pk=pk)
    services.eliminar_muestras(persona)
    services.entrenar_modelo()
    messages.success(request, 'Muestras eliminadas y modelo reentrenado.')
    return redirect('asistencia:persona_detalle', pk=pk)


@require_POST
def api_registro_iniciar(request, pk):
    persona = get_object_or_404(Persona, pk=pk, activo=True)
    sesion = services.iniciar_registro(persona)
    return JsonResponse({'ok': True, 'registro': sesion.como_dict()})


@require_GET
def api_registro_estado(request):
    procesador = services.obtener_procesador()
    sesion = procesador.registro
    return JsonResponse({'registro': sesion.como_dict() if sesion else None})


@require_POST
def api_registro_cancelar(request):
    services.obtener_procesador().cancelar_registro()
    return JsonResponse({'ok': True})


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------
@require_POST
def entrenar(request):
    resultado = services.entrenar_modelo()
    if resultado.exito:
        messages.success(request, f'{resultado.detalle} {resultado.personas} personas, '
                                  f'{resultado.muestras} muestras en {resultado.duracion_ms} ms.')
    else:
        messages.error(request, resultado.detalle)
    return redirect(request.POST.get('siguiente') or 'asistencia:panel')


# --------------------------------------------------------------------------
# Asistencias
# --------------------------------------------------------------------------
def _asistencias_filtradas(request):
    form = FiltroAsistenciaForm(request.GET or None)
    qs = Asistencia.objects.select_related('persona')
    if form.is_valid():
        d = form.cleaned_data
        if d.get('desde'):
            qs = qs.filter(fecha__gte=d['desde'])
        if d.get('hasta'):
            qs = qs.filter(fecha__lte=d['hasta'])
        if d.get('persona'):
            qs = qs.filter(persona=d['persona'])
        if d.get('estado'):
            qs = qs.filter(estado=d['estado'])
    return form, qs


def asistencias_lista(request):
    form, qs = _asistencias_filtradas(request)
    return render(request, 'asistencia/asistencias.html', {
        'form': form,
        'asistencias': qs[:300],
        'total': qs.count(),
    })
