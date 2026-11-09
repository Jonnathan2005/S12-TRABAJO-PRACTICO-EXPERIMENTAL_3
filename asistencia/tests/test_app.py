"""Pruebas de integración: modelos, servicios y vistas de Django."""

from datetime import datetime, time, timedelta
from unittest import mock

from django.core.files.uploadedfile import SimpleUploadedFile
from django.db import IntegrityError
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from asistencia import services
from asistencia.models import Asistencia, EntrenamientoModelo, Persona
from asistencia.vision.procesador import guardar_imagen

from .utilidades import EntornoVisionTemporal, imagen_png, textura


def crear_persona(**kw):
    datos = {'nombres': 'Ana', 'apellidos': 'Pérez', 'cedula': '0912345678', 'curso': 'Software A2'}
    datos.update(kw)
    return Persona.objects.create(**datos)


def momento(h, m):
    return timezone.make_aware(datetime.combine(timezone.localdate(), time(h, m)))


class ModelosTests(EntornoVisionTemporal, TestCase):
    def test_nombre_e_iniciales(self):
        p = crear_persona()
        self.assertEqual(p.nombre_completo, 'Ana Pérez')
        self.assertEqual(p.iniciales, 'AP')
        self.assertEqual(p.total_muestras, 0)

    def test_una_asistencia_por_persona_y_dia(self):
        p = crear_persona()
        Asistencia.objects.create(persona=p, fecha=timezone.localdate(), hora=time(8, 0))
        with self.assertRaises(IntegrityError):
            Asistencia.objects.create(persona=p, fecha=timezone.localdate(), hora=time(9, 0))


@override_settings(HORA_LIMITE_PUNTUAL='08:15')
class ServiciosAsistenciaTests(EntornoVisionTemporal, TestCase):
    def test_registra_puntual_y_no_duplica(self):
        p = crear_persona()
        r1 = services.registrar_asistencia(p.pk, 91.5, momento=momento(8, 5))
        r2 = services.registrar_asistencia(p.pk, 95.0, momento=momento(9, 0))
        self.assertTrue(r1['registrado'])
        self.assertFalse(r2['registrado'])
        self.assertEqual(Asistencia.objects.count(), 1)
        a = Asistencia.objects.get()
        self.assertEqual(a.estado, Asistencia.Estado.PUNTUAL)
        self.assertEqual(a.confianza, 91.5)
        self.assertEqual(r1['mensaje'], '08:05 (puntual)')

    def test_registra_atraso_despues_de_hora_limite(self):
        p = crear_persona()
        services.registrar_asistencia(p.pk, 80, momento=momento(8, 16))
        self.assertEqual(Asistencia.objects.get().estado, Asistencia.Estado.ATRASO)

    def test_persona_inactiva_o_inexistente_no_registra(self):
        p = crear_persona(activo=False)
        self.assertFalse(services.registrar_asistencia(p.pk, 90)['registrado'])
        self.assertFalse(services.registrar_asistencia(9999, 90)['registrado'])
        self.assertEqual(Asistencia.objects.count(), 0)


class EntrenamientoTests(EntornoVisionTemporal, TestCase):
    def _muestras(self, persona, semilla, n=4):
        for i in range(n):
            guardar_imagen(f'{persona.carpeta_muestras}/m{i}.png', textura(semilla, i + 1))

    def test_entrenar_con_muestras_reconoce_y_guarda_historial(self):
        ana, luis = crear_persona(), crear_persona(nombres='Luis', apellidos='Mora', cedula='0923456789')
        self._muestras(ana, 1)
        self._muestras(luis, 2)

        registro = services.entrenar_modelo()

        self.assertTrue(registro.exito)
        self.assertEqual((registro.personas, registro.muestras), (2, 8))
        self.assertEqual(EntrenamientoModelo.objects.count(), 1)
        reconocedor = services.obtener_reconocedor()
        reconocedor.umbral = 10_000
        self.assertEqual(reconocedor.predecir(textura(2, 77)).etiqueta, luis.pk)

    def test_entrenar_sin_muestras_deja_modelo_vacio(self):
        crear_persona()
        registro = services.entrenar_modelo()
        self.assertTrue(registro.exito)
        self.assertEqual(registro.muestras, 0)
        self.assertFalse(services.obtener_reconocedor().entrenado)


class VistasTests(EntornoVisionTemporal, TestCase):
    def test_paginas_principales_responden(self):
        p = crear_persona()
        for nombre, args in [('panel', []), ('en_vivo', []), ('personas', []), ('persona_crear', []),
                             ('persona_detalle', [p.pk]), ('persona_editar', [p.pk]), ('asistencias', [])]:
            with self.subTest(nombre):
                self.assertEqual(self.client.get(reverse(f'asistencia:{nombre}', args=args)).status_code, 200)

    def test_crear_persona_valida_cedula(self):
        url = reverse('asistencia:persona_crear')
        r = self.client.post(url, {'nombres': 'ana', 'apellidos': 'pérez', 'cedula': '123', 'activo': 'on'})
        self.assertEqual(r.status_code, 200)
        self.assertFalse(Persona.objects.exists())

        r = self.client.post(url, {'nombres': 'ana maría', 'apellidos': 'pérez', 'cedula': '0912345678', 'activo': 'on'})
        p = Persona.objects.get()
        self.assertRedirects(r, reverse('asistencia:persona_detalle', args=[p.pk]))
        self.assertEqual(p.nombres, 'Ana María')

    def test_buscar_personas(self):
        crear_persona()
        crear_persona(nombres='Luis', apellidos='Mora', cedula='0923456789')
        r = self.client.get(reverse('asistencia:personas'), {'q': 'mora'})
        self.assertContains(r, 'Luis Mora')
        self.assertNotContains(r, 'Ana Pérez')

    def test_eliminar_persona(self):
        p = crear_persona()
        r = self.client.post(reverse('asistencia:persona_eliminar', args=[p.pk]))
        self.assertRedirects(r, reverse('asistencia:personas'))
        self.assertFalse(Persona.objects.exists())

    def test_api_asistencias_hoy(self):
        p = crear_persona()
        services.registrar_asistencia(p.pk, 88)
        datos = self.client.get(reverse('asistencia:api_asistencias_hoy')).json()
        self.assertEqual(datos['total'], 1)
        self.assertEqual(datos['asistencias'][0]['persona'], 'Ana Pérez')


    def test_reiniciar_resumen_de_emociones(self):
        procesador = services.obtener_procesador()
        procesador.conteo_emociones['feliz'] = 5
        r = self.client.post(reverse('asistencia:api_reiniciar_emociones'))
        self.assertTrue(r.json()['ok'])
        self.assertEqual(procesador.conteo_emociones['feliz'], 0)

    def test_pagina_en_vivo_muestra_deteccion_de_emociones(self):
        r = self.client.get(reverse('asistencia:en_vivo'))
        self.assertContains(r, 'Detección de emociones en vivo')
        self.assertNotContains(r, 'Asistencias de hoy')

    def test_api_estado_no_abre_la_camara(self):
        datos = self.client.get(reverse('asistencia:api_estado')).json()
        self.assertFalse(datos['en_ejecucion'])
        self.assertIsNone(services.servicio_video_existente())


    def test_subir_foto_sin_rostro_avisa(self):
        p = crear_persona()
        archivo = SimpleUploadedFile('a.png', imagen_png(), content_type='image/png')
        r = self.client.post(reverse('asistencia:persona_subir_fotos', args=[p.pk]), {'fotos': [archivo]},
                             follow=True)
        self.assertContains(r, 'No se detectó un rostro claro')
        self.assertEqual(p.total_muestras, 0)

    def test_iniciar_y_cancelar_registro_guiado(self):
        p = crear_persona()
        with mock.patch.object(services.ServicioVideo, 'iniciar') as iniciar:
            r = self.client.post(reverse('asistencia:api_registro_iniciar', args=[p.pk]))
        self.assertTrue(iniciar.called)
        self.assertEqual(r.json()['registro']['persona_id'], p.pk)

        self.client.post(reverse('asistencia:api_registro_cancelar'))
        estado = self.client.get(reverse('asistencia:api_registro_estado')).json()['registro']
        self.assertTrue(estado['cancelada'])
