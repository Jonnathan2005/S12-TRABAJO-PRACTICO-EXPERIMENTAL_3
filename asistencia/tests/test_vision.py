"""Pruebas unitarias del paquete ``vision`` (OpenCV puro, sin base de datos)."""

import os
import shutil
import tempfile
from unittest import TestCase, mock

import numpy as np

from asistencia.vision import procesamiento
from asistencia.vision.camara import cuadro_mensaje, interpretar_fuente
from asistencia.vision.detector import DetectorRostros, Rostro
from asistencia.vision.dibujo import texto_ascii
from asistencia.vision.emociones import (EMOCIONES, ClasificadorEmociones, SuavizadorEmociones, preparar_rostro,
                                         softmax)
from asistencia.vision.procesador import ProcesadorCuadros, SesionRegistro, leer_imagen
from asistencia.vision.reconocedor import ReconocedorFacial, distancia_a_confianza

from .utilidades import textura


class ProcesamientoTests(TestCase):
    def setUp(self):
        self.color = np.random.default_rng(1).integers(0, 256, (240, 320, 3), dtype=np.uint8)

    def test_normalizar_rostro_devuelve_gris_200x200(self):
        cara = procesamiento.normalizar_rostro(self.color, (10, 20, 100, 120))
        self.assertEqual(cara.shape, (200, 200))
        self.assertEqual(cara.dtype, np.uint8)

    def test_nitidez_imagen_plana_es_cero_y_textura_es_mayor(self):
        plana = np.full((100, 100), 128, np.uint8)
        self.assertEqual(procesamiento.nitidez(plana), 0.0)
        self.assertGreater(procesamiento.nitidez(textura(3)), 0.0)

    def test_evaluar_iluminacion(self):
        self.assertEqual(procesamiento.evaluar_iluminacion(20), 'baja')
        self.assertEqual(procesamiento.evaluar_iluminacion(128), 'adecuada')
        self.assertEqual(procesamiento.evaluar_iluminacion(240), 'excesiva')



class DetectorTests(TestCase):
    def setUp(self):
        self.detector = DetectorRostros()

    def test_imagen_sin_rostros(self):
        self.assertEqual(self.detector.detectar(np.zeros((480, 640, 3), np.uint8)), [])

    def test_escala_coordenadas_al_tamano_original_y_ordena_por_area(self):
        # Cuadro de 960 px de ancho: se procesa a 480 px (factor 0.5).
        cuadro = np.zeros((720, 960, 3), np.uint8)
        self.detector.detectar_ojos = False
        self.detector.cascada_rostro = mock.Mock()
        self.detector.cascada_rostro.detectMultiScale.return_value = np.array([[10, 10, 20, 20], [50, 40, 60, 60]])
        rostros = self.detector.detectar(cuadro)
        imagen_procesada = self.detector.cascada_rostro.detectMultiScale.call_args.args[0]
        self.assertEqual(imagen_procesada.shape, (360, 480))
        self.assertEqual([r.rect for r in rostros], [(100, 80, 120, 120), (20, 20, 40, 40)])

    def test_rostro_area(self):
        self.assertEqual(Rostro(0, 0, 10, 20).area, 200)


class ReconocedorTests(TestCase):
    def _entrenar(self, umbral):
        rec = ReconocedorFacial(umbral=umbral)
        rostros = [textura(1, i) for i in range(1, 6)] + [textura(2, i) for i in range(1, 6)]
        rec.entrenar(rostros, [1] * 5 + [2] * 5)
        return rec

    def test_sin_modelo_devuelve_desconocido(self):
        pred = ReconocedorFacial().predecir(textura(1))
        self.assertFalse(pred.reconocido)
        self.assertEqual(pred.confianza, 0.0)

    def test_reconoce_la_etiqueta_correcta(self):
        rec = self._entrenar(umbral=10_000)
        self.assertEqual(rec.predecir(textura(1, 99)).etiqueta, 1)
        self.assertEqual(rec.predecir(textura(2, 99)).etiqueta, 2)

    def test_umbral_estricto_rechaza(self):
        rec = self._entrenar(umbral=0.001)
        self.assertIsNone(rec.predecir(textura(1, 99)).etiqueta)

    def test_entrenar_con_listas_de_distinto_largo_falla(self):
        with self.assertRaises(ValueError):
            ReconocedorFacial().entrenar([textura(1)], [1, 2])

    def test_guardar_y_cargar_modelo(self):
        carpeta = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, carpeta, True)
        ruta = os.path.join(carpeta, 'sub', 'modelo.yml')
        rec = ReconocedorFacial(umbral=10_000, ruta_modelo=ruta)
        rec.entrenar([textura(1, i) for i in range(3)] + [textura(2, i) for i in range(3)], [1, 1, 1, 2, 2, 2])
        self.assertTrue(rec.guardar())

        otro = ReconocedorFacial(umbral=10_000, ruta_modelo=ruta)
        self.assertTrue(otro.cargar())
        self.assertEqual(otro.predecir(textura(2, 50)).etiqueta, 2)

        otro.olvidar()
        self.assertFalse(otro.entrenado)
        self.assertFalse(os.path.exists(ruta))

    def test_distancia_a_confianza_en_rango(self):
        self.assertEqual(distancia_a_confianza(0, 70), 100.0)
        self.assertEqual(distancia_a_confianza(1000, 70), 0.0)
        self.assertTrue(0 < distancia_a_confianza(50, 70) < 100)


class DetectorFalso:
    def __init__(self, rostros):
        self.rostros = rostros

    def detectar(self, imagen):
        return list(self.rostros)


class EmocionesFalsas:
    """Clasificador de prueba: siempre devuelve la misma emoción."""

    modo = 'prueba'

    def __init__(self, emocion='feliz', prob=0.9):
        self.emocion, self.prob = emocion, prob

    def clasificar(self, imagen, rect):
        probs = np.full(len(EMOCIONES), (1 - self.prob) / (len(EMOCIONES) - 1))
        probs[EMOCIONES.index(self.emocion)] = self.prob
        return probs


class ProcesadorTests(TestCase):
    def setUp(self):
        self.cuadro = np.random.default_rng(5).integers(0, 256, (240, 320, 3), dtype=np.uint8)

    def test_detecta_la_emocion_de_cada_rostro(self):
        proc = ProcesadorCuadros(DetectorFalso([Rostro(50, 50, 100, 100)]), EmocionesFalsas('sorpresa', 0.8))
        _, info = proc.procesar(self.cuadro)
        self.assertEqual(info['rostros'], 1)
        cara = info['caras'][0]
        self.assertEqual(cara['emocion'], 'sorpresa')
        self.assertEqual(cara['nombre'], 'Sorpresa')
        self.assertAlmostEqual(cara['confianza'], 80.0, places=1)
        self.assertEqual(info['emocion_principal'], 'Sorpresa')
        self.assertEqual(info['modo_emociones'], 'prueba')
        self.assertEqual(list(cara['probabilidades'])[0], 'sorpresa')  # ordenadas de mayor a menor

    def test_resumen_acumula_cuadros_y_se_reinicia(self):
        emociones = EmocionesFalsas('feliz')
        proc = ProcesadorCuadros(DetectorFalso([Rostro(50, 50, 100, 100)]), emociones)
        for _ in range(3):
            proc.procesar(self.cuadro)
        _, info = proc.procesar(self.cuadro)
        resumen = {r['emocion']: r for r in info['resumen']}
        self.assertEqual(resumen['feliz']['cuadros'], 4)
        self.assertEqual(resumen['feliz']['porcentaje'], 100.0)

        proc.reiniciar_resumen()
        emociones.emocion = 'tristeza'
        _, info = proc.procesar(self.cuadro)
        resumen = {r['emocion']: r for r in info['resumen']}
        self.assertEqual(resumen['feliz']['cuadros'], 0)
        self.assertEqual(resumen['tristeza']['cuadros'], 1)

    def test_sin_clasificador_no_falla(self):
        proc = ProcesadorCuadros(DetectorFalso([Rostro(50, 50, 100, 100)]))
        salida, info = proc.procesar(self.cuadro)
        self.assertEqual(info['caras'], [])
        self.assertIsNone(info['emocion_principal'])
        self.assertEqual(salida.shape, self.cuadro.shape)

    def test_salida_tiene_mismo_tamano(self):
        proc = ProcesadorCuadros(DetectorFalso([]), EmocionesFalsas())
        salida, info = proc.procesar(self.cuadro)
        self.assertEqual(salida.shape, self.cuadro.shape)
        self.assertEqual(info['rostros'], 0)

    def test_registro_guiado_guarda_muestras_y_termina(self):
        carpeta = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, carpeta, True)
        completadas = []
        proc = ProcesadorCuadros(DetectorFalso([Rostro(40, 40, 150, 150)]), EmocionesFalsas(),
                                 al_completar_registro=completadas.append)
        sesion = SesionRegistro(persona_id=1, nombre='Ana', carpeta=carpeta, objetivo=4)
        proc.iniciar_registro(sesion)
        for _ in range(12):
            proc.procesar(self.cuadro)

        self.assertTrue(sesion.terminada)
        self.assertEqual(sesion.guardadas, 4)
        self.assertEqual(completadas, [sesion])
        archivos = os.listdir(carpeta)
        self.assertEqual(len(archivos), 4)
        self.assertEqual(leer_imagen(os.path.join(carpeta, archivos[0])).shape, (200, 200))

    def test_registro_rechaza_cuando_hay_varios_rostros(self):
        carpeta = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, carpeta, True)
        proc = ProcesadorCuadros(DetectorFalso([Rostro(0, 0, 80, 80), Rostro(100, 100, 80, 80)]), EmocionesFalsas())
        sesion = SesionRegistro(persona_id=1, nombre='Ana', carpeta=carpeta, objetivo=3)
        proc.iniciar_registro(sesion)
        for _ in range(6):
            proc.procesar(self.cuadro)
        self.assertEqual(sesion.guardadas, 0)
        self.assertIn('más de un rostro', sesion.mensaje)

    def test_cancelar_registro(self):
        proc = ProcesadorCuadros(DetectorFalso([]), EmocionesFalsas())
        sesion = SesionRegistro(persona_id=1, nombre='Ana', carpeta=tempfile.gettempdir())
        proc.iniciar_registro(sesion)
        proc.cancelar_registro()
        self.assertTrue(sesion.cancelada and sesion.terminada)


class ServicioVideoTests(TestCase):
    """Usa un archivo de video sintético como fuente: no requiere webcam."""

    def setUp(self):
        import cv2
        self.carpeta = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.carpeta, True)
        self.video = os.path.join(self.carpeta, 'prueba.avi')
        escritor = cv2.VideoWriter(self.video, cv2.VideoWriter_fourcc(*'MJPG'), 30, (320, 240))
        for i in range(10):
            escritor.write(np.full((240, 320, 3), i * 20, np.uint8))
        escritor.release()

    def _servicio(self, fuente):
        from asistencia.vision.camara import ServicioVideo
        proc = ProcesadorCuadros(DetectorFalso([]), EmocionesFalsas())
        servicio = ServicioVideo(proc, fuente=fuente, ancho=320, alto=240, inactividad=5)
        self.addCleanup(servicio.detener)
        return servicio

    def test_transmite_cuadros_jpeg_desde_un_video(self):
        servicio = self._servicio(self.video)
        stream = servicio.generar_mjpeg()
        parte = next(stream)
        self.assertTrue(parte.startswith(b'--frame\r\nContent-Type: image/jpeg'))
        self.assertIn(b'\xff\xd8', parte)  # marcador de inicio JPEG
        self.assertTrue(servicio.camara_ok)
        self.assertEqual(servicio.estado()['rostros'], 0)

    def test_fuente_inexistente_publica_aviso_sin_caerse(self):
        servicio = self._servicio(os.path.join(self.carpeta, 'no_existe.mp4'))
        _, jpeg = servicio.esperar_cuadro(0, timeout=5)
        self.assertIsNotNone(jpeg)
        self.assertFalse(servicio.camara_ok)
        self.assertIn('No se pudo abrir', servicio.error)

    def test_detener_cierra_los_streams_y_no_reabre_la_camara(self):
        servicio = self._servicio(self.video)
        stream = servicio.generar_mjpeg()
        next(stream)
        servicio.detener()
        self.assertEqual(list(stream), [])
        self.assertFalse(servicio.en_ejecucion)


class EmocionesTests(TestCase):
    def test_softmax_suma_uno_y_conserva_el_orden(self):
        p = softmax([2.0, 1.0, 0.1])
        self.assertAlmostEqual(p.sum(), 1.0)
        self.assertEqual(int(np.argmax(p)), 0)

    def test_preparar_rostro_devuelve_gris_64x64_aunque_toque_el_borde(self):
        img = np.random.default_rng(2).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        cara = preparar_rostro(img, (280, 200, 60, 60))
        self.assertEqual(cara.shape, (64, 64))
        self.assertIsNone(preparar_rostro(img, (400, 400, 10, 10)))

    def test_sin_archivo_de_modelo_usa_modo_basico(self):
        carpeta = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, carpeta, True)
        clasificador = ClasificadorEmociones(os.path.join(carpeta, 'no_existe.onnx'), descargar=False)
        self.assertFalse(clasificador.completo)
        self.assertIn('basico', clasificador.modo)
        probs = clasificador.clasificar(np.full((200, 200, 3), 128, np.uint8), (20, 20, 160, 160))
        self.assertAlmostEqual(probs.sum(), 1.0)
        self.assertEqual(EMOCIONES[int(np.argmax(probs))], 'neutral')

    def test_modelo_fer_plus_si_esta_descargado(self):
        from django.conf import settings
        ruta = settings.VISION['EMOTION_MODEL_PATH']
        if not os.path.exists(ruta):
            self.skipTest('Modelo FER+ no descargado (python manage.py descargar_modelo_emociones).')
        clasificador = ClasificadorEmociones(ruta, descargar=False)
        self.assertTrue(clasificador.completo)
        probs = clasificador.clasificar(textura(4)[:, :, None].repeat(3, 2), (0, 0, 200, 200))
        self.assertEqual(probs.shape, (len(EMOCIONES),))
        self.assertAlmostEqual(probs.sum(), 1.0, places=5)

    def test_suavizador_promedia_el_mismo_rostro_entre_cuadros(self):
        suav = SuavizadorEmociones(alfa=0.5)
        a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        suav.actualizar([(0, 0, 100, 100)], [a])
        (resultado,) = suav.actualizar([(5, 5, 100, 100)], [b])
        np.testing.assert_allclose(resultado, [0.5, 0.5])

    def test_suavizador_no_mezcla_rostros_lejanos(self):
        suav = SuavizadorEmociones(alfa=0.5)
        a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        suav.actualizar([(0, 0, 50, 50)], [a])
        (resultado,) = suav.actualizar([(250, 150, 50, 50)], [b])
        np.testing.assert_allclose(resultado, b)


class UtilidadesTests(TestCase):
    def test_interpretar_fuente(self):
        self.assertEqual(interpretar_fuente('0'), 0)
        self.assertEqual(interpretar_fuente(' 1 '), 1)
        self.assertEqual(interpretar_fuente('video.mp4'), 'video.mp4')

    def test_texto_ascii_quita_tildes(self):
        self.assertEqual(texto_ascii('José Núñez'), 'Jose Nunez')

    def test_cuadro_mensaje(self):
        self.assertEqual(cuadro_mensaje(320, 240, 'Hola').shape, (240, 320, 3))
