# Casos de prueba funcionales – VisionAsistencia

Estas pruebas se ejecutan manualmente sobre el sistema en funcionamiento, como pruebas de usuario. Las pruebas automatizadas de desarrollo están en `asistencia/tests/` y se ejecutan con `python manage.py test asistencia`.

**Preparación:** servidor en ejecución (`python manage.py runserver`), cámara conectada y PostgreSQL activo.

| ID | Caso | Pasos | Resultado esperado | Resultado obtenido |
|---|---|---|---|---|
| CP-01 | Transmisión en tiempo real | Abrir **Emociones en vivo** | El video aparece en menos de 3 s. El panel lateral muestra FPS mayores a 10 y «Cámara: conectada». | |
| CP-02 | Detección de un rostro | Ubicarse frente a la cámara | Aparece un recuadro alrededor del rostro con su emoción y el contador «Rostros» muestra 1. | |
| CP-03 | Detección de varios rostros | Dos o más personas frente a la cámara | Cada rostro tiene su propio recuadro y el contador coincide con el número de personas. | |
| CP-04 | Registrar persona | Personas → Nueva persona → completar datos válidos | La persona queda creada y se abre su ficha. | |
| CP-05 | Validación de cédula | Ingresar una cédula con letras o con menos de 10 dígitos | Se muestra el error «La cédula debe tener exactamente 10 dígitos» y no se guarda. | |
| CP-06 | Cédula duplicada | Crear otra persona con una cédula ya registrada | El formulario muestra un error de cédula repetida. | |
| CP-07 | Registro guiado de rostro | En la ficha, pulsar «Capturar rostro con la cámara» | La barra avanza hasta 30/30, aparece «¡Listo!… modelo entrenado» y se ven las miniaturas de las muestras. | |
| CP-08 | Registro con dos personas | Durante la captura, que aparezcan dos rostros | La captura se pausa con el mensaje «Hay más de un rostro…». | |
| CP-09 | Descarte de imágenes borrosas | Moverse rápido durante la captura | Aumentan las muestras descartadas y aparece el mensaje «Imagen borrosa descartada». | |
| CP-10 | Emoción: feliz | Sonreír frente a la cámara | El recuadro cambia al color de «Feliz», la etiqueta muestra «Feliz» con su porcentaje y las barras bajo el rostro muestran las 3 emociones más probables. | |
| CP-11 | Emoción: sorpresa / enojo / tristeza | Abrir la boca y levantar las cejas; luego fruncir el ceño; luego bajar las comisuras | La etiqueta cambia a «Sorpresa», «Enojo» y «Tristeza» respectivamente en menos de 1 s. | |
| CP-12 | Varias personas con emociones distintas | Dos personas: una sonríe y otra seria | Cada rostro muestra su propia emoción; «Emoción principal» corresponde al rostro más grande. | |
| CP-13 | Resumen de la sesión | Permanecer frente a la cámara variando gestos y pulsar «Reiniciar» | El resumen muestra el porcentaje de cuadros por emoción; al reiniciar vuelve a «Esperando rostros…». | |
| CP-14 | Modelo de emociones no disponible | Borrar `media/modelo/emotion-ferplus-8.onnx` y poner `EMOTION_MODEL_DOWNLOAD=False` | La página indica «Modelo: basico (detector de sonrisa)» y solo distingue Feliz/Neutral, sin errores. | |
| CP-15 | Iluminación baja | Cubrir parcialmente la cámara | El indicador de iluminación muestra «baja». | |
| CP-16 | Detener y reanudar cámara | Pulsar «Detener cámara» y luego «Iniciar cámara» | La cámara se libera (se apaga su luz) y el video se reanuda al iniciar. | |
| CP-17 | Cámara desconectada | Desconectar la webcam o usar `CAMERA_SOURCE=9` | Aparece el aviso «Cámara no disponible – Reintentando» y la aplicación sigue funcionando. | |
| CP-18 | Subir fotografías | Subir fotos de un rostro en la ficha | Se agregan las muestras y el modelo se reentrena. | |
| CP-19 | Foto sin rostro | Subir una foto de un paisaje | Aparece el aviso «No se detectó un rostro claro en: …». | |
| CP-20 | Eliminar persona | Ficha → Eliminar → confirmar | Se borran la persona, sus muestras y sus asistencias, y el modelo se reentrena. | |
| CP-21 | Persistencia | Reiniciar el servidor | Las personas y asistencias siguen en PostgreSQL y los modelos (LBPH y FER+) se recargan desde disco sin volver a descargarse. | |
