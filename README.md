# VisionAsistencia

**Control de asistencia por reconocimiento facial en tiempo real** con Django, OpenCV y PostgreSQL.

Práctica N.º 3 de *Construcción de Software* (UNEMI, Software 2019, A2).
Temática elegida: **Detección de rostros para control de asistencia**.

La cámara transmite video en vivo al navegador y se dedica a la **detección de emociones mediante expresiones faciales**: en cada cuadro OpenCV **detecta** los rostros (clasificador Haar), los recorta en gris a 64×64 y **clasifica su expresión** con la red neuronal FER+ ejecutada con `cv2.dnn` (neutral, feliz, sorpresa, tristeza, enojo, asco, miedo y desprecio). La emoción y su porcentaje se dibujan sobre el video y el panel lateral muestra el detalle por rostro y un resumen de la sesión.

---

## Integrantes

| Nombre | Rol |
|---|---|
| _(completar)_ | Backend Django / Base de datos |
| _(completar)_ | Módulo OpenCV (detección y reconocimiento) |
| _(completar)_ | Interfaz web |
| _(completar)_ | Pruebas y documentación |
| _(completar)_ | Integración y repositorio |

**Video demostrativo:** _(enlace de Google Drive)_

---

## Funcionalidades

| Módulo | Qué hace | Técnica OpenCV |
|---|---|---|
| **Detección de emociones** | En la cámara en vivo, clasifica la expresión facial de cada rostro en 8 emociones y muestra las 3 más probables. Las probabilidades se suavizan entre cuadros para que la etiqueta no parpadee, y se acumula un resumen de la sesión. Si el modelo no está disponible usa un modo básico (detector de sonrisa: feliz/neutral). | `cv2.dnn.readNetFromONNX` (FER+), `blobFromImage`, Haar |
| **Captura y transmisión** | Video de la webcam en vivo en el navegador (MJPEG). Un solo hilo atiende a todos los clientes, se reconecta si se pierde la cámara y la libera tras 30 s sin espectadores. | `cv2.VideoCapture`, `cv2.imencode` |
| **Detección** | Rostros frontales y ojos dentro de cada rostro. La detección corre sobre una copia reducida del cuadro para mantener buena velocidad (FPS). | `CascadeClassifier` (Haar) |
| **Procesamiento** | Escala de grises, CLAHE, recorte y normalización, medición de nitidez (varianza del Laplaciano) y de brillo. | `cvtColor`, `createCLAHE`, `resize`, `Laplacian` |
| **Reconocimiento** | Identifica a cada persona registrada y muestra el porcentaje de similitud. Los rostros que no se parecen lo suficiente se marcan como *Desconocido*. | `cv2.face.LBPHFaceRecognizer` |
| **Registro guiado de rostros** | Captura 30 muestras desde la cámara con una barra de progreso, descarta las imágenes borrosas y rechaza la captura si aparece más de un rostro. Al terminar, reentrena el modelo automáticamente. | Laplaciano, Haar, LBPH |
| **Carga de fotos** | Permite registrar el rostro subiendo fotografías: el rostro se detecta, se recorta y se guarda también su versión espejada como muestra extra. | `imdecode`, `flip` |
| **Resultados en la interfaz** | Recuadros, nombres y estado dibujados sobre el video; indicadores del día, lista de asistencias en vivo, historial filtrable y panel de administración de Django. | `rectangle`, `putText`, `circle` |

---

## Arquitectura

```
Navegador ──HTTP──► Django (views.py) ──► services.py ──► PostgreSQL
    ▲                                        │
    │  MJPEG /stream/                        ▼
    └──────────── ServicioVideo (hilo) ─► ProcesadorCuadros
                   cv2.VideoCapture         ├─ DetectorRostros   (Haar)
                                            ├─ procesamiento     (gris, CLAHE, nitidez)
                                            ├─ ClasificadorEmociones (FER+, cv2.dnn)
                                            └─ dibujo            (anotaciones)
```

* **`asistencia/vision/`** es un paquete **independiente de Django**: solo usa OpenCV y NumPy. Por eso puede reutilizarse en otro proyecto y probarse de forma aislada (reutilización de código y de diseño).
* **`asistencia/services.py`** conecta la visión con Django. El registro de asistencia llega al procesador como *callback* (inyección de dependencias), así que la tubería no sabe que existe una base de datos.

### Estructura del repositorio

```
├── config/                  # Proyecto Django (settings, urls)
├── asistencia/              # Aplicación principal
│   ├── vision/              # Módulo OpenCV reutilizable
│   │   ├── camara.py        # Captura en hilo + stream MJPEG
│   │   ├── detector.py      # Detección Haar de rostros y ojos
│   │   ├── procesamiento.py # Gris, CLAHE, nitidez, brillo
│   │   ├── emociones.py     # Emociones por expresión facial (FER+ con cv2.dnn)
│   │   ├── reconocedor.py   # LBPH (entrenar, predecir, guardar, cargar)
│   │   ├── procesador.py    # Tubería por cuadro + registro guiado
│   │   └── dibujo.py        # Anotaciones sobre el video
│   ├── management/commands/ # crear_basedatos, entrenar_modelo
│   ├── templates/ static/   # Interfaz web (HTML, CSS, JS sin dependencias)
│   ├── tests/               # 40 pruebas automatizadas
│   ├── models.py            # Persona, Asistencia, EntrenamientoModelo
│   ├── services.py  views.py  forms.py  urls.py  admin.py
├── docs/CASOS_DE_PRUEBA.md  # Casos de prueba funcionales (de usuario)
├── requirements.txt
├── .env.example
└── manage.py
```

---

## Requisitos

* Python 3.11 o superior (probado con 3.13)
* PostgreSQL 14 o superior (probado con 18)
* Una cámara web

> **Importante:** se usa `opencv-contrib-python` **4.x**. Incluye el módulo `cv2.face` (LBPH) y los clasificadores Haar, que OpenCV 5 ya no trae. No instale también `opencv-python`, porque ambos paquetes entran en conflicto.

## Instalación (Windows)

```powershell
# 1. Clonar el repositorio
git clone https://github.com/<usuario>/<repositorio>.git
cd <repositorio>

# 2. Crear y activar un entorno virtual (no se sube al repositorio)
python -m venv venv
venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
copy .env.example .env
#    -> edite .env y escriba la contraseña de su usuario de PostgreSQL (DB_PASSWORD)

# 5. Crear la base de datos y las tablas
python manage.py crear_basedatos
python manage.py migrate

# 6. (Opcional) usuario para el panel /admin
python manage.py createsuperuser

# 7. Ejecutar
python manage.py runserver
```

Abra **http://127.0.0.1:8000/** en el navegador.

En Linux o macOS se activa el entorno con `source venv/bin/activate` y se copia el archivo con `cp .env.example .env`.

## Uso

1. **Personas → Nueva persona:** registre nombres, apellidos, cédula y curso.
2. En la ficha de la persona, pulse **Capturar rostro con la cámara**. Mire a la cámara y mueva ligeramente la cabeza hasta completar las 30 muestras. También puede **subir fotografías**. Al terminar, el modelo se entrena automáticamente.
3. Abra **Emociones en vivo**. Cada rostro frente a la cámara se enmarca con el color de su emoción, junto con el porcentaje y las tres emociones más probables. El botón **Reiniciar** pone en cero el resumen de la sesión. La primera vez se descarga el modelo FER+ (~34 MB); también puede descargarlo antes con `python manage.py descargar_modelo_emociones`.
4. Consulte **Panel** y **Asistencias**, y filtre por fecha, persona o estado.

### Configuración (`.env`)

| Variable | Por defecto | Descripción |
|---|---|---|
| `CAMERA_SOURCE` | `0` | Índice de cámara, ruta a un video o URL RTSP/HTTP |
| `RECOGNITION_THRESHOLD` | `65` | Distancia LBPH máxima aceptada (menor = más estricto) |
| `EMOTION_MODEL_DOWNLOAD` | `True` | Descarga automática del modelo FER+ si no existe en `media/modelo/` |
| `ENROLL_SAMPLES` | `30` | Muestras por persona en el registro guiado |
| `HORA_LIMITE_PUNTUAL` | `08:15` | Hasta esta hora la asistencia es *puntual* |
| `IDLE_TIMEOUT` | `30` | Segundos sin espectadores antes de liberar la cámara |

Si quiere hacer una demostración sin webcam, use `CAMERA_SOURCE=ruta/a/video.mp4`: el video se reproduce en bucle.

## Pruebas

```powershell
python manage.py test asistencia
```

El proyecto incluye **48 pruebas automatizadas**:

* **`tests/test_vision.py`** (sin base de datos): procesamiento de imagen, escalado de coordenadas del detector, servicio de video con un archivo sintético (transmisión MJPEG, fuente inexistente, detener sin reabrir la cámara), entrenamiento, predicción y persistencia del LBPH, detección de emociones (clasificación, resumen de sesión, suavizado entre cuadros, modo básico sin modelo y red FER+), registro guiado (descarte con varios rostros, cancelación) y utilidades.
* **`tests/test_app.py`**: restricción de una asistencia por día, cálculo de puntual/atraso, entrenamiento con muestras reales en disco, vistas, API JSON y validación de cédula y de archivos.

Las pruebas usan carpetas temporales y nunca abren la cámara. Django crea una base de datos de prueba en PostgreSQL, por lo que el usuario configurado necesita permiso `CREATEDB`. Si quiere ejecutarlas sin PostgreSQL, puede hacerlo con SQLite:

```powershell
$env:DB_ENGINE="sqlite"; python manage.py test asistencia
```

Los casos de prueba funcionales (pruebas de usuario) están en [docs/CASOS_DE_PRUEBA.md](docs/CASOS_DE_PRUEBA.md).

## Comandos útiles

| Comando | Descripción |
|---|---|
| `python manage.py crear_basedatos` | Crea en PostgreSQL la base de datos indicada en `.env` |
| `python manage.py descargar_modelo_emociones` | Descarga la red FER+ para la detección de emociones |
| `python manage.py entrenar_modelo` | Reentrena el reconocedor con todas las muestras |
| `python manage.py test asistencia` | Ejecuta las pruebas automatizadas |

## Solución de problemas

| Problema | Solución |
|---|---|
| «Cámara no disponible» | Cierre otras aplicaciones que estén usando la cámara (Teams, Zoom). Si tiene cámara externa, pruebe `CAMERA_SOURCE=1`. |
| `module 'cv2' has no attribute 'face'` | Ejecute `pip uninstall opencv-python opencv-contrib-python` y luego `pip install -r requirements.txt`. |
| `password authentication failed` | Revise `DB_USER` y `DB_PASSWORD` en `.env`. |
| No reconoce o confunde personas | Registre más muestras con buena iluminación y ajuste `RECOGNITION_THRESHOLD`: bájelo si confunde personas y súbalo si no reconoce a nadie. |

## Tecnologías

Python · Django 6 · OpenCV 4 (contrib) · NumPy · PostgreSQL · psycopg 3 · HTML/CSS/JavaScript nativo
