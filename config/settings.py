"""
Configuración de Django para el proyecto VisionAsistencia.

Los valores sensibles (credenciales de PostgreSQL, SECRET_KEY, cámara) se leen
desde variables de entorno o desde un archivo ``.env`` en la raíz del proyecto.
Ver ``.env.example``.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')


def env_bool(nombre, defecto=False):
    return os.getenv(nombre, str(defecto)).strip().lower() in ('1', 'true', 'si', 'sí', 'yes', 'on')


SECRET_KEY = os.getenv(
    'DJANGO_SECRET_KEY',
    'django-insecure-solo-para-desarrollo-cambiar-en-produccion',
)
DEBUG = env_bool('DJANGO_DEBUG', True)
ALLOWED_HOSTS = [h.strip() for h in os.getenv('DJANGO_ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',') if h.strip()]


INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'asistencia.apps.AsistenciaConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'config.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'config.wsgi.application'


# Base de datos: PostgreSQL (requisito de la práctica).
# DB_ENGINE=sqlite solo se usa como alternativa para ejecutar pruebas sin servidor.
if os.getenv('DB_ENGINE', 'postgresql').lower() == 'sqlite':
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.postgresql',
            'NAME': os.getenv('DB_NAME', 'vision_asistencia'),
            'USER': os.getenv('DB_USER', 'postgres'),
            'PASSWORD': os.getenv('DB_PASSWORD', ''),
            'HOST': os.getenv('DB_HOST', 'localhost'),
            'PORT': os.getenv('DB_PORT', '5432'),
        }
    }


AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator'},
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator'},
    {'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator'},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]


LANGUAGE_CODE = 'es-ec'
TIME_ZONE = 'America/Guayaquil'
USE_I18N = True
USE_TZ = True

STATIC_URL = 'static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = 'media/'
MEDIA_ROOT = BASE_DIR / 'media'

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'


# ---------------------------------------------------------------------------
# Parámetros de visión por computadora (OpenCV)
# ---------------------------------------------------------------------------
VISION = {
    # Índice de la cámara (0, 1, ...) o ruta/URL a un video o stream RTSP/HTTP.
    'CAMERA_SOURCE': os.getenv('CAMERA_SOURCE', '0'),
    'FRAME_WIDTH': int(os.getenv('FRAME_WIDTH', '640')),
    'FRAME_HEIGHT': int(os.getenv('FRAME_HEIGHT', '480')),
    'JPEG_QUALITY': int(os.getenv('JPEG_QUALITY', '80')),
    # Distancia LBPH máxima para aceptar un reconocimiento (menor = más estricto).
    # Calibrado empíricamente: aciertos reales ~20-50, falsos positivos de Haar ~70-85.
    'RECOGNITION_THRESHOLD': float(os.getenv('RECOGNITION_THRESHOLD', '65')),
    # Muestras de rostro que se capturan por persona en el registro guiado.
    'ENROLL_SAMPLES': int(os.getenv('ENROLL_SAMPLES', '30')),
    # Segundos sin espectadores tras los cuales se libera la cámara.
    'IDLE_TIMEOUT': int(os.getenv('IDLE_TIMEOUT', '30')),
    'SAMPLES_DIR': MEDIA_ROOT / 'rostros',
    'MODEL_PATH': MEDIA_ROOT / 'modelo' / 'lbph_rostros.yml',
    # Red FER+ (ONNX) para detectar emociones en la cámara. Si no existe se descarga
    # automáticamente la primera vez (~34 MB); sin internet se usa un modo básico.
    'EMOTION_MODEL_PATH': Path(os.getenv('EMOTION_MODEL_PATH', MEDIA_ROOT / 'modelo' / 'emotion-ferplus-8.onnx')),
    'EMOTION_MODEL_DOWNLOAD': os.getenv('EMOTION_MODEL_DOWNLOAD', 'True').lower() in ('1', 'true', 'yes', 'si'),
}

# Hora límite para considerar una asistencia como "puntual" (HH:MM, hora local).
HORA_LIMITE_PUNTUAL = os.getenv('HORA_LIMITE_PUNTUAL', '08:15')
