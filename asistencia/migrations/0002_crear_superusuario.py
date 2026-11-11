"""
Crea el superusuario por defecto para el panel /admin al ejecutar ``migrate``.

Credenciales por defecto: usuario ``admin`` / contraseña ``Admin2026!``.
Se pueden cambiar con las variables de entorno ADMIN_USERNAME, ADMIN_PASSWORD
y ADMIN_EMAIL en ``.env``. Si el usuario ya existe, no se modifica.
"""

import os

from django.contrib.auth.hashers import make_password
from django.db import migrations


def crear_superusuario(apps, schema_editor):
    User = apps.get_model('auth', 'User')
    usuario = os.getenv('ADMIN_USERNAME', 'admin')
    if User.objects.filter(username=usuario).exists():
        return
    User.objects.create(
        username=usuario,
        email=os.getenv('ADMIN_EMAIL', 'admin@localhost'),
        password=make_password(os.getenv('ADMIN_PASSWORD', 'Admin2026!')),
        is_staff=True,
        is_superuser=True,
        is_active=True,
    )


def eliminar_superusuario(apps, schema_editor):
    User = apps.get_model('auth', 'User')
    User.objects.filter(username=os.getenv('ADMIN_USERNAME', 'admin')).delete()


class Migration(migrations.Migration):

    dependencies = [
        ('asistencia', '0001_initial'),
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.RunPython(crear_superusuario, eliminar_superusuario),
    ]
