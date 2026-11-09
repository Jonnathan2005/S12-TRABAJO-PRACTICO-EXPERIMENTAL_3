import psycopg
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from psycopg import sql


class Command(BaseCommand):
    help = 'Crea en PostgreSQL la base de datos configurada en .env (DB_NAME) si aún no existe.'

    def handle(self, *args, **options):
        db = settings.DATABASES['default']
        if 'postgresql' not in db['ENGINE']:
            raise CommandError('Este comando solo aplica cuando DB_ENGINE es postgresql.')
        try:
            conexion = psycopg.connect(dbname='postgres', user=db['USER'], password=db['PASSWORD'],
                                       host=db['HOST'], port=db['PORT'], autocommit=True)
        except psycopg.OperationalError as exc:
            raise CommandError(f'No se pudo conectar a PostgreSQL. Revise DB_USER/DB_PASSWORD en .env.\n{exc}')

        with conexion:
            existe = conexion.execute('SELECT 1 FROM pg_database WHERE datname = %s', [db['NAME']]).fetchone()
            if existe:
                self.stdout.write(f'La base de datos "{db["NAME"]}" ya existe.')
                return
            conexion.execute(sql.SQL("CREATE DATABASE {} ENCODING 'UTF8'").format(sql.Identifier(db['NAME'])))
        self.stdout.write(self.style.SUCCESS(f'Base de datos "{db["NAME"]}" creada.'))
