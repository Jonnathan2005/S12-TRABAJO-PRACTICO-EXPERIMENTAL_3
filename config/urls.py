from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

admin.site.site_header = 'VisionAsistencia - Administración'
admin.site.site_title = 'VisionAsistencia'

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('asistencia.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
