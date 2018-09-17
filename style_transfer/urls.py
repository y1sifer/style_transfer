from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

app_name = 'style_transfer'

urlpatterns = [
    path('', views.index, name='index'),
    path('show', views.showImg, name='show_img'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)