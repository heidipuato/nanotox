from django.urls import path
from . import views

urlpatterns = [
    path('',views.home, name='home'),
    path('test/',views.test, name='test'),
    path('predict/',views.predict, name='predict'),
    path('about/',views.about, name='about'),
]