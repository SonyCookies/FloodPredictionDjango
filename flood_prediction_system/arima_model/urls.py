from django.contrib import admin
from django.urls import path 
from . import views

urlpatterns = [
    path('/', views.run_arima_view, name='run_arima_view'),
]
