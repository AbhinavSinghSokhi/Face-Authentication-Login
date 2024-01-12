# from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
 path("", views.index, name="indexpage"),
 path("signupPage", views.signup, name="signupPage"),
 path("loginpage",views.login, name="loginpage"),
#  path("facecapture", views.capture_video, name="facecapture"),
 path('video_feed', views.video_feed, name='video_feed'),
 path('open_camera', views.open_camera, name='open_camera'),
]
