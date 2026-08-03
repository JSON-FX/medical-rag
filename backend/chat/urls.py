from django.urls import path

from . import views

urlpatterns = [
    path("health/", views.health, name="health"),
    path("chat/", views.chat, name="chat"),
    path("chat/sessions/<uuid:session_id>/messages/", views.session_messages, name="session-messages"),
]
