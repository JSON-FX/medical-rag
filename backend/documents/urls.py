from django.urls import path

from . import views

urlpatterns = [
    path("documents/", views.documents_collection, name="documents"),
    path("documents/<int:document_id>/", views.document_detail, name="document-detail"),
]
