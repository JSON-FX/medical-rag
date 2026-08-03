from django.contrib import admin

from .models import Chunk, Document


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("title", "status", "page_count", "chunk_count", "uploaded_at")
    list_filter = ("status",)
    search_fields = ("title",)
    readonly_fields = ("uploaded_at",)


@admin.register(Chunk)
class ChunkAdmin(admin.ModelAdmin):
    list_display = ("document", "chunk_index", "page_number")
    search_fields = ("text",)
    autocomplete_fields = ("document",)
