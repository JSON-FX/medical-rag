from django.contrib import admin

from .models import ChatMessage, ChatSession


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "created_at")
    search_fields = ("title", "id")


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    # was_declined and decline_reason are the point: the spec's claim that
    # gate decisions are inspectable in the admin lives or dies on these two
    # columns being visible in the list view, not buried in the detail page.
    list_display = (
        "id",
        "session",
        "role",
        "was_declined",
        "decline_reason",
        "truncated",
        "created_at",
    )
    list_filter = ("role", "was_declined", "decline_reason", "truncated")
    search_fields = ("content",)
    autocomplete_fields = ("session",)
    readonly_fields = ("created_at",)
