from django.db import models


class Document(models.Model):
    STATUS_CHOICES = [
        ("processing", "Processing"),
        ("ready", "Ready"),
        ("failed", "Failed"),
    ]
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to="documents/")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="processing")
    page_count = models.IntegerField(null=True, blank=True)
    chunk_count = models.IntegerField(default=0)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    error_message = models.TextField(blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self) -> str:
        return f"{self.title} ({self.status})"


class Chunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    chunk_index = models.IntegerField()
    page_number = models.IntegerField()
    text = models.TextField()

    class Meta:
        unique_together = [("document", "chunk_index")]
        indexes = [models.Index(fields=["document", "chunk_index"])]
        ordering = ["document_id", "chunk_index"]

    @property
    def vector_id(self) -> str:
        """Deterministic id shared with Chroma (spec 10)."""
        return f"{self.document_id}_{self.chunk_index}"
