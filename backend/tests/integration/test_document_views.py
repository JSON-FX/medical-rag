import json

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client

from documents.models import Chunk, Document
from tests.fixtures.make_fixture_pdf import make_pdf

pytestmark = pytest.mark.django_db


@pytest.fixture
def client():
    return Client()


@pytest.fixture(autouse=True)
def isolated_services(chroma_store, fake_embedder, monkeypatch):
    """Point the views at a throwaway Chroma and the shared fake embedder."""
    import documents.services as services

    monkeypatch.setattr(services, "get_store", lambda: chroma_store)
    monkeypatch.setattr(services, "get_embedder", lambda: fake_embedder)
    return chroma_store


def _pdf_upload(tmp_path, name="mono.pdf"):
    path = make_pdf(tmp_path / name, ["Metformin adult dose is 500mg daily."])
    return SimpleUploadedFile(name, path.read_bytes(), content_type="application/pdf")


def test_upload_returns_201_and_ready_status(client, tmp_path):
    resp = client.post("/api/documents/", {"file": _pdf_upload(tmp_path)})
    body = json.loads(resp.content)
    assert resp.status_code == 201
    assert body["status"] == "ready"
    assert body["chunk_count"] > 0
    assert body["page_count"] == 1
    assert body["title"] == "mono.pdf"


def test_upload_rejects_non_pdf_with_400(client):
    bad = SimpleUploadedFile("notes.txt", b"hello", content_type="text/plain")
    resp = client.post("/api/documents/", {"file": bad})
    assert resp.status_code == 400
    assert "pdf" in json.loads(resp.content)["error"].lower()


def test_upload_rejects_oversized_file_with_413(client, settings):
    big = SimpleUploadedFile("big.pdf", b"%PDF-1.4\n" + b"0" * (16 * 1024 * 1024), "application/pdf")
    resp = client.post("/api/documents/", {"file": big})
    assert resp.status_code == 413
    assert Document.objects.count() == 0


def test_upload_with_no_file_returns_400(client):
    assert client.post("/api/documents/", {}).status_code == 400


def test_list_returns_documents_newest_first(client, tmp_path):
    client.post("/api/documents/", {"file": _pdf_upload(tmp_path, "a.pdf")})
    client.post("/api/documents/", {"file": _pdf_upload(tmp_path, "b.pdf")})
    body = json.loads(client.get("/api/documents/").content)
    assert [d["title"] for d in body] == ["b.pdf", "a.pdf"]
    assert set(body[0]) >= {"id", "title", "status", "page_count", "chunk_count", "uploaded_at"}


def test_delete_removes_document_chunks_and_vectors(client, tmp_path, isolated_services):
    created = json.loads(client.post("/api/documents/", {"file": _pdf_upload(tmp_path)}).content)
    assert isolated_services.count() > 0

    resp = client.delete(f"/api/documents/{created['id']}/")
    assert resp.status_code == 204
    assert Document.objects.count() == 0
    assert Chunk.objects.count() == 0
    assert isolated_services.count() == 0


def test_delete_missing_document_returns_404(client):
    assert client.delete("/api/documents/999/").status_code == 404


def test_service_initialisation_failure_creates_no_stranded_document(tmp_path, monkeypatch):
    """If Chroma cannot be constructed, no Document row should exist at all —
    a row created first would sit in `processing` with nothing to advance it."""
    import documents.services as services

    def boom():
        raise RuntimeError("could not connect to tenant default_tenant")

    monkeypatch.setattr(services, "get_store", boom)
    resp = Client().post("/api/documents/", {"file": _pdf_upload(tmp_path)})
    assert resp.status_code == 503
    assert Document.objects.count() == 0, "document stranded in processing"


def test_error_message_does_not_leak_absolute_server_paths(client, tmp_path, settings):
    """error_message reaches the API client; it must not carry the server's
    filesystem layout."""
    doc = Document.objects.create(
        title="x.pdf",
        status="failed",
        error_message=f"[Errno 2] No such file or directory: '{settings.MEDIA_ROOT}/documents/x.pdf'",
    )
    body = json.loads(client.get("/api/documents/").content)
    entry = next(d for d in body if d["id"] == doc.id)
    assert str(settings.MEDIA_ROOT) not in entry["error_message"]
    assert "<media>" in entry["error_message"]


def test_get_store_is_safe_under_concurrent_first_construction(tmp_path, settings):
    """Sync views run in uvicorn's threadpool; concurrent first requests must
    not race chromadb's tenant initialisation."""
    import importlib
    from concurrent.futures import ThreadPoolExecutor
    import documents.services as services

    settings.CHROMA_PATH = tmp_path / "race"
    importlib.reload(services)

    with ThreadPoolExecutor(max_workers=8) as pool:
        stores = [f.result() for f in [pool.submit(services.get_store) for _ in range(8)]]

    assert len({id(s) for s in stores}) == 1, "more than one store constructed"
    importlib.reload(services)
