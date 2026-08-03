"""Detects and repairs drift between SQLite chunks and Chroma vectors.

Two stores cannot share a transaction (spec 10), so a crash mid-ingest can
leave either side ahead. This is the repair path.
"""
from django.core.management.base import BaseCommand

from documents import services
from documents.models import Chunk, Document

REUPLOAD_MESSAGE = "Vectors are missing for this document. Please delete and re-upload it."


class Command(BaseCommand):
    help = "Report (and optionally repair) drift between SQLite chunks and Chroma vectors."

    def add_arguments(self, parser):
        parser.add_argument("--fix", action="store_true", help="Repair the drift, not just report it.")

    def handle(self, *args, **options):
        store = services.get_store()
        vector_ids = store.all_ids()
        chunks = {
            c.vector_id: c
            for c in Chunk.objects.only("id", "document_id", "chunk_index")
        }

        missing_vectors = set(chunks) - vector_ids     # chunk row, no vector
        orphan_vectors = vector_ids - set(chunks)      # vector, no chunk row

        if not missing_vectors and not orphan_vectors:
            self.stdout.write(self.style.SUCCESS("No drift: SQLite and Chroma agree."))
            return

        self.stdout.write(f"{len(missing_vectors)} chunk(s) missing a vector")
        self.stdout.write(f"{len(orphan_vectors)} vector(s) orphaned with no chunk row")

        if not options["fix"]:
            self.stdout.write(self.style.WARNING("Run with --fix to repair."))
            return

        # Delete exactly the orphaned ids — never a whole document. Deleting by
        # document_id would take that document's valid vectors with it, turning
        # a harmless orphan into an unsearchable `ready` document: strictly
        # worse than the drift being repaired.
        if orphan_vectors:
            store.delete_ids(sorted(orphan_vectors))
            still_present = store.all_ids() & orphan_vectors
            self.stdout.write(
                f"removed {len(orphan_vectors) - len(still_present)} orphan vector(s)"
            )
            if still_present:
                self.stdout.write(
                    self.style.WARNING(f"{len(still_present)} orphan(s) could not be removed")
                )

        # Runs independently of the orphan cleanup above. A document that reads
        # `ready` while being unsearchable is the more damaging drift, and must
        # not be left unmarked because orphan removal had a problem.
        affected = {chunks[v].document_id for v in missing_vectors}
        if affected:
            Document.objects.filter(id__in=affected).update(
                status="failed", error_message=REUPLOAD_MESSAGE
            )
            self.stdout.write(f"marked {len(affected)} document(s) failed")

        self.stdout.write(self.style.SUCCESS("Repair complete."))
