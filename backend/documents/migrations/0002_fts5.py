from django.db import migrations

CREATE = [
    """
    CREATE VIRTUAL TABLE chunk_fts USING fts5(
        text,
        content='documents_chunk',
        content_rowid='id',
        tokenize='porter unicode61'
    );
    """,
    """
    CREATE TRIGGER chunk_fts_ai AFTER INSERT ON documents_chunk BEGIN
        INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text);
    END;
    """,
    """
    CREATE TRIGGER chunk_fts_ad AFTER DELETE ON documents_chunk BEGIN
        INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES('delete', old.id, old.text);
    END;
    """,
    """
    CREATE TRIGGER chunk_fts_au AFTER UPDATE ON documents_chunk BEGIN
        INSERT INTO chunk_fts(chunk_fts, rowid, text) VALUES('delete', old.id, old.text);
        INSERT INTO chunk_fts(rowid, text) VALUES (new.id, new.text);
    END;
    """,
]

DROP = [
    "DROP TRIGGER IF EXISTS chunk_fts_au;",
    "DROP TRIGGER IF EXISTS chunk_fts_ad;",
    "DROP TRIGGER IF EXISTS chunk_fts_ai;",
    "DROP TABLE IF EXISTS chunk_fts;",
]


class Migration(migrations.Migration):
    dependencies = [("documents", "0001_initial")]

    operations = [
        migrations.RunSQL(sql=CREATE, reverse_sql=DROP),
    ]
