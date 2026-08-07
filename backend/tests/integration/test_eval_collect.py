"""The collect pass must measure the eval corpus and nothing else.

Chroma is sandboxed to a temp dir, but the lexical leg is not:
`chat/lexical_search.py` reads the `chunk_fts` table in whatever database
Django is configured with. Two things go wrong if that is the developer's
`db.sqlite3` — chunks from anything uploaded through /documents match eval
questions and enter RRF fusion, and the corpus the pass ingests itself
accumulates across runs. The second one has already bitten: three corpus
copies built up, duplicate chunks tied in rank fusion, and a genuinely
answerable question was refused.

`evals/collect.py` fixes both by pointing DJANGO_DB_NAME at a fresh temp
database before `django.setup()`. That ordering is the whole mechanism, so it
is what this test pins.
"""
import pathlib
import subprocess
import sys

BACKEND = pathlib.Path(__file__).resolve().parents[2]

# Import the module and report the database Django actually ended up with.
# A subprocess is required: importing evals.collect mutates os.environ and
# calls django.setup(), which cannot be re-run inside an already-configured
# pytest process — and doing so would leak the temp database into every test
# that followed. Importing does not run the pass; main() is __main__-guarded.
PROBE = """
import evals.collect  # noqa: F401  — sets DJANGO_DB_NAME, then django.setup()
from django.conf import settings
print(settings.DATABASES["default"]["NAME"])
"""


def _configured_database() -> str:
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=BACKEND,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"probe failed:\n{result.stderr}"
    return result.stdout.strip().splitlines()[-1]


def test_collect_runs_against_an_isolated_database_not_the_developers():
    configured = pathlib.Path(_configured_database())
    dev_database = BACKEND / "db.sqlite3"

    assert configured != dev_database, (
        "the collect pass would read and write the developer's own database; "
        "uploaded documents would contaminate the lexical leg"
    )
    assert not configured.is_relative_to(BACKEND), (
        f"{configured} is inside the repo — the pass must use a temp database"
    )


def test_the_isolated_database_is_fresh_per_run():
    """Two runs must not share a database, or the corpus accumulates and
    duplicate chunks displace a relevant chunk out of the top-k."""
    assert _configured_database() != _configured_database()
