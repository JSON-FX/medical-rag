"""One-time openFDA fetch. The ONLY component that touches the network.

Labels are pinned by set_id so a re-fetch returns the same document, and the
extracted text is committed so the eval is reproducible offline and an upstream
change cannot silently move the results.

US federal government works are public domain. No clinical content is authored
for this project.
"""
from __future__ import annotations

import json
import pathlib
import urllib.parse
import urllib.request

from evals.axes import verified_absent_axes
from evals.corpus import FIXTURES, assemble_text

DRUGS = {
    "metformin": "011de1a5-1ac0-4831-9e8d-26ec79ba2205",
    "atenolol": "09b21985-1818-449d-9b29-98f733cf7b9f",
    "amoxicillin": "00fbd46e-05fd-4f8a-9f59-a7a4d01c8e54",
}
INCLUDE = [
    "indications_and_usage",
    "dosage_and_administration",
    "contraindications",
    "adverse_reactions",
    "drug_interactions",
]
WITHHOLD = ["pediatric_use", "overdosage", "pregnancy"]


def fetch(set_id: str) -> dict:
    query = urllib.parse.quote(f'set_id:"{set_id}"')
    url = f"https://api.fda.gov/drug/label.json?search={query}&limit=1"
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)["results"][0]


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "openFDA drug label API (US federal work, public domain)", "drugs": {}}

    for slug, set_id in DRUGS.items():
        record = fetch(set_id)
        included = {s: record[s][0] for s in INCLUDE if record.get(s)}
        withheld = {s: record[s][0] for s in WITHHOLD if record.get(s)}
        (FIXTURES / f"{slug}.json").write_text(
            json.dumps({"set_id": set_id, "included": included, "withheld": withheld}, indent=1)
        )
        manifest["drugs"][slug] = {
            "set_id": set_id,
            "included_sections": sorted(included),
            "withheld_sections": sorted(withheld),
            "included_chars": sum(len(v) for v in included.values()),
            # Measured over the text that actually ships, not inferred from
            # which sections were withheld (spec 2.2.1).
            "verified_absent": verified_absent_axes(assemble_text(included)),
        }
        print(f"{slug:12} {manifest['drugs'][slug]['included_chars']:6} chars  "
              f"absent={manifest['drugs'][slug]['verified_absent']}")

    (FIXTURES / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\nwrote {FIXTURES / 'manifest.json'}")


if __name__ == "__main__":
    main()
