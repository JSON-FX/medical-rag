"""Near-miss axis detection.

A near-miss question is only legitimate if the corpus genuinely lacks the
answer. Withholding a label SECTION is not enough to guarantee that: metformin's
`dosage_and_administration` carries a full "Pediatric Dosage" paragraph even
with `pediatric_use` withheld, so a pediatric question about metformin is
answerable from the shipped text. Absence is therefore measured over the
assembled corpus, not inferred from section names (spec 2.2.1).
"""
from __future__ import annotations

import re

NEAR_MISS_AXES: dict[str, list[str]] = {
    "pediatric": [r"pediatric", r"children", r"\bchild\b", r"infant", r"neonate", r"adolescent"],
    "overdose": [r"overdos", r"toxicity", r"ingestion of amounts"],
    "pregnancy": [r"pregnan", r"lactation", r"nursing", r"breast-?feed", r"teratogen"],
    "geriatric": [r"geriatric", r"elderly", r"older patients"],
    "hepatic": [r"hepatic impairment", r"liver impairment", r"hepatic dysfunction"],
}


def verified_absent_axes(text: str) -> list[str]:
    """Axes with no keyword anywhere in `text`, sorted for stable manifests."""
    lowered = text.lower()
    return sorted(
        axis
        for axis, patterns in NEAR_MISS_AXES.items()
        if not any(re.search(p, lowered) for p in patterns)
    )
