"""Preliminary text-overlap audit between conference and journal LaTeX.

This is not a substitute for Crossref Similarity Check.  It reports normalized
8-word shingle containment after removing comments and common LaTeX markup.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


JOURNAL_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = JOURNAL_ROOT.parent
CONFERENCE = PROJECT_ROOT / "Camera-Ready" / "main.tex"
JOURNAL = JOURNAL_ROOT / "tdsc_main.tex"
OUTPUT = JOURNAL_ROOT / "experiments" / "outputs" / "conference_overlap_audit.json"


def normalized_tokens(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"(?m)(?<!\\)%.*$", " ", text)
    text = re.sub(r"\\(?:cite|ref|eqref|label|url|path)(?:\[[^]]*\])?\{[^}]*\}", " ", text)
    text = re.sub(r"\\begin\{[^}]*\}|\\end\{[^}]*\}", " ", text)
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    return re.findall(r"[a-z0-9]+", text.lower())


def shingles(tokens: list[str], width: int) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + width]) for i in range(len(tokens) - width + 1)}


def main() -> None:
    width = 8
    conference_tokens = normalized_tokens(CONFERENCE)
    journal_tokens = normalized_tokens(JOURNAL)
    conference = shingles(conference_tokens, width)
    journal = shingles(journal_tokens, width)
    common = conference & journal
    result = {
        "audit": "preliminary normalized LaTeX overlap",
        "not_crossref_similarity_check": True,
        "shingle_width_words": width,
        "conference_tokens": len(conference_tokens),
        "journal_tokens": len(journal_tokens),
        "conference_unique_shingles": len(conference),
        "journal_unique_shingles": len(journal),
        "common_unique_shingles": len(common),
        "journal_shingle_containment_percent": 100.0 * len(common) / len(journal),
        "conference_shingle_containment_percent": 100.0 * len(common) / len(conference),
        "interpretation": (
            "A diagnostic for verbatim-style reuse after rough LaTeX normalization; "
            "publisher similarity tools use different corpora and rules."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
