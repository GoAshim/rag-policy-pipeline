"""
Keyword tagging: find known names (drugs, states, ...) in a piece of text.

Embedding models are good at meaning but bad at telling apart similar
proper nouns -- to them "Xifaxan" and "Vraylar" are both just "some drug".
So at indexing time we tag every chunk with the known names it mentions,
and at question time we pull the same names out of the question and only
search chunks carrying at least one of those tags.

The lists are plain data in data/keyword_lexicon.json so adding a new
name is a one-line edit. We use exact word matching against a known list
rather than a general NER model: spaCy's stock English model labels
"Xifaxan" as a PERSON, and doesn't know most brand-name drugs at all.
"""

import json
import re
from functools import lru_cache

from src import config

LEXICON_PATH = config.PROJECT_ROOT / "data" / "keyword_lexicon.json"


@lru_cache(maxsize=1)
def _patterns() -> list[tuple[str, re.Pattern]]:
    lexicon = json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
    pats = []
    for category, names in lexicon.items():
        for name in names:
            # Allow any run of spaces/markdown junk between words of a
            # multi-word name ("Trelegy<br>Ellipta", "New  York").
            words = [re.escape(w) for w in name.split()]
            body = r"(?:\s|<br>|\*|_)+".join(words)
            pats.append((name, re.compile(rf"(?<![A-Za-z]){body}(?![A-Za-z])", re.IGNORECASE)))
    return pats


def extract_keywords(text: str) -> list[str]:
    """Return the sorted list of known names that appear in `text`."""
    if not text:
        return []
    return sorted({name for name, pat in _patterns() if pat.search(text)})
