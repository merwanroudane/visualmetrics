"""Trilingual glossary and cross-language search index.

Typing ``power``, ``statistical power``, ``القوة الإحصائية`` or
``puissance statistique`` must all reach the same concept (blueprint 38.5).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from .translator import LANGUAGES, _load_bundle

__all__ = ["GlossaryEntry", "glossary_entries", "lookup", "normalize", "search_terms"]

_AR_DIACRITICS = dict.fromkeys(range(0x064B, 0x0653))
_AR_TATWEEL = {0x0640: None}


def normalize(text: str) -> str:
    """Fold case, accents and Arabic orthographic variation for search."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.translate(_AR_DIACRITICS).translate(_AR_TATWEEL)
    text = (
        text.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ة", "ه")
    )
    return " ".join(text.lower().split())


@dataclass(frozen=True)
class GlossaryEntry:
    """One scientific term in English, Arabic and French."""

    key: str
    terms: dict[str, str] = field(default_factory=dict)
    definitions: dict[str, str] = field(default_factory=dict)
    acronym: str = ""
    aliases: tuple[str, ...] = ()
    concept_id: str | None = None
    related: tuple[str, ...] = ()

    def term(self, language: str = "en") -> str:
        return self.terms.get(language) or self.terms.get("en") or self.key

    def definition(self, language: str = "en") -> str:
        return self.definitions.get(language) or self.definitions.get("en") or ""

    def bilingual(self, language: str = "en") -> str:
        english = self.terms.get("en", self.key)
        local = self.terms.get(language, english)
        return local if local == english else f"{local} ({english})"

    def search_keys(self) -> list[str]:
        out = [normalize(v) for v in self.terms.values() if v]
        out += [normalize(a) for a in self.aliases if a]
        if self.acronym:
            out.append(normalize(self.acronym))
        out.append(normalize(self.key.rsplit(".", 1)[-1].replace("_", " ")))
        return [k for k in out if k]


@lru_cache(maxsize=1)
def glossary_entries() -> dict[str, GlossaryEntry]:
    """Assemble glossary entries by merging the three language bundles."""
    keys: set[str] = set()
    bundles = {lang: _load_bundle(lang) for lang in LANGUAGES}
    for bundle in bundles.values():
        for key in bundle:
            if key.startswith("glossary.") and key.endswith(".term"):
                keys.add(key[len("glossary.") : -len(".term")])

    entries: dict[str, GlossaryEntry] = {}
    for key in sorted(keys):
        terms: dict[str, str] = {}
        definitions: dict[str, str] = {}
        for lang, bundle in bundles.items():
            term = bundle.get(f"glossary.{key}.term")
            if isinstance(term, str) and term:
                terms[lang] = term
            definition = bundle.get(f"glossary.{key}.definition")
            if isinstance(definition, str) and definition:
                definitions[lang] = definition
        meta = bundles["en"]
        raw_aliases = meta.get(f"glossary.{key}.aliases", "")
        aliases = tuple(a.strip() for a in str(raw_aliases).split("|") if a.strip())
        entries[key] = GlossaryEntry(
            key=key,
            terms=terms,
            definitions=definitions,
            acronym=str(meta.get(f"glossary.{key}.acronym", "") or ""),
            aliases=aliases,
            concept_id=meta.get(f"glossary.{key}.concept") or None,
            related=tuple(
                r.strip()
                for r in str(meta.get(f"glossary.{key}.related", "") or "").split("|")
                if r.strip()
            ),
        )
    return entries


def lookup(term: str) -> GlossaryEntry | None:
    """Find a glossary entry by any of its language forms or aliases."""
    needle = normalize(term)
    if not needle:
        return None
    entries = glossary_entries()
    if term in entries:
        return entries[term]
    for entry in entries.values():
        if needle in entry.search_keys():
            return entry
    for entry in entries.values():
        if any(needle in key for key in entry.search_keys()):
            return entry
    return None


def search_terms(query: str, limit: int = 10) -> list[GlossaryEntry]:
    """Rank glossary entries against a query in any supported language."""
    needle = normalize(query)
    if not needle:
        return []
    scored: list[tuple[float, GlossaryEntry]] = []
    for entry in glossary_entries().values():
        best = 0.0
        for key in entry.search_keys():
            if key == needle:
                best = max(best, 1.0)
            elif key.startswith(needle):
                best = max(best, 0.8)
            elif needle in key:
                best = max(best, 0.6)
            elif key in needle:
                best = max(best, 0.5)
        if best:
            scored.append((best, entry))
    scored.sort(key=lambda pair: (-pair[0], pair[1].key))
    return [entry for _, entry in scored[:limit]]


def glossary_payload(language: str = "en") -> list[dict[str, Any]]:
    """Flat structure used by the GUI glossary page."""
    return [
        {
            "key": e.key,
            "term": e.term(language),
            "bilingual": e.bilingual(language),
            "definition": e.definition(language),
            "acronym": e.acronym,
            "concept": e.concept_id,
        }
        for e in glossary_entries().values()
    ]
