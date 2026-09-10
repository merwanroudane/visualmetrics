"""Cross-language concept search.

The index covers concept ids, tags, aliases and the English/Arabic/French
titles, summaries and glossary terms, so a learner can type ``power``,
``puissance`` or ``القوة الإحصائية`` and land on the same lab.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from ..core.registry import registry
from ..i18n.glossary import glossary_entries, normalize
from ..i18n.translator import LANGUAGES, _load_bundle

__all__ = ["search_concepts", "build_index", "clear_index"]


@lru_cache(maxsize=1)
def build_index() -> dict[str, list[tuple[str, float]]]:
    """token -> [(concept_id, weight)] built from every language bundle."""
    registry.ensure_loaded()
    index: dict[str, list[tuple[str, float]]] = {}

    def add(text: str, concept_id: str, weight: float) -> None:
        for token in normalize(text).split():
            if len(token) < 2:
                continue
            index.setdefault(token, []).append((concept_id, weight))
        whole = normalize(text)
        if " " in whole:
            index.setdefault(whole, []).append((concept_id, weight * 1.5))

    bundles = {lang: _load_bundle(lang) for lang in LANGUAGES}
    for spec in registry.specs():
        add(spec.id.replace(".", " ").replace("_", " "), spec.id, 3.0)
        for tag in spec.tags:
            add(tag, spec.id, 2.5)
        for alias in spec.aliases:
            add(alias, spec.id, 3.0)
        add(spec.subdomain.replace("_", " "), spec.id, 1.2)
        add(spec.domain.value.replace("_", " "), spec.id, 0.8)
        for bundle in bundles.values():
            title = bundle.get(spec.title_key)
            if isinstance(title, str):
                add(title, spec.id, 3.0)
            summary = bundle.get(spec.summary_key)
            if isinstance(summary, str):
                add(summary, spec.id, 0.7)

    for entry in glossary_entries().values():
        if not entry.concept_id:
            continue
        for key in entry.search_keys():
            add(key, entry.concept_id, 2.2)

    return index


def clear_index() -> None:
    build_index.cache_clear()


def search_concepts(query: str, limit: int = 12, *, include_planned: bool = True
                    ) -> list[dict[str, Any]]:
    """Rank concepts against a free-text query in any supported language."""
    from ..i18n.translator import get_translator

    needle = normalize(query)
    if not needle:
        return []
    index = build_index()
    scores: dict[str, float] = {}

    for cid, weight in index.get(needle, []):
        scores[cid] = scores.get(cid, 0.0) + weight * 2.0
    for token in needle.split():
        for cid, weight in index.get(token, []):
            scores[cid] = scores.get(cid, 0.0) + weight
        if len(token) >= 3:
            for key, entries in index.items():
                if key.startswith(token) and key != token:
                    for cid, weight in entries:
                        scores[cid] = scores.get(cid, 0.0) + weight * 0.35

    tr = get_translator()
    out: list[dict[str, Any]] = []
    for cid, score in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])):
        try:
            spec = registry.get(cid)
        except Exception:
            continue
        if not include_planned and not spec.status.is_implemented:
            continue
        out.append(
            {
                "id": cid,
                "score": round(score, 3),
                "title": tr.t(spec.title_key),
                "summary": tr.t(spec.summary_key),
                "domain": spec.domain.value,
                "status": spec.status.value,
                "implemented": spec.status.is_implemented,
            }
        )
        if len(out) >= limit:
            break
    return out
