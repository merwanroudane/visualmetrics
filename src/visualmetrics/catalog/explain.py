"""Plain-text concept explanations for the API, CLI and notebook."""

from __future__ import annotations

from ..core.evidence import badge_for
from ..core.registry import registry
from ..i18n.translator import get_translator

__all__ = ["explain_concept", "concept_card"]


def concept_card(concept_id: str, *, language: str | None = None) -> dict[str, object]:
    """Structured concept summary used by the GUI header and the CLI."""
    spec = registry.get(registry.resolve(concept_id))
    tr = get_translator(language)
    badge = badge_for(spec.evidence)
    return {
        "id": spec.id,
        "title": tr.t(spec.title_key),
        "summary": tr.t(spec.summary_key),
        "domain": tr.t(spec.domain.label_key, spec.domain.value.replace("_", " ").title()),
        "subdomain": spec.subdomain.replace("_", " "),
        "status": spec.status.value,
        "evidence": spec.evidence.value,
        "evidence_label": tr.t(badge.label_key),
        "evidence_caveat": tr.t(badge.caveat_key),
        "levels": [lv.value for lv in spec.levels],
        "modes": [m.value for m in spec.modes],
        "prerequisites": list(spec.prerequisites),
        "related": list(spec.related),
        "scenarios": [s.id for s in spec.scenarios],
        "controls": [c.id for c in spec.controls],
        "references": [
            {"citation": r.citation, "kind": r.kind, "url": r.url, "doi": r.doi}
            for r in spec.references
        ],
        "required_extras": list(spec.required_extras),
    }


def explain_concept(concept_id: str, *, level: str | None = None,
                    language: str | None = None) -> str:
    """Render a concept as readable localized text."""
    resolved = registry.resolve(concept_id)
    spec = registry.get(resolved)
    tr = get_translator(language)
    card = concept_card(resolved, language=language)

    lines = [str(card["title"]), "=" * max(len(str(card["title"])), 8), ""]
    lines.append(str(card["summary"]))
    lines.append("")
    lines.append(f"{tr.t('ui.evidence', 'Evidence')}: {card['evidence_label']}")
    lines.append(f"  {card['evidence_caveat']}")
    lines.append("")

    if spec.prerequisites:
        lines.append(tr.t("ui.prerequisites", "Learn first") + ":")
        for prereq in spec.prerequisites:
            try:
                lines.append(f"  - {tr.t(registry.get(prereq).title_key)}  [{prereq}]")
            except Exception:
                lines.append(f"  - {prereq}")
        lines.append("")

    # explanation blocks contributed by the lab's own translation keys
    for tab in ("intuition", "assumptions", "math", "misconceptions", "warning"):
        key = f"concepts.{resolved}.{tab}"
        text = tr.t(key, None)
        if text:
            lines.append(tr.t(f"tabs.{tab}", tab.title()) + ":")
            lines.append("  " + text.replace("\n", "\n  "))
            lines.append("")

    if spec.scenarios:
        lines.append(tr.t("ui.scenarios", "Scenarios") + ":")
        for sc in spec.scenarios:
            lines.append(f"  - {tr.t(sc.translation_key, sc.id)}  [{sc.id}]")
        lines.append("")

    if spec.references:
        lines.append(tr.t("ui.references", "References") + ":")
        for r in spec.references:
            suffix = f"  <{r.url}>" if r.url else (f"  doi:{r.doi}" if r.doi else "")
            lines.append(f"  - {r.citation}{suffix}")
        lines.append("")

    if not spec.status.is_implemented:
        lines.append(tr.t("ui.planned_notice",
                          "This concept is catalogued but not implemented yet."))
    del level
    return "\n".join(lines).rstrip() + "\n"
