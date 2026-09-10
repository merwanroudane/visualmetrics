"""The concept knowledge graph.

Answers "what should I learn first?", "what comes next?" and "what is this
often confused with?" without requiring networkx.
"""

from __future__ import annotations

from typing import Any

from ..core.registry import registry

__all__ = [
    "prerequisite_chain",
    "successors",
    "neighbours",
    "graph_payload",
    "topological_order",
]


def prerequisite_chain(concept_id: str, *, depth: int = 3) -> list[str]:
    """Prerequisites in learning order (deepest first), de-duplicated."""
    ordered: list[str] = []
    seen: set[str] = set()

    def walk(cid: str, level: int) -> None:
        if level > depth or cid in seen:
            return
        seen.add(cid)
        try:
            spec = registry.get(cid)
        except Exception:  # noqa: BLE001
            return
        for prereq in spec.prerequisites:
            walk(prereq, level + 1)
        if cid != concept_id and cid not in ordered:
            ordered.append(cid)

    walk(concept_id, 0)
    return ordered


def successors(concept_id: str) -> list[str]:
    """Concepts that list ``concept_id`` as a prerequisite, plus declared next steps."""
    registry.ensure_loaded()
    out = [s.id for s in registry.specs() if concept_id in s.prerequisites]
    try:
        out.extend(registry.get(concept_id).next_concepts)
    except Exception:  # noqa: BLE001
        pass
    return sorted(dict.fromkeys(out))


def neighbours(concept_id: str) -> dict[str, list[str]]:
    spec = registry.get(concept_id)
    return {
        "prerequisites": list(spec.prerequisites),
        "related": list(spec.related),
        "next": successors(concept_id),
        "confused_with": list(spec.confused_with),
    }


def topological_order(concept_ids: list[str] | None = None) -> list[str]:
    """Order concepts so that prerequisites always come first (stable)."""
    registry.ensure_loaded()
    ids = list(concept_ids) if concept_ids else [s.id for s in registry.specs()]
    id_set = set(ids)
    visited: set[str] = set()
    stack: set[str] = set()
    out: list[str] = []

    def visit(cid: str) -> None:
        if cid in visited or cid not in id_set:
            return
        if cid in stack:  # cycle - fall back to insertion order
            return
        stack.add(cid)
        try:
            for prereq in registry.get(cid).prerequisites:
                visit(prereq)
        except Exception:  # noqa: BLE001
            pass
        stack.discard(cid)
        visited.add(cid)
        out.append(cid)

    for cid in sorted(ids):
        visit(cid)
    return out


def graph_payload(*, implemented_only: bool = False) -> dict[str, Any]:
    """Nodes and edges for the GUI's prerequisite map."""
    from ..i18n.translator import get_translator

    tr = get_translator()
    specs = registry.specs(implemented_only=implemented_only)
    known = {s.id for s in specs}
    nodes = [
        {
            "id": s.id,
            "label": tr.t(s.title_key),
            "domain": s.domain.value,
            "status": s.status.value,
            "levels": [lv.value for lv in s.levels],
        }
        for s in specs
    ]
    edges: list[dict[str, str]] = []
    for s in specs:
        for prereq in s.prerequisites:
            if prereq in known:
                edges.append({"source": prereq, "target": s.id, "kind": "prerequisite"})
        for rel in s.related:
            if rel in known and rel > s.id:
                edges.append({"source": s.id, "target": rel, "kind": "related"})
    return {"nodes": nodes, "edges": edges}
