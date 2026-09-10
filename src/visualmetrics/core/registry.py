"""The concept registry.

The registry holds *specifications* (cheap) and resolves *labs* (expensive) on
demand. Concept modules are imported only when a lab is actually opened, so a
probability lab never drags in torch, shap, econml or manim.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Iterator
from typing import Any

from .concepts import ConceptSpec, Domain, Lab, Level, Status
from .exceptions import ConceptNotFoundError, DuplicateConceptError, PluginError

__all__ = ["ConceptRegistry", "registry", "register", "get_spec", "get_lab", "all_specs"]


class ConceptRegistry:
    """Thread-safe registry of concept specs and lazily loaded labs."""

    def __init__(self) -> None:
        self._specs: dict[str, ConceptSpec] = {}
        self._labs: dict[str, Lab] = {}
        self._lock = threading.RLock()
        self._catalog_loaded = False
        self._plugins_loaded = False

    # -- registration ------------------------------------------------------
    def register(self, spec: ConceptSpec, *, replace: bool = False) -> ConceptSpec:
        with self._lock:
            if spec.id in self._specs and not replace:
                raise DuplicateConceptError(
                    f"concept {spec.id!r} is already registered", concept=spec.id
                )
            self._specs[spec.id] = spec
            self._labs.pop(spec.id, None)
        return spec

    def register_lab(self, lab: Lab, *, replace: bool = False) -> Lab:
        self.register(lab.spec, replace=replace)
        with self._lock:
            self._labs[lab.spec.id] = lab
        return lab

    # -- catalog loading ---------------------------------------------------
    def ensure_loaded(self) -> None:
        """Import the built-in catalog of concept specs (metadata only)."""
        if self._catalog_loaded:
            return
        with self._lock:
            if self._catalog_loaded:
                return
            self._catalog_loaded = True
            from ..catalog.builtin import load_builtin_catalog

            load_builtin_catalog(self)
            self._load_plugins()

    def _load_plugins(self) -> None:
        if self._plugins_loaded:
            return
        self._plugins_loaded = True
        from ..plugins.discovery import discover_plugins

        for name, plugin in discover_plugins():
            try:
                for spec in plugin.concepts():
                    self.register(spec, replace=False)
            except Exception as exc:  # pragma: no cover - defensive
                raise PluginError(f"plugin {name!r} failed to register: {exc}") from exc

    # -- lookup ------------------------------------------------------------
    def __contains__(self, concept_id: object) -> bool:
        self.ensure_loaded()
        return concept_id in self._specs

    def __len__(self) -> int:
        self.ensure_loaded()
        return len(self._specs)

    def __iter__(self) -> Iterator[ConceptSpec]:
        self.ensure_loaded()
        return iter(sorted(self._specs.values(), key=lambda s: s.id))

    def get(self, concept_id: str) -> ConceptSpec:
        self.ensure_loaded()
        try:
            return self._specs[concept_id]
        except KeyError:
            suggestion = self._suggest(concept_id)
            hint = f" Did you mean {suggestion!r}?" if suggestion else ""
            raise ConceptNotFoundError(
                f"unknown concept {concept_id!r}.{hint}", concept=concept_id
            ) from None

    def _suggest(self, concept_id: str) -> str | None:
        import difflib

        matches = difflib.get_close_matches(concept_id, self._specs, n=1, cutoff=0.6)
        if matches:
            return matches[0]
        tail = concept_id.rsplit(".", 1)[-1]
        for cid in self._specs:
            if cid.endswith("." + tail):
                return cid
        return None

    def resolve(self, concept_id: str) -> str:
        """Accept a full id or an unambiguous short name."""
        self.ensure_loaded()
        if concept_id in self._specs:
            return concept_id
        tail_matches = [c for c in self._specs if c.rsplit(".", 1)[-1] == concept_id]
        if len(tail_matches) == 1:
            return tail_matches[0]
        return concept_id

    def lab(self, concept_id: str) -> Lab:
        """Return the lab implementation, importing its module on first use."""
        concept_id = self.resolve(concept_id)
        spec = self.get(concept_id)
        with self._lock:
            cached = self._labs.get(concept_id)
        if cached is not None:
            return cached
        if spec.status is Status.PLANNED or not spec.module:
            raise ConceptNotFoundError(
                f"concept {concept_id!r} is catalogued as '{spec.status.value}' and has "
                "no implementation yet. It is listed for transparency, not as a working lab.",
                key="errors.concept_planned",
                concept=concept_id,
                status=spec.status.value,
            )
        module = importlib.import_module(spec.module)
        lab = getattr(module, "LAB", None)
        if lab is None:
            raise ConceptNotFoundError(
                f"module {spec.module!r} does not expose a LAB object", concept=concept_id
            )
        with self._lock:
            self._labs[concept_id] = lab
            # a lab module may refine its own spec
            self._specs[concept_id] = lab.spec
        return lab

    # -- queries -----------------------------------------------------------
    def specs(
        self,
        *,
        domain: Domain | str | None = None,
        level: Level | str | None = None,
        status: Status | str | None = None,
        implemented_only: bool = False,
        tag: str | None = None,
    ) -> list[ConceptSpec]:
        self.ensure_loaded()
        out = list(self._specs.values())
        if domain is not None:
            dom = Domain(domain)
            out = [s for s in out if s.domain is dom]
        if level is not None:
            lv = Level(level)
            out = [s for s in out if lv in s.levels]
        if status is not None:
            st = Status(status)
            out = [s for s in out if s.status is st]
        if implemented_only:
            out = [s for s in out if s.status.is_implemented and s.module]
        if tag is not None:
            out = [s for s in out if tag in s.tags]
        return sorted(out, key=lambda s: (s.domain.value, s.subdomain, s.id))

    def domains(self) -> list[Domain]:
        self.ensure_loaded()
        seen = {s.domain for s in self._specs.values()}
        return [d for d in Domain if d in seen]

    def stats(self) -> dict[str, Any]:
        self.ensure_loaded()
        by_status: dict[str, int] = {}
        by_domain: dict[str, int] = {}
        for s in self._specs.values():
            by_status[s.status.value] = by_status.get(s.status.value, 0) + 1
            by_domain[s.domain.value] = by_domain.get(s.domain.value, 0) + 1
        return {
            "total": len(self._specs),
            "implemented": sum(1 for s in self._specs.values() if s.status.is_implemented),
            "by_status": dict(sorted(by_status.items())),
            "by_domain": dict(sorted(by_domain.items())),
        }

    def clear(self) -> None:
        """Testing helper - drop everything and force a reload."""
        with self._lock:
            self._specs.clear()
            self._labs.clear()
            self._catalog_loaded = False
            self._plugins_loaded = False


registry = ConceptRegistry()


def register(spec: ConceptSpec, *, replace: bool = False) -> ConceptSpec:
    return registry.register(spec, replace=replace)


def get_spec(concept_id: str) -> ConceptSpec:
    return registry.get(registry.resolve(concept_id))


def get_lab(concept_id: str) -> Lab:
    return registry.lab(concept_id)


def all_specs(**kw: Any) -> list[ConceptSpec]:
    return registry.specs(**kw)
