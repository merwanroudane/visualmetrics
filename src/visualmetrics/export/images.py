"""Export figures and results to files (blueprint section 66).

Format is chosen from the file extension:

``.html``   self-contained interactive page (no optional dependency needed);
``.json``   the lab state plus every figure as Plotly JSON;
``.png``    static raster - needs ``kaleido``;
``.svg``    static vector - needs ``kaleido``;
``.pdf``    static vector - needs ``kaleido``.

Multi-panel results write one file per panel, with the panel id appended to
the stem, and the returned list says exactly which files were written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.exceptions import ExportError, MissingDependencyError

if TYPE_CHECKING:  # pragma: no cover
    from ..core.state import LabResult

__all__ = [
    "json_dumps",
    "export_result",
    "export_figure",
    "figure_to_html",
    "figure_to_json",
    "SUPPORTED_FORMATS",
    "STATIC_FORMATS",
]

STATIC_FORMATS = frozenset({".png", ".svg", ".pdf", ".jpeg", ".jpg", ".webp"})
SUPPORTED_FORMATS = frozenset({".html", ".json"}) | STATIC_FORMATS


def _require_kaleido(fmt: str) -> None:
    try:
        import kaleido  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any import failure means unusable
        raise MissingDependencyError(
            "kaleido", "export", feature=f"static {fmt.lstrip('.')} export"
        ) from exc


def json_dumps(payload: Any, *, indent: int | None = None) -> str:
    """Serialise a payload that may contain numpy arrays.

    Plotly keeps numpy arrays inside ``to_plotly_json()`` output, which the
    stdlib encoder refuses. Plotly ships an encoder that handles them; when
    Plotly is absent the arrays are converted first.
    """
    try:
        from plotly.utils import PlotlyJSONEncoder

        return json.dumps(payload, cls=PlotlyJSONEncoder, ensure_ascii=False, indent=indent)
    except Exception:  # noqa: BLE001 - plotly missing or its encoder unusable
        return json.dumps(_plain(payload), ensure_ascii=False, indent=indent)


def figure_to_html(figure: Any, *, full_page: bool = True, include_js: bool = True) -> str:
    """Render a Plotly figure as HTML text."""
    if hasattr(figure, "to_html"):
        return figure.to_html(
            full_html=full_page,
            include_plotlyjs="cdn" if include_js else False,
            config={"displaylogo": False, "responsive": True},
        )
    raise ExportError(
        f"cannot export an object of type {type(figure).__name__} to HTML",
        key="errors.export",
    )


def figure_to_json(figure: Any) -> dict[str, Any]:
    """Render a Plotly figure as a plain dictionary."""
    if hasattr(figure, "to_plotly_json"):
        return figure.to_plotly_json()
    if isinstance(figure, dict):
        return figure
    raise ExportError(
        f"cannot serialise an object of type {type(figure).__name__}", key="errors.export"
    )


def export_figure(figure: Any, path: str | Path, *, scale: float = 2.0,
                  width: int | None = None, height: int | None = None) -> Path:
    """Write one figure to ``path``, choosing the format from the suffix."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ExportError(
            f"unsupported export format {suffix!r}; "
            f"choose one of {sorted(SUPPORTED_FORMATS)}",
            key="errors.export",
        )
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".html":
        path.write_text(figure_to_html(figure), encoding="utf-8")
    elif suffix == ".json":
        path.write_text(json_dumps(figure_to_json(figure), indent=2), encoding="utf-8")
    else:
        _require_kaleido(suffix)
        kwargs: dict[str, Any] = {"scale": scale}
        if width:
            kwargs["width"] = width
        if height:
            kwargs["height"] = height
        figure.write_image(str(path), **kwargs)
    return path


def export_result(result: LabResult, path: str | Path, **kwargs: Any) -> list[Path]:
    """Export every panel of a lab result.

    A single panel writes exactly ``path``. Several panels write
    ``stem-<panel id><suffix>`` each, so nothing is silently overwritten.

    For ``.json`` the whole result is written as one document - figures,
    metrics, resolved parameters, warnings and the generated code - because a
    reader of that file wants the science, not only the picture.
    """
    path = Path(path)
    if not result.panels:
        raise ExportError(
            f"{result.concept_id or 'this result'} produced no figure to export",
            key="errors.export",
        )

    if path.suffix.lower() == ".json":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_dumps(result_to_dict(result), indent=2), encoding="utf-8")
        return [path]

    if len(result.panels) == 1:
        return [export_figure(result.panels[0].figure, path, **kwargs)]

    written: list[Path] = []
    for panel in result.panels:
        target = path.with_name(f"{path.stem}-{panel.id}{path.suffix}")
        written.append(export_figure(panel.figure, target, **kwargs))
    return written


def result_to_dict(result: LabResult) -> dict[str, Any]:
    """Serialise a whole lab result, evidence labelling included."""
    return {
        "concept_id": result.concept_id,
        "evidence": result.evidence.value if result.evidence else None,
        "resolved_parameters": _plain(result.resolved_parameters),
        "dgp": result.dgp,
        "code": result.code,
        "warnings": list(result.warnings),
        "metrics": [
            {"key": m.key, "label": m.label, "value": _plain(m.value)} for m in result.metrics
        ],
        "assumptions": [
            {"id": a.id, "label": a.label, "holds": bool(a.holds)} for a in result.assumptions
        ],
        "explanations": [
            {"tab": b.tab, "title": b.title, "body": b.body} for b in result.explanations
        ],
        "animations": [
            {
                "id": a.id,
                "purpose": a.purpose,
                "summary": a.summary,
                "evidence": a.evidence.value,
                "steps": [
                    {
                        "id": s.id,
                        "frame": s.frame,
                        "title": s.title,
                        "what_you_see": s.what_you_see,
                        "what_changed": s.what_changed,
                        "why": s.why,
                        "interpretation": s.interpretation,
                        "conclusion": s.conclusion,
                        "warning": s.warning,
                    }
                    for s in a.steps
                ],
            }
            for a in result.animations
        ],
        "panels": [
            {
                "id": p.id,
                "title": p.title,
                "caption": p.caption,
                "evidence": p.evidence.value if p.evidence else None,
                "figure": figure_to_json(p.figure) if p.figure is not None else None,
            }
            for p in result.panels
        ],
    }


def _plain(value: Any) -> Any:
    """Make numpy types JSON-serialisable without pulling numpy into the import path."""
    import numpy as np

    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
