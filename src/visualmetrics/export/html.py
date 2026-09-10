"""Self-contained HTML reports (blueprint section 66.2).

A report is not a screenshot: it carries the figures *and* the layer that makes
them scientific - the evidence badge with its caveat, the assumptions and their
status, the warnings, the numbers, the generated Python, and for animations the
per-frame explanation. Exporting a picture without that layer would strip out
exactly what the package exists to add.

The page needs no build step and no optional dependency; Plotly is pulled from
its CDN, and if it is unavailable the figures degrade to their captions while
the text remains readable.
"""

from __future__ import annotations

import html as _html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.evidence import badge_for
from ..core.exceptions import ExportError
from ..i18n.rtl import direction, is_rtl
from ..i18n.translator import get_translator
from ..version import __version__
from .images import figure_to_json, json_dumps

if TYPE_CHECKING:  # pragma: no cover
    from ..core.state import LabResult

__all__ = ["export_html_report", "render_html_report"]

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"

_CSS = """
:root {
  --ink: #14181f; --muted: #5b6472; --line: #dfe3ea; --bg: #ffffff;
  --panel: #f7f8fa; --accent: #1f4e79; --warn: #a34a00; --ok: #1c6b3a;
}
@media (prefers-color-scheme: dark) {
  :root { --ink: #e8ecf2; --muted: #9aa4b2; --line: #2b313b; --bg: #12151a;
          --panel: #1a1e25; --accent: #7ab3e6; --warn: #e0a35f; --ok: #6cc48f; }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 2rem 1.25rem 4rem; background: var(--bg); color: var(--ink);
       font: 15px/1.6 system-ui, -apple-system, "Segoe UI", Roboto, "Noto Naskh Arabic", sans-serif; }
main { max-width: 60rem; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 2.25rem 0 .5rem; padding-bottom: .3rem;
     border-bottom: 1px solid var(--line); }
h3 { font-size: 1rem; margin: 1.25rem 0 .35rem; }
.sub { color: var(--muted); margin: 0 0 1.25rem; }
.badge { display: inline-block; padding: .2rem .6rem; border-radius: 999px;
         border: 1px solid var(--accent); color: var(--accent); font-size: .8rem;
         font-weight: 600; }
.caveat { color: var(--muted); font-size: .88rem; margin: .4rem 0 0; }
.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
         padding: 1rem; margin: 1rem 0; }
.figure { width: 100%; min-height: 380px; }
table { border-collapse: collapse; width: 100%; margin: .5rem 0; }
th, td { text-align: start; padding: .4rem .6rem; border-bottom: 1px solid var(--line); }
th { color: var(--muted); font-weight: 600; font-size: .85rem; }
td.num { font-variant-numeric: tabular-nums; }
.warn { color: var(--warn); }
.ok { color: var(--ok); }
ul.plain { list-style: none; padding: 0; }
ul.plain li { padding: .3rem 0; border-bottom: 1px solid var(--line); }
pre { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
      padding: .9rem; overflow-x: auto; font-size: .85rem; direction: ltr; text-align: left; }
footer { color: var(--muted); font-size: .82rem; margin-top: 3rem;
         border-top: 1px solid var(--line); padding-top: .75rem; }
"""


def _esc(text: Any) -> str:
    return _html.escape(str(text if text is not None else ""), quote=True)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "n/a"
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e6):
            return f"{value:.3e}"
        return f"{value:.4f}".rstrip("0").rstrip(".") or "0"
    return _esc(value)


def render_html_report(
    result: LabResult,
    *,
    title: str | None = None,
    language: str | None = None,
    include_code: bool = True,
) -> str:
    """Build the report as a string."""
    if not result.panels:
        raise ExportError(
            f"{result.concept_id or 'this result'} produced no figure to report",
            key="errors.export",
        )
    tr = get_translator(language)
    lang = getattr(tr, "language", language or "en")
    rtl = is_rtl(lang)

    heading = title or tr.t(f"concepts.{result.concept_id}.title", result.concept_id)
    parts: list[str] = []

    parts.append(f"<h1>{_esc(heading)}</h1>")
    if result.evidence is not None:
        badge = badge_for(result.evidence)
        label = tr.t(badge.label_key, result.evidence.value.replace("_", " ").title())
        caveat = tr.t(badge.caveat_key, "")
        parts.append(f'<p class="sub"><span class="badge">{badge.icon} {_esc(label)}</span></p>')
        if caveat:
            parts.append(f'<p class="caveat">{_esc(caveat)}</p>')

    if result.warnings:
        parts.append(f"<h2>{_esc(tr.t('common.warnings', 'Warnings'))}</h2><ul class='plain'>")
        parts += [f'<li class="warn">{_esc(w)}</li>' for w in result.warnings]
        parts.append("</ul>")

    # -- figures ----------------------------------------------------------
    figures: list[tuple[str, dict[str, Any]]] = []
    parts.append(f"<h2>{_esc(tr.t('common.figures', 'Figures'))}</h2>")
    for index, panel in enumerate(result.panels):
        div_id = f"vm-fig-{index}"
        caption = panel.caption or panel.description or ""
        badge_html = ""
        if panel.evidence is not None:
            pb = badge_for(panel.evidence)
            plabel = tr.t(pb.label_key, panel.evidence.value)
            badge_html = f'<span class="badge">{pb.icon} {_esc(plabel)}</span>'
        parts.append(
            f'<div class="panel"><h3>{_esc(panel.title or panel.id)} {badge_html}</h3>'
            f'<div class="figure" id="{div_id}"></div>'
            + (f"<p class='caveat'>{_esc(caption)}</p>" if caption else "")
            + "</div>"
        )
        if panel.figure is not None:
            figures.append((div_id, figure_to_json(panel.figure)))

    # -- numbers ----------------------------------------------------------
    if result.metrics:
        parts.append(f"<h2>{_esc(tr.t('common.metrics', 'Results'))}</h2><table><tbody>")
        for metric in result.metrics:
            parts.append(
                f"<tr><th>{_esc(metric.label)}</th>"
                f'<td class="num">{_fmt(metric.value)}</td></tr>'
            )
        parts.append("</tbody></table>")

    # -- assumptions ------------------------------------------------------
    if result.assumptions:
        parts.append(f"<h2>{_esc(tr.t('common.assumptions', 'Assumptions'))}</h2><table><tbody>")
        for assumption in result.assumptions:
            state = "ok" if assumption.holds else "warn"
            mark = "✓" if assumption.holds else "✗"
            note = getattr(assumption, "detail", "") or ""
            parts.append(
                f"<tr><th>{_esc(assumption.label)}</th>"
                f'<td class="{state}">{mark} {_esc(note)}</td></tr>'
            )
        parts.append("</tbody></table>")

    # -- explanations -----------------------------------------------------
    if result.explanations:
        parts.append(f"<h2>{_esc(tr.t('common.explanation', 'Explanation'))}</h2>")
        for block in result.explanations:
            parts.append(
                f'<div class="panel"><h3>{_esc(block.title)}</h3><p>{_esc(block.body)}</p></div>'
            )

    # -- animation explanation layer --------------------------------------
    for animation in result.animations:
        parts.append(f"<h2>{_esc(tr.t('common.animation', 'Animation'))}: {_esc(animation.id)}</h2>")
        if animation.purpose:
            parts.append(f"<p>{_esc(animation.purpose)}</p>")
        parts.append("<table><thead><tr>")
        for column in ("#", "What you see", "What changed", "Why", "How to read it"):
            parts.append(f"<th>{_esc(tr.t(f'common.animation_col.{column}', column))}</th>")
        parts.append("</tr></thead><tbody>")
        for step in animation.steps:
            parts.append(
                f"<tr><td class='num'>{step.frame}</td><td>{_esc(step.what_you_see)}</td>"
                f"<td>{_esc(step.what_changed)}</td><td>{_esc(step.why)}</td>"
                f"<td>{_esc(step.interpretation)}</td></tr>"
            )
        parts.append("</tbody></table>")
        if animation.summary:
            parts.append(f"<p><strong>{_esc(animation.summary)}</strong></p>")

    # -- reproducibility --------------------------------------------------
    if include_code and result.code:
        parts.append(f"<h2>{_esc(tr.t('common.code', 'Reproduce this in Python'))}</h2>")
        parts.append(f"<pre><code>{_esc(result.code)}</code></pre>")
    if result.dgp:
        parts.append(f"<h2>{_esc(tr.t('common.dgp', 'Data generating process'))}</h2>")
        parts.append(f"<pre><code>{_esc(result.dgp)}</code></pre>")

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts.append(
        f"<footer>VisualMetrics {_esc(__version__)} &middot; {_esc(result.concept_id)} "
        f"&middot; {stamp}</footer>"
    )

    script = ""
    if figures:
        payload = json_dumps([{"id": div, "fig": fig} for div, fig in figures])
        script = (
            f'<script src="{PLOTLY_CDN}"></script>\n<script>\n'
            f"const VM_FIGURES = {payload};\n"
            "if (window.Plotly) {\n"
            "  for (const item of VM_FIGURES) {\n"
            "    Plotly.newPlot(item.id, item.fig.data, item.fig.layout,\n"
            "                   {displaylogo: false, responsive: true});\n"
            "  }\n"
            "}\n</script>"
        )

    return (
        f'<!doctype html>\n<html lang="{lang}" dir="{direction(lang)}">\n<head>\n'
        f'<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(heading)}</title>\n<style>{_CSS}</style>\n</head>\n"
        f'<body{" class=rtl" if rtl else ""}>\n<main>\n'
        + "\n".join(parts)
        + f"\n</main>\n{script}\n</body>\n</html>\n"
    )


def export_html_report(result: LabResult, path: str | Path, **kwargs: Any) -> Path:
    """Write a complete HTML report to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html_report(result, **kwargs), encoding="utf-8")
    return path
