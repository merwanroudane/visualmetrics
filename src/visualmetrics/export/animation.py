"""Export animations with their explanation layer attached (blueprint section 66.3).

An exported animation must carry the same pedagogical layer as the one in the
GUI: what you see, what changed, why, how to read it, and the closing summary
(blueprint section 36.3.1). A silent GIF of moving dots teaches nothing and is
exactly what this package refuses to produce, so the HTML exporter renders the
frame commentary beside the figure and the frame exporter writes the commentary
next to the images.
"""

from __future__ import annotations

import html as _html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.evidence import badge_for
from ..core.exceptions import ExportError
from ..i18n.rtl import direction
from ..i18n.translator import get_translator
from ..visuals.plotly.primitives import strip_plotly_transport
from .images import export_figure, figure_to_json, json_dumps

if TYPE_CHECKING:  # pragma: no cover
    from ..core.state import AnimationSpec

__all__ = ["export_animation", "render_animation_html", "export_animation_frames"]

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"

_CSS = """
:root { --ink:#14181f; --muted:#5b6472; --line:#dfe3ea; --bg:#fff; --panel:#f7f8fa;
        --accent:#1f4e79; --warn:#a34a00; }
@media (prefers-color-scheme: dark) {
  :root { --ink:#e8ecf2; --muted:#9aa4b2; --line:#2b313b; --bg:#12151a; --panel:#1a1e25;
          --accent:#7ab3e6; --warn:#e0a35f; }
}
* { box-sizing:border-box; }
body { margin:0; padding:1.5rem; background:var(--bg); color:var(--ink);
       font:15px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,"Noto Naskh Arabic",sans-serif; }
main { max-width:64rem; margin:0 auto; }
h1 { font-size:1.35rem; margin:0 0 .3rem; }
.badge { display:inline-block; padding:.15rem .55rem; border-radius:999px;
         border:1px solid var(--accent); color:var(--accent); font-size:.78rem; font-weight:600; }
.caveat, .muted { color:var(--muted); font-size:.87rem; }
.stage { display:grid; grid-template-columns:minmax(0,1.4fr) minmax(0,1fr); gap:1rem;
         align-items:start; margin-top:1rem; }
@media (max-width:800px) { .stage { grid-template-columns:1fr; } }
#figure { width:100%; min-height:420px; background:var(--panel);
          border:1px solid var(--line); border-radius:10px; }
.notes { background:var(--panel); border:1px solid var(--line); border-radius:10px;
         padding:1rem; }
.notes h2 { font-size:1rem; margin:0 0 .5rem; }
.notes dt { color:var(--muted); font-size:.8rem; text-transform:uppercase;
            letter-spacing:.03em; margin-top:.7rem; }
.notes dd { margin:.15rem 0 0; }
.controls { display:flex; gap:.5rem; align-items:center; margin-top:1rem; flex-wrap:wrap; }
button { font:inherit; padding:.4rem .9rem; border-radius:7px; border:1px solid var(--line);
         background:var(--bg); color:var(--ink); cursor:pointer; }
button:hover { border-color:var(--accent); }
input[type=range] { flex:1; min-width:12rem; }
.warn { color:var(--warn); }
.summary { margin-top:1.25rem; padding:.9rem; border:1px solid var(--line);
           border-radius:10px; background:var(--panel); }
"""


def _esc(value: Any) -> str:
    return _html.escape(str(value if value is not None else ""), quote=True)


def render_animation_html(
    animation: AnimationSpec,
    *,
    title: str | None = None,
    language: str | None = None,
) -> str:
    """Render one animation as a standalone, self-contained page."""
    if not animation.steps:
        raise ExportError(
            f"animation {animation.id!r} has no annotated steps, so there is nothing to "
            "explain; exporting it would strip the layer that makes it teachable",
            key="errors.export",
        )
    if animation.figure is None:
        raise ExportError(f"animation {animation.id!r} carries no figure", key="errors.export")

    tr = get_translator(language)
    lang = getattr(tr, "language", language or "en")
    badge = badge_for(animation.evidence)
    label = tr.t(badge.label_key, animation.evidence.value.replace("_", " ").title())
    caveat = tr.t(badge.caveat_key, "")

    labels = {
        "what_you_see": tr.t("common.animation_col.what_you_see", "What you see"),
        "what_changed": tr.t("common.animation_col.what_changed", "What changed"),
        "why": tr.t("common.animation_col.why", "Why"),
        "interpretation": tr.t("common.animation_col.interpretation", "How to read it"),
        "conclusion": tr.t("common.animation_col.conclusion", "What to conclude"),
        "warning": tr.t("common.animation_col.warning", "Warning"),
        "step_of": tr.t("proofs.ui.step_of", "Step {current} of {total}"),
        "previous": tr.t("common.previous", "Previous"),
        "next": tr.t("common.next", "Next"),
        "play": tr.t("common.play", "Play"),
        "pause": tr.t("common.pause", "Pause"),
    }

    steps = [
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
            "math": s.math,
            "outputs": {k: str(v) for k, v in (s.outputs or {}).items()},
        }
        for s in animation.steps
    ]

    payload = json_dumps(
        {
            # The page provides its own controls, tied to the commentary.
            "figure": figure_to_json(strip_plotly_transport(animation.figure)),
            "steps": steps,
            "labels": labels,
            "duration": int(animation.frame_duration_ms),
            "loop": bool(animation.loop),
        }
    )

    heading = title or animation.id
    body = [
        f"<h1>{_esc(heading)}</h1>",
        f'<p><span class="badge">{badge.icon} {_esc(label)}</span></p>',
    ]
    if caveat:
        body.append(f'<p class="caveat">{_esc(caveat)}</p>')
    if animation.purpose:
        body.append(f"<p>{_esc(animation.purpose)}</p>")
    body.append(
        '<div class="stage"><div id="figure"></div>'
        '<div class="notes"><h2 id="step-title"></h2>'
        '<p class="muted" id="step-counter"></p><dl id="step-notes"></dl></div></div>'
        '<div class="controls">'
        '<button id="prev"></button><button id="play"></button><button id="next"></button>'
        '<input type="range" id="slider" min="0" value="0" step="1">'
        "</div>"
    )
    if animation.summary:
        body.append(f'<div class="summary">{_esc(animation.summary)}</div>')

    script = f"""
<script src="{PLOTLY_CDN}"></script>
<script>
const VM = {payload};
const stepEls = {{
  title: document.getElementById('step-title'),
  counter: document.getElementById('step-counter'),
  notes: document.getElementById('step-notes'),
}};
const slider = document.getElementById('slider');
slider.max = String(VM.steps.length - 1);
let index = 0, timer = null;

function frames() {{
  const fig = VM.figure;
  return (fig.frames && fig.frames.length) ? fig.frames : null;
}}

function draw() {{
  const step = VM.steps[index];
  stepEls.title.textContent = step.title || '';
  stepEls.counter.textContent = VM.labels.step_of
      .replace('{{current}}', index + 1).replace('{{total}}', VM.steps.length);
  const rows = [
    ['what_you_see', step.what_you_see], ['what_changed', step.what_changed],
    ['why', step.why], ['interpretation', step.interpretation],
    ['conclusion', step.conclusion], ['warning', step.warning],
  ];
  stepEls.notes.innerHTML = rows
    .filter(([, value]) => value)
    .map(([key, value]) => {{
      const cls = key === 'warning' ? ' class="warn"' : '';
      const dt = document.createElement('dt');
      dt.textContent = VM.labels[key] || key;
      const dd = document.createElement('dd');
      dd.textContent = value;
      return `<dt>${{dt.innerHTML}}</dt><dd${{cls}}>${{dd.innerHTML}}</dd>`;
    }})
    .join('');
  slider.value = String(index);
  const fs = frames();
  if (window.Plotly && fs) {{
    const target = Math.min(step.frame, fs.length - 1);
    Plotly.animate('figure', [fs[target].name || String(target)],
      {{mode: 'immediate', frame: {{duration: 0, redraw: true}},
        transition: {{duration: 0}}}});
  }}
}}

function go(delta) {{
  index = (index + delta + VM.steps.length) % VM.steps.length;
  draw();
}}

document.getElementById('prev').textContent = VM.labels.previous;
document.getElementById('next').textContent = VM.labels.next;
const playBtn = document.getElementById('play');
playBtn.textContent = VM.labels.play;
document.getElementById('prev').onclick = () => {{ stop(); go(-1); }};
document.getElementById('next').onclick = () => {{ stop(); go(1); }};
slider.oninput = () => {{ stop(); index = Number(slider.value); draw(); }};

function stop() {{
  if (timer) {{ clearInterval(timer); timer = null; playBtn.textContent = VM.labels.play; }}
}}
playBtn.onclick = () => {{
  if (timer) {{ stop(); return; }}
  playBtn.textContent = VM.labels.pause;
  timer = setInterval(() => {{
    if (!VM.loop && index === VM.steps.length - 1) {{ stop(); return; }}
    go(1);
  }}, VM.duration);
}};

if (window.Plotly) {{
  Plotly.newPlot('figure', VM.figure.data, VM.figure.layout,
                 {{displaylogo: false, responsive: true}}).then(draw);
}} else {{
  draw();
}}
</script>
"""
    return (
        f'<!doctype html>\n<html lang="{lang}" dir="{direction(lang)}">\n<head>\n'
        f'<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(heading)}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n<main>\n"
        + "\n".join(body)
        + f"\n</main>\n{script}\n</body>\n</html>\n"
    )


def export_animation(animation: AnimationSpec, path: str | Path, **kwargs: Any) -> Path:
    """Write a standalone animation page, explanation layer included."""
    path = Path(path)
    if path.suffix.lower() != ".html":
        raise ExportError(
            f"animations export to .html, not {path.suffix!r}. A static format would drop "
            "the per-frame explanation; use export_animation_frames for images.",
            key="errors.export",
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_animation_html(animation, **kwargs), encoding="utf-8")
    return path


def export_animation_frames(
    animation: AnimationSpec,
    directory: str | Path,
    *,
    suffix: str = ".png",
    **kwargs: Any,
) -> list[Path]:
    """Write one image per frame plus a JSON file holding the commentary.

    The commentary file is not optional: images alone would lose the layer that
    makes the animation teachable, so it is always written beside them.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if animation.figure is None:
        raise ExportError(f"animation {animation.id!r} carries no figure", key="errors.export")

    written: list[Path] = []
    figure = animation.figure
    frames = list(getattr(figure, "frames", []) or [])
    for position, step in enumerate(animation.steps):
        target = directory / f"{animation.id}-{position:03d}{suffix}"
        snapshot = figure
        if frames:
            index = min(step.frame, len(frames) - 1)
            snapshot = _figure_at_frame(figure, frames[index])
        written.append(export_figure(snapshot, target, **kwargs))

    notes = directory / f"{animation.id}-explanation.json"
    notes.write_text(
        json.dumps(
            {
                "id": animation.id,
                "purpose": animation.purpose,
                "summary": animation.summary,
                "evidence": animation.evidence.value,
                "steps": [
                    {
                        "image": written[i].name,
                        "title": s.title,
                        "what_you_see": s.what_you_see,
                        "what_changed": s.what_changed,
                        "why": s.why,
                        "interpretation": s.interpretation,
                        "conclusion": s.conclusion,
                        "warning": s.warning,
                    }
                    for i, s in enumerate(animation.steps)
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    written.append(notes)
    return written


def _figure_at_frame(figure: Any, frame: Any) -> Any:
    """A copy of ``figure`` showing the data of one frame."""
    import copy

    snapshot = copy.deepcopy(figure)
    data = getattr(frame, "data", None)
    if data:
        for trace_index, trace in enumerate(data):
            if trace_index < len(snapshot.data):
                snapshot.data[trace_index].update(trace)
    layout = getattr(frame, "layout", None)
    if layout:
        snapshot.update_layout(layout)
    snapshot.frames = []
    return snapshot
