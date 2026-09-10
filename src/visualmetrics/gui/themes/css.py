"""Turning a figure theme into page styling (blueprint section 36.6).

The seven themes already exist for the figures. The page must agree with them,
otherwise a publication-theme chart sits on a classroom-theme background and
the export looks wrong. So the page pulls its colours from the very same
:class:`~visualmetrics.visuals.themes.palette.Theme` object the figures use,
rather than keeping a second palette that can drift.

No NiceGUI here either: this produces a CSS string, which is testable.
"""

from __future__ import annotations

from ...visuals.themes.palette import Theme, get_theme

__all__ = ["theme_css", "theme_variables", "is_dark", "font_stack", "ARABIC_FONT_STACK"]

ARABIC_FONT_STACK = (
    '"Noto Naskh Arabic", "Amiri", "Segoe UI", "Tahoma", system-ui, sans-serif'
)
LATIN_FONT_STACK = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'


def is_dark(theme: str | Theme | None) -> bool:
    return bool(get_theme(theme).dark)


def font_stack(language: str) -> str:
    """Arabic needs a font with real Naskh shaping, not a Latin fallback."""
    return ARABIC_FONT_STACK if language == "ar" else LATIN_FONT_STACK


def theme_variables(theme: str | Theme | None) -> dict[str, str]:
    """The CSS custom properties for one theme.

    Every semantic role becomes a variable, so a component asks for
    ``var(--vm-warning)`` and automatically follows the theme - including
    high-contrast and colourblind-safe variants.
    """
    resolved = get_theme(theme)
    variables = {f"--vm-{role.replace('_', '-')}": value
                 for role, value in resolved.roles.items()}
    background = resolved.roles.get("background", "#ffffff")
    surface = resolved.roles.get("surface", background)
    foreground = resolved.roles.get("foreground", "#111111")
    variables.update({
        "--vm-bg": background,
        "--vm-surface": surface,
        "--vm-ink": foreground,
        "--vm-line": resolved.roles.get("grid", "#dddddd"),
        "--vm-font-size": f"{resolved.font_size}px",
        "--vm-font-scale": str(resolved.font_scale),
    })
    return variables


def theme_css(
    theme: str | Theme | None,
    *,
    language: str = "en",
    reduced_motion: bool = False,
    presentation: bool = False,
) -> str:
    """The full stylesheet for one session.

    ``reduced_motion`` is honoured as a hard rule, not a hint: an animation
    that ignores the preference is an accessibility failure, so transitions are
    disabled outright rather than merely shortened.
    """
    resolved = get_theme(theme)
    variables = theme_variables(resolved)
    scale = 1.35 if presentation else 1.0
    declarations = "\n  ".join(f"{name}: {value};" for name, value in variables.items())

    css = f""":root {{
  {declarations}
  --vm-font: {font_stack(language)};
  --vm-scale: {scale};
  --vm-radius: 10px;
}}
body, .nicegui-content {{
  background: var(--vm-bg);
  color: var(--vm-ink);
  font-family: var(--vm-font);
  font-size: calc(var(--vm-font-size) * var(--vm-scale));
}}
.vm-surface {{
  background: var(--vm-surface);
  border: 1px solid var(--vm-line);
  border-radius: var(--vm-radius);
}}
.vm-header {{
  position: sticky; top: 0; z-index: 10;
  border-radius: 0; border-inline: none; border-top: none;
  backdrop-filter: blur(6px);
}}
.vm-muted {{ color: var(--vm-muted); }}
.vm-warning {{ color: var(--vm-warning); }}
.vm-negative {{ color: var(--vm-negative); }}
.vm-positive {{ color: var(--vm-positive); }}
.vm-badge {{
  display: inline-flex; align-items: center; gap: .35rem;
  padding: .15rem .6rem; border-radius: 999px;
  border: 1px solid var(--vm-primary); color: var(--vm-primary);
  font-size: .8em; font-weight: 600; white-space: nowrap;
}}
.vm-badge[data-proof="false"] {{ border-color: var(--vm-info); color: var(--vm-info); }}
.vm-numeric {{ font-variant-numeric: tabular-nums; }}
.vm-code {{
  direction: ltr; text-align: left;
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
}}
.vm-focusable:focus-visible {{
  outline: 3px solid var(--vm-highlight);
  outline-offset: 2px;
}}
"""

    if language == "ar":
        css += """
body { direction: rtl; }
.vm-ltr { direction: ltr; text-align: left; unicode-bidi: isolate; }
.q-field__label, .q-item__label { text-align: right; }
"""

    if reduced_motion:
        css += """
*, *::before, *::after {
  animation-duration: 0s !important;
  animation-iteration-count: 1 !important;
  transition-duration: 0s !important;
  scroll-behavior: auto !important;
}
"""
    else:
        css += """
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0s !important;
    transition-duration: 0s !important;
  }
}
"""

    if presentation:
        css += """
.vm-hide-in-presentation { display: none !important; }
.vm-surface { box-shadow: none; }
h1, .vm-title { font-size: 1.9em; }
"""

    if resolved.name == "high_contrast":
        css += """
.vm-surface { border-width: 2px; }
a, button { text-decoration-thickness: 2px; }
"""

    return css
