"""The animation player (blueprint section 36.3.1).

The blueprint's hardest rule lives here: an animation without an explanation
layer is not allowed. Every frame states what you see, what changed, why, how
to read it, and where relevant what to conclude and what to be careful about -
and the player shows that text *beside the figure*, at the same time, not in a
caption underneath that nobody reads.

Two more consequences of that rule are enforced here:

* if a session asks for reduced motion, the player does not autoplay and
  presents the frames as steps you advance yourself;
* an animation that arrives without annotated steps is refused with a visible
  message rather than being played silently.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from ...core.evidence import badge_for
from ...visuals.plotly.primitives import strip_plotly_transport

__all__ = ["render_animation", "AnimationPlayer"]


class AnimationPlayer:
    """Figure, frame commentary and transport controls, kept in step.

    The player owns one index. Everything else - the figure, the commentary,
    the counter - is redrawn from it, so the picture and the words can never
    disagree about which frame you are looking at.
    """

    def __init__(self, animation: Any, *, translator: Any = None,
                 reduced_motion: bool = False) -> None:
        self.animation = animation
        self.translator = translator
        self.reduced_motion = reduced_motion
        self.index = 0
        self.timer: Any = None
        self._figure: Any = None
        self._counter: Any = None
        self._notes: Any = None
        self._play_button: Any = None

    # -- text -------------------------------------------------------------

    def t(self, key: str, default: str, **params: Any) -> str:
        if self.translator is None:
            return default.format(**params) if params else default
        return self.translator.t(key, default, **params)

    @property
    def steps(self) -> list[Any]:
        return list(self.animation.steps)

    @property
    def step(self) -> Any:
        return self.steps[self.index]

    # -- building ---------------------------------------------------------

    def render(self) -> None:
        animation = self.animation
        if not animation.steps:
            # Refused loudly: playing an unexplained animation would break the
            # rule this whole module exists to keep.
            ui.label(self.t(
                "ui.animation_unexplained",
                "This animation carries no per-frame explanation, so it is not played.",
            )).classes("vm-warning")
            return

        badge = badge_for(animation.evidence)
        with ui.column().classes("w-full gap-2"):
            with ui.row().classes("items-center gap-2 flex-wrap"):
                ui.html(
                    f'<span class="vm-badge" '
                    f'data-proof="{str(animation.evidence.is_proof).lower()}">'
                    f'{badge.icon} {self.t(badge.label_key, animation.evidence.value)}</span>'
                )
                if animation.purpose:
                    ui.label(animation.purpose).classes("text-sm grow")

            with ui.row().classes("w-full items-start gap-4 no-wrap"):
                with ui.column().classes("grow min-w-0"):
                    # The player owns the frames, so Plotly's own transport is
                    # removed: using it would move the figure while the
                    # commentary beside it stayed on the previous frame.
                    self._figure = ui.plotly(
                        strip_plotly_transport(animation.figure)
                    ).classes("w-full")
                with ui.column().classes("vm-surface p-3 gap-1").style(
                    "flex: 0 0 22rem; max-width: 22rem;"
                ):
                    self._counter = ui.label().classes("text-xs vm-muted")
                    self._notes = ui.column().classes("gap-1 w-full")

            self._render_controls()
            if animation.summary:
                with ui.card().classes("w-full vm-surface").props("flat bordered"):
                    ui.label(self.t("ui.animation_summary", "In summary")).classes(
                        "text-xs vm-muted"
                    )
                    ui.label(animation.summary).classes("text-sm")
            self.update()

    def _render_controls(self) -> None:
        with ui.row().classes("w-full items-center gap-2 flex-wrap"):
            ui.button(icon="first_page", on_click=self.first).props(
                "flat dense"
            ).classes("vm-focusable").tooltip(self.t("actions.first", "First frame"))
            ui.button(icon="chevron_left", on_click=self.previous).props(
                "flat dense"
            ).classes("vm-focusable").tooltip(self.t("actions.previous", "Previous"))
            self._play_button = ui.button(
                icon="play_arrow", on_click=self.toggle_play
            ).props("flat dense").classes("vm-focusable")
            self._play_button.tooltip(self.t("actions.play", "Play"))
            ui.button(icon="chevron_right", on_click=self.next).props(
                "flat dense"
            ).classes("vm-focusable").tooltip(self.t("actions.next", "Next"))
            ui.slider(
                min=0, max=max(len(self.steps) - 1, 0), step=1, value=0,
                on_change=lambda e: self.go(int(e.value)),
            ).classes("grow").props("label-always")

        if self.reduced_motion:
            ui.label(self.t(
                "ui.reduced_motion_note",
                "Reduced motion is on: advance the frames yourself; every frame shows "
                "the same evidence as the animation.",
            )).classes("text-xs vm-muted")

    # -- transport --------------------------------------------------------

    def go(self, index: int) -> None:
        if not self.steps:
            return
        self.index = max(0, min(index, len(self.steps) - 1))
        self.update()

    def first(self) -> None:
        self.stop()
        self.go(0)

    def next(self) -> None:
        self.stop()
        self.go((self.index + 1) % len(self.steps))

    def previous(self) -> None:
        self.stop()
        self.go((self.index - 1) % len(self.steps))

    def toggle_play(self) -> None:
        if self.timer is not None:
            self.stop()
            return
        if self.reduced_motion:
            # Honour the preference instead of quietly overriding it.
            ui.notify(self.t(
                "ui.reduced_motion_blocked",
                "Autoplay is off because reduced motion is enabled.",
            ), type="info")
            return
        interval = max(self.animation.frame_duration_ms, 120) / 1000
        self.timer = ui.timer(interval, self._advance)
        if self._play_button is not None:
            self._play_button.props("icon=pause")

    def stop(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        if self._play_button is not None:
            self._play_button.props("icon=play_arrow")

    def _advance(self) -> None:
        if self.index >= len(self.steps) - 1 and not self.animation.loop:
            self.stop()
            return
        self.go((self.index + 1) % len(self.steps))

    # -- redraw -----------------------------------------------------------

    def update(self) -> None:
        step = self.step
        if self._counter is not None:
            self._counter.set_text(self.t(
                "ui.frame_of", "Frame {current} of {total}",
                current=self.index + 1, total=len(self.steps),
            ))
        self._update_figure(step)
        self._update_notes(step)

    def _update_figure(self, step: Any) -> None:
        if self._figure is None:
            return
        figure = self._figure.figure
        frames = list(getattr(figure, "frames", []) or [])
        if not frames:
            return
        frame = frames[min(step.frame, len(frames) - 1)]
        data = getattr(frame, "data", None)
        if data:
            for position, trace in enumerate(data):
                if position < len(figure.data):
                    figure.data[position].update(trace)
        layout = getattr(frame, "layout", None)
        if layout:
            figure.update_layout(layout)
        self._figure.update()

    def _update_notes(self, step: Any) -> None:
        if self._notes is None:
            return
        self._notes.clear()
        rows = (
            ("ui.what_you_see", "What you see", step.what_you_see, ""),
            ("ui.what_changed", "What changed", step.what_changed, ""),
            ("ui.why", "Why", step.why, ""),
            ("ui.how_to_read", "How to read it", step.interpretation, ""),
            ("ui.what_to_conclude", "What to conclude", step.conclusion, ""),
            ("ui.caution", "Careful", step.warning, "vm-warning"),
        )
        with self._notes:
            if step.title:
                ui.label(step.title).classes("font-semibold")
            for key, default, text, extra in rows:
                if not text:
                    continue
                ui.label(self.t(key, default)).classes("text-xs vm-muted mt-1")
                ui.label(text).classes(f"text-sm {extra}".strip())
            if step.math:
                ui.markdown(step.math, extras=["latex"]).classes("text-sm")
            if step.outputs:
                ui.separator().classes("my-1")
                for name, value in step.outputs.items():
                    with ui.row().classes("justify-between w-full gap-2"):
                        ui.label(str(name)).classes("text-xs vm-muted")
                        ui.label(str(value)).classes("text-xs vm-numeric")
            if step.violated_assumptions:
                ui.label(self.t(
                    "ui.violated_here", "Violated in this frame: {items}",
                    items=", ".join(step.violated_assumptions),
                )).classes("text-xs vm-warning mt-1")


def render_animation(animation: Any, *, translator: Any = None,
                     reduced_motion: bool = False) -> AnimationPlayer:
    """Render one animation with its explanation layer."""
    player = AnimationPlayer(
        animation, translator=translator, reduced_motion=reduced_motion
    )
    player.render()
    return player
