"""The proof step navigator (blueprint section 48.3).

A proof is read one licensed move at a time, so the viewer walks the steps and
shows, for the current one, the claim it advances and *why* that move is
allowed. Two things stay on screen throughout, because a proof read without
them is a proof misread: the assumptions in force, and the statement of what
the proof does not establish.

Numerical checks are shown in their own section, labelled as checks of the
code - never as evidence for the theorem.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

__all__ = ["ProofViewer", "render_proof"]


class ProofViewer:
    """Step-by-step navigation over a rendered proof."""

    def __init__(self, proof: Any, rendered: Any, *, translator: Any = None,
                 reduced_motion: bool = False) -> None:
        self.proof = proof
        self.rendered = rendered
        self.translator = translator
        self.reduced_motion = reduced_motion
        self.index = 0
        self.show_algebra = True
        self.timer: Any = None
        self._body: Any = None
        self._counter: Any = None
        self._play_button: Any = None

    def t(self, key: str, default: str, **params: Any) -> str:
        if self.translator is None:
            return default.format(**params) if params else default
        return self.translator.t(key, default, **params)

    @property
    def steps(self) -> list[Any]:
        return list(self.rendered.steps)

    @property
    def step(self) -> Any:
        return self.steps[self.index]

    # -- rendering --------------------------------------------------------

    def render(self) -> None:
        r = self.rendered
        with ui.column().classes("w-full gap-3"):
            ui.label(r.title).classes("text-xl font-semibold")
            with ui.row().classes("items-center gap-2 flex-wrap"):
                ui.html(
                    f'<span class="vm-badge">{r.evidence_icon} {_attr(r.evidence_label)}</span>'
                )
                if r.evidence_caveat:
                    ui.label(r.evidence_caveat).classes("text-xs vm-muted")

            if r.intuition:
                with ui.card().classes("w-full vm-surface").props("flat bordered"):
                    ui.label(self.t("proofs.ui.intuition", "In plain language")).classes(
                        "text-xs vm-muted"
                    )
                    ui.label(r.intuition).classes("text-sm")

            with ui.card().classes("w-full vm-surface").props("flat bordered"):
                ui.label(self.t("proofs.ui.claim", "Claim")).classes("text-xs vm-muted")
                ui.label(r.claim).classes("font-medium")

            self._render_assumptions()

            with ui.row().classes("w-full items-start gap-4 no-wrap"):
                with ui.column().classes("grow min-w-0 gap-2"):
                    self._counter = ui.label().classes("text-xs vm-muted")
                    self._body = ui.column().classes("w-full gap-2")
                with ui.column().classes("vm-surface p-3 gap-1").style(
                    "flex: 0 0 18rem; max-width: 18rem;"
                ):
                    ui.label(self.t("proofs.ui.steps", "Steps")).classes(
                        "text-xs vm-muted"
                    )
                    for position, step in enumerate(self.steps):
                        ui.button(
                            f"{step.number}. {step.statement[:40]}",
                            on_click=lambda _=None, i=position: self.go(i),
                        ).props("flat dense no-caps align=left").classes(
                            "w-full text-xs vm-focusable"
                        )

            self._render_controls()
            self._render_conclusion()
            self._render_checks()
            self.update()

    def _render_assumptions(self) -> None:
        if not self.rendered.assumptions:
            return
        with ui.expansion(
            self.t("proofs.ui.assumptions", "Assumptions in force"), value=True
        ).classes("w-full vm-surface"):
            for assumption in self.rendered.assumptions:
                tag = self.t(
                    "proofs.ui.assumption_essential" if assumption.essential
                    else "proofs.ui.assumption_relaxable",
                    "essential" if assumption.essential else "can be relaxed",
                )
                with ui.row().classes("w-full items-baseline gap-2"):
                    ui.label(f"({assumption.id})").classes("text-xs vm-muted vm-ltr")
                    ui.label(assumption.statement).classes("text-sm grow")
                    ui.label(tag).classes("text-xs vm-muted")
                if assumption.if_violated:
                    ui.label(
                        f"{self.t('proofs.ui.if_violated', 'If this fails')}: "
                        f"{assumption.if_violated}"
                    ).classes("text-xs vm-warning ms-4")

    def _render_controls(self) -> None:
        with ui.row().classes("w-full items-center gap-2 flex-wrap"):
            ui.button(icon="chevron_left", on_click=self.previous).props(
                "flat dense"
            ).classes("vm-focusable").tooltip(self.t("proofs.ui.previous", "Previous step"))
            self._play_button = ui.button(icon="play_arrow", on_click=self.toggle_play).props(
                "flat dense"
            ).classes("vm-focusable")
            ui.button(icon="chevron_right", on_click=self.next).props(
                "flat dense"
            ).classes("vm-focusable").tooltip(self.t("proofs.ui.next", "Next step"))
            ui.button(icon="restart_alt", on_click=self.reset).props(
                "flat dense"
            ).classes("vm-focusable").tooltip(self.t("proofs.ui.reset", "Restart proof"))
            ui.switch(
                self.t("proofs.ui.show_algebra", "Show algebra"),
                value=True,
                on_change=self._toggle_algebra,
            ).classes("vm-focusable")

    def _toggle_algebra(self, event: Any) -> None:
        self.show_algebra = bool(getattr(event, "value", True))
        self.update()

    def _render_conclusion(self) -> None:
        r = self.rendered
        with ui.card().classes("w-full vm-surface").props("flat bordered"):
            ui.label(self.t("proofs.ui.conclusion", "Conclusion")).classes(
                "text-xs vm-muted"
            )
            ui.label(r.conclusion).classes("text-sm")
        # Always visible, never behind a toggle: the scope of a proof is part of
        # the proof.
        with ui.card().classes("w-full vm-surface").props("flat bordered"):
            ui.label(
                self.t("proofs.ui.limitations", "What this does NOT establish")
            ).classes("text-xs vm-warning font-semibold")
            ui.label(r.limitations).classes("text-sm")

    def _render_checks(self) -> None:
        if not self.proof.checks:
            return
        with ui.expansion(
            self.t("proofs.ui.checks", "Numerical checks"), value=False
        ).classes("w-full vm-surface"):
            ui.label(self.t(
                "proofs.ui.checks_note",
                "These checks confirm that the code implements the proved identity. "
                "They are numerical demonstrations, not evidence for the theorem.",
            )).classes("text-xs vm-muted")
            container = ui.column().classes("w-full gap-1")

            def run() -> None:
                container.clear()
                with container:
                    for outcome in self.proof.run_checks():
                        word = self.t(
                            "proofs.ui.check_passed" if outcome.passed
                            else "proofs.ui.check_failed",
                            "passed" if outcome.passed else "failed",
                        )
                        colour = "vm-positive" if outcome.passed else "vm-negative"
                        with ui.row().classes("w-full items-baseline gap-2"):
                            ui.icon("check" if outcome.passed else "close").classes(colour)
                            ui.label(word).classes(f"text-xs {colour}")
                            ui.label(outcome.detail).classes("text-xs vm-muted grow")

            ui.button(
                self.t("actions.run", "Run"), on_click=run
            ).props("flat dense no-caps").classes("vm-focusable")

    # -- navigation -------------------------------------------------------

    def go(self, index: int) -> None:
        self.index = max(0, min(index, len(self.steps) - 1))
        self.update()

    def next(self) -> None:
        self.stop()
        self.go((self.index + 1) % len(self.steps))

    def previous(self) -> None:
        self.stop()
        self.go((self.index - 1) % len(self.steps))

    def reset(self) -> None:
        self.stop()
        self.go(0)

    def toggle_play(self) -> None:
        if self.timer is not None:
            self.stop()
            return
        if self.reduced_motion:
            ui.notify(self.t(
                "ui.reduced_motion_blocked",
                "Autoplay is off because reduced motion is enabled.",
            ), type="info")
            return
        self.timer = ui.timer(3.0, self._advance)
        if self._play_button is not None:
            self._play_button.props("icon=pause")

    def stop(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        if self._play_button is not None:
            self._play_button.props("icon=play_arrow")

    def _advance(self) -> None:
        if self.index >= len(self.steps) - 1:
            self.stop()
            return
        self.go(self.index + 1)

    # -- redraw -----------------------------------------------------------

    def update(self) -> None:
        step = self.step
        if self._counter is not None:
            self._counter.set_text(self.t(
                "proofs.ui.step_of", "Step {current} of {total}",
                current=step.number, total=len(self.steps),
            ))
        if self._body is None:
            return
        self._body.clear()
        with self._body:
            with ui.row().classes("items-center gap-2"):
                ui.label(step.kind_label).classes("text-xs vm-badge")
                ui.label(step.statement).classes("font-medium grow")
            if step.equation and self.show_algebra:
                ui.markdown(f"$${step.equation}$$", extras=["latex"]).classes(
                    "w-full vm-ltr"
                )
            if step.justification:
                ui.label(self.t("proofs.ui.why", "Why this step is allowed")).classes(
                    "text-xs vm-muted mt-2"
                )
                ui.label(step.justification).classes("text-sm")
            if step.uses:
                ui.label(
                    f"{self.t('proofs.ui.uses', 'Uses')}: {', '.join(step.uses)}"
                ).classes("text-xs vm-muted vm-ltr")


def render_proof(proof: Any, *, session: Any = None, translator: Any = None) -> ProofViewer:
    """Render one proof with its navigator."""
    tr = translator or (session.translator() if session is not None else None)
    rendered = proof.render(tr)
    viewer = ProofViewer(
        proof, rendered, translator=tr,
        reduced_motion=bool(getattr(session, "reduced_motion", False)),
    )
    viewer.render()
    return viewer


def _attr(text: Any) -> str:
    return str(text or "").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
