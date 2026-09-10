"""The self-check component (blueprint section 62).

Two behaviours matter more than the questions themselves:

* the explanation appears after every answer, right *and* wrong, and a wrong
  answer also names the correct option and says why the distractor is tempting;
* nothing is graded or stored. The tally is shown with that caveat attached,
  and a missed question links to the lab scenario that settles it, so the quiz
  sends the learner back to the evidence rather than to a score.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

__all__ = ["render_quiz"]


def render_quiz(concept_id: str | None, session: Any, *, count: int = 3) -> None:
    """Render a small self-check for one concept."""
    from ...education.quiz import build_quiz

    tr = session.translator()
    quiz = build_quiz(concept_id, count=count, seed=session.seed)
    if not quiz.questions:
        ui.label(tr.t("quiz.none", "No self-check questions for this concept yet.")).classes(
            "text-sm vm-muted"
        )
        return

    answers: dict[str, int] = {}
    ui.label(tr.t(
        "quiz.caveat",
        "A self-check, not an assessment: nothing is graded or stored, and every question "
        "explains its answer.",
    )).classes("text-xs vm-muted")

    for question in quiz.questions:
        order = quiz.orders[question.id]
        with ui.card().classes("w-full vm-surface my-1").props("flat bordered"):
            ui.label(question.prompt).classes("font-medium")
            feedback = ui.column().classes("w-full gap-1")

            def answer(index: int, q: Any = question, o: Any = order,
                       box: Any = feedback) -> None:
                answers[q.id] = index
                correct, explanation = q.check(index, o)
                box.clear()
                with box:
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("check_circle" if correct else "cancel").classes(
                            "vm-positive" if correct else "vm-negative"
                        )
                        ui.label(tr.t(
                            "quiz.correct" if correct else "quiz.incorrect",
                            "Correct" if correct else "Not quite",
                        )).classes(
                            "text-sm " + ("vm-positive" if correct else "vm-negative")
                        )
                    ui.label(explanation).classes("text-sm")
                    if q.try_it:
                        target, scenario = q.try_it
                        ui.link(
                            tr.t("quiz.try_it", "Check it in the lab"),
                            f"/lab/{target}?scenario={scenario}",
                        ).classes("text-sm vm-focusable")

            for index, choice in enumerate(order):
                ui.button(
                    choice.text, on_click=lambda _=None, i=index: answer(i)
                ).props("flat no-caps align=left").classes("w-full vm-focusable")

    def show_score() -> None:
        from ...education.quiz import score as score_quiz

        outcome = score_quiz(quiz, answers)
        ui.notify(
            f"{outcome['correct']}/{outcome['total']} - {outcome['caveat']}",
            type="info",
            multi_line=True,
        )

    ui.button(
        tr.t("quiz.summary", "Show summary"), on_click=show_score
    ).props("flat dense no-caps").classes("vm-focusable")
