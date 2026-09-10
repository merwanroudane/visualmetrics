"""Objectives, misconceptions and the self-check quiz."""

from __future__ import annotations

import pytest

import visualmetrics as vm
from visualmetrics.catalog.builtin import CATALOG
from visualmetrics.education import (
    MISCONCEPTIONS,
    QUESTIONS,
    all_objectives,
    build_quiz,
    coverage,
    misconception,
    misconceptions_for,
    objectives_for,
    quiz_for,
    score,
)

CONCEPT_IDS = {s.id for s in CATALOG}
IMPLEMENTED = [s.id for s in CATALOG if s.status.is_implemented]


class TestObjectives:
    def test_every_implemented_concept_has_objectives(self):
        assert coverage()["missing"] == [], "a lab with no objectives teaches nothing stated"

    def test_objectives_say_how_to_check_them(self):
        for objective in all_objectives():
            assert objective.statement.strip()
            assert objective.check.strip(), (
                f"{objective.id} states a goal with no way to verify it in the lab"
            )

    def test_objectives_are_actions_not_vague_understanding(self):
        """'Understand X' is not checkable; the objectives must start with a verb."""
        vague = [o.id for o in all_objectives()
                 if o.statement.lower().startswith(("understand", "learn about", "know about"))]
        assert not vague, f"vague objectives: {vague[:5]}"

    def test_objectives_belong_to_real_concepts(self):
        for objective in all_objectives():
            assert objective.concept_id in CONCEPT_IDS

    def test_lookup_by_concept(self):
        assert len(objectives_for("inference.power")) >= 3
        assert objectives_for("not.a.concept") == ()


class TestMisconceptions:
    def test_every_entry_is_complete(self):
        for entry in MISCONCEPTIONS.values():
            assert entry.claim and entry.why_wrong and entry.correct
            assert entry.concept_ids, f"{entry.id} is attached to no concept"

    def test_the_correction_is_not_a_restatement(self):
        for entry in MISCONCEPTIONS.values():
            assert entry.correct.lower() != entry.claim.lower()
            assert len(entry.why_wrong) > 60, (
                f"{entry.id}: 'why wrong' must give the mechanism, not just a verdict"
            )

    def test_entries_point_at_real_concepts(self):
        for entry in MISCONCEPTIONS.values():
            for concept_id in entry.concept_ids:
                assert concept_id in CONCEPT_IDS, f"{entry.id} names unknown {concept_id}"

    def test_demos_point_at_real_labs(self):
        for entry in MISCONCEPTIONS.values():
            if entry.demo:
                assert entry.demo[0] in CONCEPT_IDS

    def test_every_id_referenced_by_a_lab_exists(self):
        for concept_id in IMPLEMENTED:
            for name in vm.registry.lab(concept_id).spec.misconceptions:
                assert name in MISCONCEPTIONS, f"{concept_id} references missing {name}"

    def test_lookup_collects_from_both_directions(self):
        found = {m.id for m in misconceptions_for("inference.power")}
        assert "observed_power" in found  # declared on the spec
        assert "p_measures_effect" in found  # declared on the misconception

    def test_unknown_id_raises(self):
        with pytest.raises(KeyError):
            misconception("no_such_misconception")


class TestQuiz:
    def test_every_question_has_exactly_one_answer(self):
        for question in QUESTIONS:
            assert sum(1 for c in question.choices if c.correct) == 1

    def test_every_choice_explains_itself(self):
        """A distractor with no explanation wastes the teachable moment."""
        for question in QUESTIONS:
            for choice in question.choices:
                assert choice.why.strip(), f"{question.id}: a choice has no explanation"

    def test_questions_belong_to_real_concepts(self):
        for question in QUESTIONS:
            assert question.concept_id in CONCEPT_IDS

    def test_try_it_points_at_a_real_lab(self):
        for question in QUESTIONS:
            if question.try_it:
                assert question.try_it[0] in CONCEPT_IDS

    def test_referenced_misconceptions_exist(self):
        for question in QUESTIONS:
            if question.misconception_id:
                assert question.misconception_id in MISCONCEPTIONS

    def test_a_wrong_answer_names_the_right_one(self):
        question = QUESTIONS[0]
        wrong = next(i for i, c in enumerate(question.choices) if not c.correct)
        correct, explanation = question.check(wrong)
        assert not correct
        assert question.answer.text[:20] in explanation

    def test_a_right_answer_still_explains(self):
        question = QUESTIONS[0]
        right = next(i for i, c in enumerate(question.choices) if c.correct)
        correct, explanation = question.check(right)
        assert correct and len(explanation) > 20

    def test_an_out_of_range_answer_is_handled(self):
        correct, _ = QUESTIONS[0].check(99)
        assert not correct

    def test_a_quiz_is_reproducible_from_its_seed(self):
        first = build_quiz(seed=11, count=4)
        second = build_quiz(seed=11, count=4)
        assert [q.id for q in first.questions] == [q.id for q in second.questions]
        assert [c.text for c in first.orders[first.questions[0].id]] == [
            c.text for c in second.orders[second.questions[0].id]
        ]

    def test_different_seeds_draw_differently(self):
        a = [q.id for q in build_quiz(seed=1, count=5).questions]
        b = [q.id for q in build_quiz(seed=2, count=5).questions]
        assert a != b

    def test_option_order_is_shuffled_so_position_carries_no_hint(self):
        positions = set()
        for seed in range(12):
            quiz = build_quiz("inference.hypothesis_testing", seed=seed, count=2)
            for question in quiz.questions:
                order = quiz.orders[question.id]
                positions.add(next(i for i, c in enumerate(order) if c.correct))
        assert len(positions) > 1, "the correct option always sat in the same place"

    def test_scoring_reports_a_caveat_and_never_a_grade(self):
        quiz = build_quiz(seed=3, count=3)
        outcome = score(quiz, {})
        assert outcome["total"] == 3
        assert "not an assessment" in outcome["caveat"]
        assert not quiz.is_assessment
        assert "grade" not in str(outcome).lower().replace("not a grade", "")

    def test_scoring_points_back_at_the_misconception(self):
        quiz = build_quiz("inference.clt", seed=5, count=1)
        question = quiz.questions[0]
        order = quiz.orders[question.id]
        wrong = next(i for i, c in enumerate(order) if not c.correct)
        outcome = score(quiz, {question.id: wrong})
        if question.misconception_id:
            assert question.misconception_id in outcome["misconceptions_to_revisit"]

    def test_quiz_for_a_concept_without_questions_is_empty_not_faked(self):
        assert quiz_for("spatial.autocorrelation") == () or all(
            q.concept_id == "spatial.autocorrelation" for q in quiz_for("spatial.autocorrelation")
        )

    def test_a_question_with_two_answers_is_rejected(self):
        from visualmetrics.education.quiz import Choice, Question, QuestionKind

        with pytest.raises(ValueError, match="exactly one correct choice"):
            Question(
                id="bad", concept_id="inference.power", kind=QuestionKind.MULTIPLE_CHOICE,
                prompt="?",
                choices=(Choice("a", True, why="x"), Choice("b", True, why="y")),
            )

    def test_a_question_with_an_unexplained_choice_is_rejected(self):
        from visualmetrics.education.quiz import Choice, Question, QuestionKind

        with pytest.raises(ValueError, match="teaches nothing"):
            Question(
                id="bad", concept_id="inference.power", kind=QuestionKind.MULTIPLE_CHOICE,
                prompt="?",
                choices=(Choice("a", True, why="x"), Choice("b", why="")),
            )


class TestPublicApi:
    def test_helpers_are_exposed(self):
        assert vm.objectives("inference.power")
        assert vm.misconceptions("inference.power")
        assert len(vm.quiz("inference.hypothesis_testing", count=2)) >= 1
