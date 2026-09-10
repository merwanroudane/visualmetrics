"""Teaching material attached to the concepts (blueprint sections 62 and 63).

Three pieces, all data rather than prose buried in a lab:

``objectives``      what a learner should be able to *do*, and where to check it;
``misconceptions``  the plausible wrong belief, why it is wrong, and the fix;
``quiz``            self-check questions that always explain their answer.

Nothing here grades anyone, and nothing is stored.
"""

from .misconceptions import MISCONCEPTIONS, Misconception
from .misconceptions import for_concept as misconceptions_for
from .misconceptions import get as misconception
from .objectives import OBJECTIVES, Objective, all_objectives, coverage
from .objectives import for_concept as objectives_for
from .quiz import (
    QUESTIONS,
    Choice,
    Question,
    QuestionKind,
    Quiz,
    build_quiz,
    quiz_for,
    score,
)

__all__ = [
    "Objective",
    "OBJECTIVES",
    "objectives_for",
    "all_objectives",
    "coverage",
    "Misconception",
    "MISCONCEPTIONS",
    "misconception",
    "misconceptions_for",
    "Question",
    "QuestionKind",
    "Choice",
    "Quiz",
    "QUESTIONS",
    "quiz_for",
    "build_quiz",
    "score",
]
