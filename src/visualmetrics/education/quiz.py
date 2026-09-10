"""Self-check questions (blueprint section 62).

Design rules taken from the blueprint, and enforced here:

* every question explains its answer, whether you got it right or wrong, and
  the explanation says *why* the distractors are wrong - a question that only
  says "incorrect" teaches nothing;
* questions are drawn with a seeded generator, so a quiz is reproducible and
  can be shared;
* this is a self-check, not an assessment. Nothing is graded, nothing is
  stored, and :func:`score` returns a count with that caveat attached.

Many questions are built directly on a misconception, so the wrong answer a
learner is most likely to pick is a real option rather than an obvious filler.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .misconceptions import MISCONCEPTIONS

__all__ = [
    "QuestionKind",
    "Choice",
    "Question",
    "Quiz",
    "QUESTIONS",
    "quiz_for",
    "build_quiz",
    "score",
]


class QuestionKind(str, Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    TRUE_FALSE = "true_false"
    PREDICT_RESULT = "predict_result"
    SELECT_VIOLATED_ASSUMPTION = "select_violated_assumption"
    CHOOSE_THE_GRAPH = "choose_the_graph"
    IDENTIFY_ROLE = "identify_role"
    COMPARE_BEHAVIOUR = "compare_behaviour"


@dataclass(frozen=True)
class Choice:
    """One option, with the reason it is right or wrong."""

    text: str
    correct: bool = False
    why: str = ""


@dataclass(frozen=True)
class Question:
    id: str
    concept_id: str
    kind: QuestionKind
    prompt: str
    choices: tuple[Choice, ...]
    explanation: str = ""
    level: str = "intermediate"
    misconception_id: str | None = None
    try_it: tuple[str, str] | None = None
    """``(concept_id, scenario_id)`` where the learner can check the answer."""

    def __post_init__(self) -> None:
        correct = [c for c in self.choices if c.correct]
        if len(correct) != 1:
            raise ValueError(
                f"question {self.id!r} must have exactly one correct choice, has {len(correct)}"
            )
        if not all(c.why for c in self.choices):
            raise ValueError(
                f"question {self.id!r} has a choice with no explanation; a quiz that "
                "only says 'wrong' teaches nothing"
            )

    @property
    def answer(self) -> Choice:
        return next(c for c in self.choices if c.correct)

    def shuffled(self, rng: random.Random) -> tuple[Choice, ...]:
        """Options in a reproducible random order, so position carries no hint."""
        options = list(self.choices)
        rng.shuffle(options)
        return tuple(options)

    def check(self, chosen: int, order: tuple[Choice, ...] | None = None) -> tuple[bool, str]:
        """Return whether the pick was right, and the explanation either way."""
        options = order or self.choices
        if not 0 <= chosen < len(options):
            return False, "That is not one of the options."
        picked = options[chosen]
        detail = picked.why
        if not picked.correct:
            detail = f"{detail} The correct answer: {self.answer.text} - {self.answer.why}"
        return picked.correct, f"{detail} {self.explanation}".strip()


@dataclass
class Quiz:
    """A drawn set of questions, reproducible from its seed."""

    questions: tuple[Question, ...]
    seed: int = 42
    orders: dict[str, tuple[Choice, ...]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.questions)

    @property
    def is_assessment(self) -> bool:
        """Always false. This is a self-check; nothing here grades anyone."""
        return False


def _q(**kwargs: Any) -> Question:
    return Question(**kwargs)


QUESTIONS: tuple[Question, ...] = (
    _q(
        id="p_value.meaning",
        concept_id="inference.hypothesis_testing",
        kind=QuestionKind.MULTIPLE_CHOICE,
        misconception_id="p_is_prob_h0",
        prompt="A test returns p = 0.03. What does that number mean?",
        choices=(
            Choice(
                "If the null hypothesis were true, data at least this extreme would occur "
                "3% of the time.",
                correct=True,
                why="This is the definition: a probability computed under the null, about data.",
            ),
            Choice(
                "There is a 3% probability that the null hypothesis is true.",
                why="This reverses the conditioning. Getting P(H0 | data) requires a prior "
                    "and Bayes' rule, and the answer is usually far larger than 0.03.",
            ),
            Choice(
                "There is a 97% probability that the alternative hypothesis is true.",
                why="Same reversal, stated from the other side; the p-value says nothing "
                    "directly about the probability of either hypothesis.",
            ),
            Choice(
                "The effect is large enough to matter in practice.",
                why="The p-value mixes effect size with sample size and noise. A trivial "
                    "effect becomes significant with enough data.",
            ),
        ),
        explanation="Run the null scenario and watch the p-values spread out uniformly.",
        try_it=("inference.hypothesis_testing", "null_true"),
        level="beginner",
    ),
    _q(
        id="p_value.sample_size",
        concept_id="inference.hypothesis_testing",
        kind=QuestionKind.PREDICT_RESULT,
        misconception_id="p_measures_effect",
        prompt=(
            "You hold the true effect fixed at a tiny value and increase the sample size "
            "tenfold. What happens to the p-value?"
        ),
        choices=(
            Choice(
                "It tends to fall, because precision rises while the effect stays put.",
                correct=True,
                why="The standard error shrinks like 1/sqrt(n), so even a negligible effect "
                    "eventually clears any fixed threshold.",
            ),
            Choice(
                "It stays about the same, because the effect did not change.",
                why="The p-value depends on the effect *relative to its standard error*, and "
                    "the standard error changed.",
            ),
            Choice(
                "It rises, because more data means more noise.",
                why="More data means less noise in the estimate, not more.",
            ),
        ),
        explanation=(
            "This is why significance and importance must be reported separately."
        ),
        try_it=("inference.power", "large_sample"),
    ),
    _q(
        id="confidence_interval.meaning",
        concept_id="inference.confidence_intervals",
        kind=QuestionKind.MULTIPLE_CHOICE,
        misconception_id="ci_probability_statement",
        prompt="You compute a 95% confidence interval of [2.1, 4.7]. Which statement is right?",
        choices=(
            Choice(
                "95% of intervals built this way, across repeated samples, contain the true "
                "value.",
                correct=True,
                why="The confidence level describes the procedure, not this particular interval.",
            ),
            Choice(
                "There is a 95% probability that the parameter lies between 2.1 and 4.7.",
                why="Both numbers are now fixed and so is the parameter; no probability is "
                    "left. That reading needs a Bayesian credible interval.",
            ),
            Choice(
                "95% of the observations lie between 2.1 and 4.7.",
                why="That is a prediction interval for data, a different and much wider thing.",
            ),
            Choice(
                "The estimate is accurate to within 95%.",
                why="Accuracy and confidence level are unrelated quantities.",
            ),
        ),
        explanation="Draw many samples and count how many intervals cover the true value.",
        try_it=("inference.confidence_intervals", "coverage_check"),
        level="beginner",
    ),
    _q(
        id="clt.what_converges",
        concept_id="inference.clt",
        kind=QuestionKind.MULTIPLE_CHOICE,
        misconception_id="clt_makes_data_normal",
        prompt=(
            "You draw 5000 observations from a strongly right-skewed distribution. "
            "What does the central limit theorem promise?"
        ),
        choices=(
            Choice(
                "The sampling distribution of the mean is approximately normal.",
                correct=True,
                why="The theorem is about the standardised average across repeated samples.",
            ),
            Choice(
                "The histogram of the 5000 observations is approximately normal.",
                why="A large sample from a skewed distribution is a large skewed sample. The "
                    "data keep their shape.",
            ),
            Choice(
                "The median and the mean now coincide.",
                why="Skewness of the population is untouched by the sample size.",
            ),
        ),
        explanation="Compare the two histograms side by side in the lab.",
        try_it=("inference.clt", "skewed_population"),
        level="beginner",
    ),
    _q(
        id="power.observed",
        concept_id="inference.power",
        kind=QuestionKind.TRUE_FALSE,
        misconception_id="observed_power",
        prompt=(
            "True or false: after a non-significant result, computing power at the observed "
            "effect size tells you whether the study was large enough."
        ),
        choices=(
            Choice(
                "False - observed power is a function of the p-value and adds nothing.",
                correct=True,
                why="A p just above 0.05 always produces low observed power, whatever the "
                    "design, so the reasoning is circular.",
            ),
            Choice(
                "True - it shows how much power the study actually had.",
                why="It shows the p-value in different units. Power must be computed for the "
                    "smallest effect worth detecting, decided in advance.",
            ),
        ),
        explanation="Report the confidence interval instead; it shows what has been ruled out.",
        try_it=("inference.power", "underpowered"),
        level="advanced",
    ),
    _q(
        id="ols.orthogonality",
        concept_id="regression.ols_geometry",
        kind=QuestionKind.TRUE_FALSE,
        prompt=(
            "In a regression with an omitted confounder, is X'e = 0 still exactly zero in "
            "the fitted sample?"
        ),
        choices=(
            Choice(
                "Yes - it is an algebraic identity of least squares, whatever the model.",
                correct=True,
                why="Orthogonality follows from the normal equations alone, so it holds even "
                    "when the model is badly misspecified.",
            ),
            Choice(
                "No - endogeneity makes the residual correlate with the regressor.",
                why="That correlation is in the *population* error, which you never observe. "
                    "The fitted residual is orthogonal by construction.",
            ),
        ),
        explanation=(
            "This is exactly why X'e = 0 can never be used as evidence of exogeneity."
        ),
        try_it=("regression.ols_geometry", "canonical"),
        level="advanced",
    ),
    _q(
        id="r_squared.meaning",
        concept_id="regression.simple_linear",
        kind=QuestionKind.MULTIPLE_CHOICE,
        misconception_id="high_r2_means_correct",
        prompt="Two independent random walks are regressed on each other. What do you expect?",
        choices=(
            Choice(
                "A high R-squared and a significant slope, both meaningless.",
                correct=True,
                why="This is the spurious regression: shared trending behaviour, no relationship.",
            ),
            Choice(
                "An R-squared near zero, since the series are independent.",
                why="Independence in the innovations does not stop the levels from drifting "
                    "together over a finite sample.",
            ),
            Choice(
                "A significant slope only if the series really are related.",
                why="The usual t statistic does not have its usual distribution here, so it "
                    "rejects far too often.",
            ),
        ),
        explanation="Run the spurious scenario and read both numbers.",
        try_it=("timeseries.cointegration", "spurious"),
        level="advanced",
    ),
    _q(
        id="dag.collider",
        concept_id="causal.dag",
        kind=QuestionKind.IDENTIFY_ROLE,
        prompt=(
            "Treatment and outcome both cause a third variable. What happens if you control "
            "for that third variable?"
        ),
        choices=(
            Choice(
                "It creates an association where none existed - collider bias.",
                correct=True,
                why="Conditioning on a common effect makes its causes dependent given it.",
            ),
            Choice(
                "It removes confounding and improves the estimate.",
                why="A confounder is a common *cause*. This variable is a common effect, and "
                    "controlling for it does the opposite.",
            ),
            Choice(
                "It has no effect, since the variable is downstream.",
                why="Being downstream is precisely what makes conditioning on it harmful.",
            ),
        ),
        explanation="Adjust for the collider in the lab and watch a null association appear.",
        try_it=("causal.dag", "collider"),
        level="advanced",
    ),
    _q(
        id="did.parallel_trends",
        concept_id="causal.did",
        kind=QuestionKind.TRUE_FALSE,
        prompt=(
            "True or false: parallel pre-treatment trends prove the identifying assumption of "
            "difference-in-differences."
        ),
        choices=(
            Choice(
                "False - the assumption is about counterfactual post-treatment trends.",
                correct=True,
                why="What must be parallel is what *would have* happened after treatment, "
                    "which is unobservable. Pre-trends are suggestive evidence, not proof.",
            ),
            Choice(
                "True - if the trends match before, they would have matched after.",
                why="That is the assumption restated, not established. Nothing in the "
                    "pre-period rules out a divergence that begins with the treatment.",
            ),
        ),
        explanation="Run the violated-trends scenario, where the pre-trends look fine.",
        try_it=("causal.did", "violated_trends"),
        level="advanced",
    ),
    _q(
        id="iv.weak",
        concept_id="econometrics.endogeneity_iv",
        kind=QuestionKind.COMPARE_BEHAVIOUR,
        prompt="Your instrument is valid but weak (first-stage F of about 3). What follows?",
        choices=(
            Choice(
                "IV may be more biased than OLS, with badly wrong confidence intervals.",
                correct=True,
                why="Weak instruments push the estimator toward OLS and break the normal "
                    "approximation, so consistency in the limit offers no protection here.",
            ),
            Choice(
                "IV is still consistent, so it is the better choice.",
                why="Consistency is asymptotic. At any finite n a weak instrument can be far "
                    "worse than the OLS it was meant to repair.",
            ),
            Choice(
                "The estimate is unbiased but imprecise, so widen the interval.",
                why="It is biased, not merely imprecise, and the conventional interval does "
                    "not cover at its nominal rate.",
            ),
        ),
        explanation="Lower the instrument strength and compare the two estimators' spread.",
        try_it=("econometrics.endogeneity_iv", "weak_instrument"),
        level="advanced",
    ),
    _q(
        id="heteroskedasticity.what_breaks",
        concept_id="econometrics.heteroskedasticity",
        kind=QuestionKind.SELECT_VIOLATED_ASSUMPTION,
        prompt="Under heteroskedasticity, what actually goes wrong with OLS?",
        choices=(
            Choice(
                "The coefficients stay unbiased; the standard errors are wrong.",
                correct=True,
                why="Unbiasedness needs only the zero conditional mean. Constant variance "
                    "enters the variance formula, so inference is what breaks.",
            ),
            Choice(
                "The coefficients become biased.",
                why="Bias comes from a violated exogeneity assumption, not from a changing "
                    "error variance.",
            ),
            Choice(
                "Nothing important, since OLS is robust.",
                why="Test size can reach two or three times the nominal level, which is not a "
                    "small matter.",
            ),
        ),
        explanation=(
            "Compare classical and robust rejection rates under the null in the lab."
        ),
        try_it=("econometrics.heteroskedasticity", "severe"),
    ),
    _q(
        id="lasso.zeros",
        concept_id="ml.regularization",
        kind=QuestionKind.MULTIPLE_CHOICE,
        prompt="Why does the lasso set coefficients to exactly zero while ridge does not?",
        choices=(
            Choice(
                "The absolute value has a kink at zero, so its subgradient is an interval "
                "that can absorb the data's pull.",
                correct=True,
                why="Whenever the pull is weaker than lambda, zero satisfies the optimality "
                    "condition exactly.",
            ),
            Choice(
                "The lasso penalty is larger, so it shrinks more.",
                why="At a comparable amount of shrinkage ridge still produces no exact zeros. "
                    "The difference is the kink, not the size.",
            ),
            Choice(
                "The lasso discards variables whose t statistics are too small.",
                why="It performs no testing; the zeros come out of the optimisation itself.",
            ),
        ),
        explanation="Compare the two coefficient paths at the same penalty.",
        try_it=("ml.regularization", "sparse_truth"),
        level="advanced",
    ),
    _q(
        id="pca.variance",
        concept_id="multivariate.pca",
        kind=QuestionKind.TRUE_FALSE,
        prompt=(
            "True or false: the component that explains the most variance is the most useful "
            "for predicting an outcome."
        ),
        choices=(
            Choice(
                "False - PCA never looks at the outcome.",
                correct=True,
                why="A direction can carry most of the variance and none of the signal you "
                    "care about; the components are chosen without reference to y.",
            ),
            Choice(
                "True - more variance means more information.",
                why="More variance means more spread, which is not the same as more relevance.",
            ),
        ),
        explanation="Compare the components with a supervised method on the same data.",
        try_it=("multivariate.pca", "canonical"),
    ),
    _q(
        id="shap.causal",
        concept_id="xai.shap",
        kind=QuestionKind.TRUE_FALSE,
        prompt="True or false: a large SHAP value means the feature causes the outcome.",
        choices=(
            Choice(
                "False - it attributes the model's prediction, not the world's mechanism.",
                correct=True,
                why="SHAP explains what the model does with its inputs. A confounded feature "
                    "the model leans on gets a large attribution and no causal status.",
            ),
            Choice(
                "True - the attribution measures the feature's effect.",
                why="It measures the feature's effect on the *prediction*, which is a fact "
                    "about the model rather than about the data-generating process.",
            ),
        ),
        explanation="Run the correlated-features scenario and compare with the true model.",
        try_it=("xai.shap", "correlated_features"),
        level="advanced",
    ),
    _q(
        id="lln.gamblers",
        concept_id="inference.lln",
        kind=QuestionKind.MULTIPLE_CHOICE,
        misconception_id="gamblers_fallacy",
        prompt="A fair coin lands heads eight times running. What does the law of large numbers say about the next toss?",
        choices=(
            Choice(
                "Nothing - it is still 50/50; the imbalance is diluted, never repaid.",
                correct=True,
                why="The tosses are independent. Convergence of the proportion happens because "
                    "the denominator grows, not because the coin compensates.",
            ),
            Choice(
                "Tails is now more likely, to restore the balance.",
                why="This is the gambler's fallacy. The coin has no memory of the eight heads.",
            ),
            Choice(
                "Heads is more likely; the coin is evidently biased.",
                why="Eight in a row happens once in 256 fair sequences - surprising, but far "
                    "from evidence of bias on its own.",
            ),
        ),
        explanation=(
            "Watch the running proportion settle while the running count difference grows."
        ),
        try_it=("inference.lln", "long_run"),
        level="beginner",
    ),
)

_BY_CONCEPT: dict[str, list[Question]] = {}
for _question in QUESTIONS:
    _BY_CONCEPT.setdefault(_question.concept_id, []).append(_question)


def quiz_for(concept_id: str) -> tuple[Question, ...]:
    """Every question written for a concept."""
    return tuple(_BY_CONCEPT.get(concept_id, ()))


def build_quiz(
    concept_id: str | None = None,
    *,
    count: int = 5,
    seed: int = 42,
    level: str | None = None,
) -> Quiz:
    """Draw a reproducible quiz.

    Drawing is seeded, so the same seed gives the same questions in the same
    order with the same option ordering - a quiz can be shared as a link.
    """
    pool = list(quiz_for(concept_id)) if concept_id else list(QUESTIONS)
    if level:
        pool = [q for q in pool if q.level == level] or pool
    rng = random.Random(seed)
    rng.shuffle(pool)
    chosen = tuple(pool[: max(1, count)])
    return Quiz(
        questions=chosen,
        seed=seed,
        orders={q.id: q.shuffled(random.Random(seed + i)) for i, q in enumerate(chosen)},
    )


def score(quiz: Quiz, answers: dict[str, int]) -> dict[str, Any]:
    """Count correct answers, with the explanations and an explicit caveat.

    The caveat is part of the return value on purpose: this is a self-check,
    and a number without that context invites being read as a grade.
    """
    details = []
    correct = 0
    for question in quiz.questions:
        order = quiz.orders.get(question.id)
        chosen = answers.get(question.id, -1)
        ok, explanation = question.check(chosen, order)
        correct += ok
        details.append({
            "id": question.id,
            "correct": ok,
            "explanation": explanation,
            "misconception_id": question.misconception_id,
            "try_it": question.try_it,
        })
    return {
        "correct": correct,
        "total": len(quiz),
        "details": details,
        "caveat": (
            "This is a self-check, not an assessment. Nothing is recorded, and the count "
            "is not a grade - the explanations are the point."
        ),
        "misconceptions_to_revisit": [
            d["misconception_id"]
            for d in details
            if not d["correct"] and d["misconception_id"] in MISCONCEPTIONS
        ],
    }
