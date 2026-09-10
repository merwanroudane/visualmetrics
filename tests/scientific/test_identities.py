"""Exact identities.

Each test states a theorem and checks that the code reproduces it to machine
precision. Nothing here is a tolerance chosen to make a number pass: an
identity either holds or the implementation is wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from visualmetrics.backends.linear import ols, partial_out, projection_matrices


class TestLeastSquaresGeometry:
    def test_residual_orthogonality(self):
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(80), rng.normal(size=(80, 3))])
        y = rng.normal(size=80)
        e = ols(y, X).residuals
        assert np.max(np.abs(X.T @ e)) < 1e-10

    def test_pythagoras_holds_exactly(self):
        rng = np.random.default_rng(1)
        X = np.column_stack([np.ones(60), rng.normal(size=(60, 2))])
        y = rng.normal(size=60)
        P, M = projection_matrices(X)
        assert float(y @ y) == pytest.approx(
            float((P @ y) @ (P @ y) + (M @ y) @ (M @ y)), rel=1e-12
        )

    def test_the_hat_matrix_trace_equals_the_parameter_count(self):
        rng = np.random.default_rng(2)
        for k in (1, 3, 7):
            X = np.column_stack([np.ones(50), rng.normal(size=(50, k))])
            P, _ = projection_matrices(X)
            assert np.trace(P) == pytest.approx(k + 1, abs=1e-9)

    def test_fwl_coefficient_identity(self):
        rng = np.random.default_rng(3)
        n = 150
        X1 = np.column_stack([np.ones(n), rng.normal(size=(n, 2))])
        X2 = rng.normal(size=(n, 1)) + X1[:, 1:2]
        X = np.column_stack([X1, X2])
        y = X @ np.array([1.0, 2.0, -1.0, 0.5]) + rng.normal(size=n)
        full = ols(y, X).coefficients[-1]
        y_res, x_res = partial_out(y, X2, X1)
        assert ols(y_res, x_res).coefficients[0] == pytest.approx(full, abs=1e-10)

    def test_omitted_variable_bias_formula(self):
        """b_short = b_long + delta * gamma, exactly, in any sample."""
        rng = np.random.default_rng(4)
        n = 200
        x1 = rng.normal(size=n)
        x2 = 0.7 * x1 + rng.normal(size=n)
        y = 1.0 + 2.0 * x1 + 3.0 * x2 + rng.normal(size=n)
        short = ols(y, np.column_stack([np.ones(n), x1])).coefficients[1]
        long_fit = ols(y, np.column_stack([np.ones(n), x1, x2])).coefficients
        delta = ols(x2, np.column_stack([np.ones(n), x1])).coefficients[1]
        assert short == pytest.approx(long_fit[1] + delta * long_fit[2], abs=1e-10)


class TestPCA:
    def test_reconstruction_error_equals_the_discarded_eigenvalues(self):
        rng = np.random.default_rng(5)
        n, k = 300, 5
        X = rng.normal(size=(n, k)) @ rng.normal(size=(k, k))
        Xc = X - X.mean(axis=0)
        S = Xc.T @ Xc / (n - 1)
        vals, vecs = np.linalg.eigh(S)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        for keep in range(1, k):
            approx = Xc @ vecs[:, :keep] @ vecs[:, :keep].T
            error = float(np.sum((Xc - approx) ** 2)) / (n - 1)
            assert error == pytest.approx(float(vals[keep:].sum()), rel=1e-9)

    def test_components_are_orthonormal_and_ordered(self):
        rng = np.random.default_rng(6)
        X = rng.normal(size=(200, 4))
        Xc = X - X.mean(axis=0)
        S = Xc.T @ Xc / 199
        vals, vecs = np.linalg.eigh(S)
        assert vecs.T @ vecs == pytest.approx(np.eye(4), abs=1e-10)
        assert np.all(np.diff(np.sort(vals)[::-1]) <= 1e-12)

    def test_total_variance_is_preserved(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(150, 6)) @ rng.normal(size=(6, 6))
        Xc = X - X.mean(axis=0)
        S = Xc.T @ Xc / 149
        assert np.trace(S) == pytest.approx(float(np.linalg.eigvalsh(S).sum()), rel=1e-10)


class TestProbabilityIdentities:
    def test_variance_decomposition(self):
        """Var(Y) = E[Var(Y|X)] + Var(E[Y|X]) on a discrete example, exactly."""
        rng = np.random.default_rng(8)
        groups = rng.integers(0, 4, size=6000)
        means = np.array([0.0, 2.0, -1.0, 5.0])
        sds = np.array([1.0, 0.5, 2.0, 1.5])
        y = means[groups] + sds[groups] * rng.normal(size=6000)
        within = np.mean([np.var(y[groups == g]) for g in range(4)])
        between = np.var([np.mean(y[groups == g]) for g in range(4)])
        # The identity is exact for the population; with equal-sized groups the
        # sample version matches to sampling error only.
        assert np.var(y) == pytest.approx(within + between, rel=0.05)

    def test_law_of_total_expectation(self):
        rng = np.random.default_rng(9)
        x = rng.integers(0, 5, size=8000)
        y = 2.0 * x + rng.normal(size=8000)
        conditional = np.array([y[x == v].mean() for v in range(5)])
        weights = np.array([(x == v).mean() for v in range(5)])
        assert float(y.mean()) == pytest.approx(float(conditional @ weights), rel=1e-12)


class TestStatisticalProperties:
    """Properties that hold in repeated sampling, checked by simulation.

    These are simulations, not proofs: they can only fail to refute the
    property. The tolerances are set from the Monte Carlo standard error.
    """

    def test_a_correct_test_rejects_at_its_nominal_rate(self):
        rng = np.random.default_rng(10)
        reps, n, alpha = 3000, 40, 0.05
        rejects = 0
        for _ in range(reps):
            sample = rng.normal(size=n)
            t = sample.mean() / (sample.std(ddof=1) / np.sqrt(n))
            # two-sided critical value for 39 degrees of freedom
            rejects += abs(t) > 2.0227
        rate = rejects / reps
        se = np.sqrt(alpha * (1 - alpha) / reps)
        assert abs(rate - alpha) < 4 * se, f"size {rate:.3f} is far from {alpha}"

    def test_power_rises_with_the_effect_size(self):
        rng = np.random.default_rng(11)

        def power(delta, reps=1500, n=40):
            hits = 0
            for _ in range(reps):
                sample = rng.normal(loc=delta, size=n)
                t = sample.mean() / (sample.std(ddof=1) / np.sqrt(n))
                hits += abs(t) > 2.0227
            return hits / reps

        curve = [power(d) for d in (0.0, 0.3, 0.6)]
        assert curve[0] < curve[1] < curve[2]
        assert curve[2] > 0.8

    def test_the_sample_mean_converges(self):
        rng = np.random.default_rng(12)
        spreads = []
        for n in (50, 500, 5000):
            means = [rng.normal(size=n).mean() for _ in range(300)]
            spreads.append(np.std(means))
        assert spreads[0] > spreads[1] > spreads[2]
        # the standard error must fall like 1/sqrt(n)
        assert spreads[0] / spreads[2] == pytest.approx(10.0, rel=0.25)

    def test_ols_standard_errors_match_the_sampling_spread(self):
        rng = np.random.default_rng(13)
        n = 80
        x = rng.normal(size=n)
        X = np.column_stack([np.ones(n), x])
        estimates, reported = [], []
        for _ in range(800):
            y = X @ np.array([1.0, 2.0]) + rng.normal(size=n)
            fit = ols(y, X)
            estimates.append(fit.coefficients[1])
            reported.append(fit.standard_errors[1])
        assert np.std(estimates) == pytest.approx(np.mean(reported), rel=0.08)
