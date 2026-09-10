"""The numpy-native linear-model engine.

Everything here is checked against a closed form or an exact identity, never
against "the number the code produced last time".
"""

from __future__ import annotations

import numpy as np
import pytest

from visualmetrics.backends.linear import (
    cluster_cov,
    hac_cov,
    iv_2sls,
    ols,
    partial_out,
    projection_matrices,
    restricted_ols,
    vif,
    wls,
)


@pytest.fixture
def design():
    rng = np.random.default_rng(0)
    n = 200
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.normal(size=n)])
    beta = np.array([1.0, 2.0, -0.5])
    y = X @ beta + rng.normal(scale=1.0, size=n)
    return y, X, beta


class TestOLS:
    def test_matches_the_normal_equations(self, design):
        y, X, _ = design
        res = ols(y, X)
        closed_form = np.linalg.solve(X.T @ X, X.T @ y)
        assert res.coefficients == pytest.approx(closed_form, abs=1e-12)

    def test_residuals_are_orthogonal_to_the_regressors(self, design):
        y, X, _ = design
        res = ols(y, X)
        assert np.max(np.abs(X.T @ res.residuals)) < 1e-9

    def test_sums_of_squares_decompose_exactly(self, design):
        y, X, _ = design
        res = ols(y, X)
        tss = float(np.sum((y - y.mean()) ** 2))
        ess = float(np.sum((res.fitted_values - y.mean()) ** 2))
        rss = float(np.sum(res.residuals**2))
        assert tss == pytest.approx(ess + rss, rel=1e-10)
        assert res.r_squared == pytest.approx(1 - rss / tss, rel=1e-10)

    def test_recovers_the_truth_on_average(self):
        """Unbiasedness is a property of the estimator, checked by simulation."""
        rng = np.random.default_rng(1)
        beta = np.array([1.0, 2.0])
        estimates = []
        for _ in range(400):
            x = rng.normal(size=60)
            X = np.column_stack([np.ones(60), x])
            y = X @ beta + rng.normal(scale=2.0, size=60)
            estimates.append(ols(y, X).coefficients)
        mean = np.mean(estimates, axis=0)
        assert mean == pytest.approx(beta, abs=0.06)

    @pytest.mark.parametrize("cov_type", ["nonrobust", "HC0", "HC1", "HC2", "HC3"])
    def test_every_covariance_estimator_is_positive_definite(self, design, cov_type):
        y, X, _ = design
        res = ols(y, X, cov_type=cov_type)
        eig = np.linalg.eigvalsh((res.covariance + res.covariance.T) / 2)
        assert eig.min() > 0
        assert np.all(res.standard_errors > 0)

    def test_hc_standard_errors_exceed_classical_under_heteroskedasticity(self):
        rng = np.random.default_rng(2)
        n = 400
        x = rng.normal(size=n)
        X = np.column_stack([np.ones(n), x])
        y = X @ np.array([0.0, 1.0]) + rng.normal(scale=np.exp(x), size=n)
        classical = ols(y, X).standard_errors[1]
        robust = ols(y, X, cov_type="HC1").standard_errors[1]
        assert robust > classical


class TestWLS:
    def test_weighting_reproduces_ols_when_weights_are_equal(self, design):
        y, X, _ = design
        w = np.ones(len(y))
        assert wls(y, X, w).coefficients == pytest.approx(ols(y, X).coefficients, abs=1e-10)

    def test_gls_beats_ols_under_known_heteroskedasticity(self):
        """With the correct weights the sampling variance must be smaller."""
        rng = np.random.default_rng(3)
        n = 120
        x = np.linspace(-2, 2, n)
        X = np.column_stack([np.ones(n), x])
        sigma = np.exp(0.8 * x)
        ols_est, wls_est = [], []
        for _ in range(400):
            y = X @ np.array([0.0, 1.0]) + rng.normal(scale=sigma)
            ols_est.append(ols(y, X).coefficients[1])
            wls_est.append(wls(y, X, 1.0 / sigma**2).coefficients[1])
        assert np.var(wls_est) < np.var(ols_est)


class TestRestricted:
    def test_imposing_a_true_restriction_barely_moves_the_fit(self, design):
        y, X, beta = design
        R = np.array([[0.0, 1.0, 0.0]])
        r = np.array([2.0])
        res, test = restricted_ols(y, X, R, r)
        assert res.coefficients[1] == pytest.approx(2.0, abs=1e-12)
        assert test.p_value > 0.01

    def test_imposing_a_false_restriction_is_rejected(self, design):
        y, X, _ = design
        R = np.array([[0.0, 1.0, 0.0]])
        res, test = restricted_ols(y, X, R, np.array([-5.0]))
        assert test.p_value < 1e-6
        assert float(np.sum(res.residuals**2)) > float(np.sum(ols(y, X).residuals**2))


class TestProjection:
    def test_p_and_m_are_complementary_projections(self, design):
        _, X, _ = design
        P, M = projection_matrices(X)
        n = X.shape[0]
        assert P @ P == pytest.approx(P, abs=1e-9)
        assert M @ M == pytest.approx(M, abs=1e-9)
        assert P + M == pytest.approx(np.eye(n), abs=1e-9)
        assert P @ M == pytest.approx(np.zeros((n, n)), abs=1e-9)
        assert np.trace(P) == pytest.approx(X.shape[1], abs=1e-9)

    def test_partial_out_reproduces_the_multiple_regression_slope(self, design):
        y, X, _ = design
        full = ols(y, X).coefficients[2]
        y_res, x_res = partial_out(y, X[:, [2]], X[:, [0, 1]])
        short = ols(y_res, x_res).coefficients[0]
        assert short == pytest.approx(full, abs=1e-9)


class TestDiagnostics:
    def test_vif_rises_with_correlation(self):
        rng = np.random.default_rng(4)
        n = 300
        x1 = rng.normal(size=n)
        independent = np.column_stack([np.ones(n), x1, rng.normal(size=n)])
        collinear = np.column_stack([np.ones(n), x1, 0.98 * x1 + 0.02 * rng.normal(size=n)])
        assert vif(independent)[1] < 1.2
        assert vif(collinear)[1] > 10

    def test_hac_widens_standard_errors_under_autocorrelation(self):
        rng = np.random.default_rng(5)
        n = 300
        u = np.zeros(n)
        for t in range(1, n):
            u[t] = 0.8 * u[t - 1] + rng.normal()
        x = np.arange(n) / n
        X = np.column_stack([np.ones(n), x])
        y = X @ np.array([0.0, 1.0]) + u
        plain = ols(y, X)
        XtX_inv = np.linalg.inv(X.T @ X)
        cov = hac_cov(X, plain.residuals, XtX_inv, maxlags=6)
        assert np.sqrt(cov[1, 1]) > plain.standard_errors[1]

    def test_cluster_covariance_reflects_within_group_correlation(self):
        rng = np.random.default_rng(6)
        g, m = 30, 20
        groups = np.repeat(np.arange(g), m)
        shock = np.repeat(rng.normal(scale=2.0, size=g), m)
        x = np.repeat(rng.normal(size=g), m)
        X = np.column_stack([np.ones(g * m), x])
        y = X @ np.array([0.0, 1.0]) + shock + rng.normal(size=g * m)
        plain = ols(y, X)
        cov = cluster_cov(X, plain.residuals, np.linalg.inv(X.T @ X), groups)
        assert np.sqrt(cov[1, 1]) > plain.standard_errors[1]


class TestIV:
    def test_iv_removes_the_endogeneity_bias(self):
        rng = np.random.default_rng(8)
        n = 2000
        z = rng.normal(size=n)
        u = rng.normal(size=n)
        x = 0.8 * z + 0.9 * u + rng.normal(scale=0.3, size=n)
        y = 1.0 + 2.0 * x + u
        X = np.column_stack([np.ones(n), x])
        Z = np.column_stack([np.ones(n), z])
        biased = ols(y, X).coefficients[1]
        res = iv_2sls(y, X, Z, endog_index=1)
        assert abs(biased - 2.0) > 0.3
        assert res.second_stage.coefficients[1] == pytest.approx(2.0, abs=0.15)
        assert res.first_stage_f > 50
        assert not res.weak_instruments

    def test_a_weak_instrument_is_reported_as_weak(self):
        rng = np.random.default_rng(9)
        n = 400
        z = rng.normal(size=n)
        u = rng.normal(size=n)
        x = 0.02 * z + 0.9 * u + rng.normal(size=n)
        y = 1.0 + 2.0 * x + u
        res = iv_2sls(
            y,
            np.column_stack([np.ones(n), x]),
            np.column_stack([np.ones(n), z]),
            endog_index=1,
        )
        assert res.first_stage_f < 10
        assert res.weak_instruments
