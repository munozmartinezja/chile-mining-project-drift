import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from cpdrift.regression import fit_firth
from cpdrift.verify import _verify_firth


def test_firth_is_finite_for_separable_data() -> None:
    x = np.column_stack([np.ones(8), np.arange(8, dtype=float)])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)

    result = fit_firth(x, y)

    assert np.isfinite(result.coef).all()
    assert np.isfinite(result.se).all()
    assert result.converged


def test_firth_is_close_to_mle_in_large_nonseparable_sample() -> None:
    rng = np.random.default_rng(20261022)
    feature = rng.normal(size=5000)
    probability = 1 / (1 + np.exp(-(-0.3 + 0.7 * feature)))
    y = rng.binomial(1, probability)
    x = sm.add_constant(pd.DataFrame({"x": feature})).to_numpy()

    firth = fit_firth(x, y)
    mle = sm.Logit(y, x).fit(disp=False)

    assert firth.coef[1] == pytest.approx(mle.params[1], abs=0.05)


def test_independent_firth_matches_model_implementation() -> None:
    rng = np.random.default_rng(44)
    feature = rng.normal(size=300)
    x = np.column_stack([np.ones(len(feature)), feature])
    probability = 1 / (1 + np.exp(-0.4 * feature))
    y = rng.binomial(1, probability)

    assert _verify_firth(x, y) == pytest.approx(fit_firth(x, y).coef, abs=1e-9)


def test_firth_rejects_rank_deficient_design() -> None:
    x = np.column_stack([np.ones(6), np.ones(6)])
    y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    with pytest.raises(ValueError, match="rango completo"):
        fit_firth(x, y)
