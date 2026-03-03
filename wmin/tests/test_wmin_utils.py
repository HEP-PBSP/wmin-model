"""
Tests for the wmin.utils module.
"""

from unittest.mock import Mock

import jax.numpy as jnp
import pandas as pd
from wmin.utils import likelihood_time
from colibri.tests.conftest import TEST_XGRID

N_MOCK_DATA = 100
MOCK_NAME_THEORY = "test_theory"

mock_penalty_posdata = (
    lambda pdf, alpha, lambda_positivity, fast_kernel_arrays: jnp.array([0])
)
central_covmat_index = Mock()
central_covmat_index.central_values = jnp.ones(N_MOCK_DATA)
central_covmat_index.covmat = jnp.eye(N_MOCK_DATA)
central_covmat_index.central_values_idx = jnp.arange(N_MOCK_DATA)

fast_kernel_arrays = ((jnp.ones(N_MOCK_DATA)),)
positivity_fast_kernel_arrays = ((jnp.ones(N_MOCK_DATA)),)

mock_pred_data = lambda pdf, fast_kernel_arrays: jnp.ones(N_MOCK_DATA)


NS_SETTINGS = {
    "n_posterior_samples": 10,
    "ReactiveNS_settings": {
        "vectorized": False,
        "ndraw_max": 500,
    },
    "Run_settings": {
        "min_num_live_points": 500,
        "min_ess": 50,
        "frac_remain": 0.01,
    },
}


class MockPDFModel:
    n_basis = N_MOCK_DATA

    @staticmethod
    def pred_and_pdf_func(FIT_XGRID, forward_map):
        def pred_and_pdf(params, fast_kernel_arrays):
            predictions = params * 2
            pdf = params * 3
            return predictions, pdf

        return pred_and_pdf


def mock_bayesian_prior(rng):
    return rng * 0.5


SETUP = {
    "_penalty_posdata": mock_penalty_posdata,
    "central_covmat_index": central_covmat_index,
    "fast_kernel_arrays": fast_kernel_arrays,
    "positivity_fast_kernel_arrays": positivity_fast_kernel_arrays,
    "_pred_data": mock_pred_data,
    "FIT_XGRID": TEST_XGRID,
    "pdf_model": MockPDFModel(),
    "bayesian_prior": mock_bayesian_prior,
    "theoryid": MOCK_NAME_THEORY,
    "n_prior_samples": 100,
    "positivity_penalty_settings": {
        "positivity_penalty": False,
        "alpha": 1e-7,
        "lambda_positivity": 1000,
    },
}


def test_likelihood_time_structure():
    """
    test the structure of the output of likelihood_time
    """
    # Debug: Check what likelihood_time actually is
    print(f"likelihood_time type: {type(likelihood_time)}")
    print(f"likelihood_time: {likelihood_time}")
    print(f"likelihood_time module: {likelihood_time.__module__}")

    result = likelihood_time(**SETUP)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["Ndata", "Theory", "Likelihood eval time (s)"]
    assert result.index[0] == "wmin"


def test_likelihood_time_values():
    """
    test the values of the output of likelihood_time
    """

    result = likelihood_time(**SETUP)
    assert result.loc["wmin", "Ndata"] == N_MOCK_DATA
    assert result.loc["wmin", "Theory"] == MOCK_NAME_THEORY
    assert isinstance(result.loc["wmin", "Likelihood eval time (s)"], float)


def test_likelihood_time_samples():
    """
    test the number of samples in the output of likelihood_time
    """

    SETUP["n_prior_samples"] = 5
    result = likelihood_time(**SETUP)
    assert len(result) == 1
    assert result.loc["wmin", "Ndata"] == N_MOCK_DATA
    assert result.loc["wmin", "Theory"] == MOCK_NAME_THEORY
    assert isinstance(result.loc["wmin", "Likelihood eval time (s)"], float)
    assert result.loc["wmin", "Likelihood eval time (s)"] > 0
