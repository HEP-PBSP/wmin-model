"""
wmin.utils is a module that contains several utils for PDF fits in
the wmin parameterization.
"""

import logging
import resource
import time

import jax
import numpy as np
import pandas as pd
from colibri.loss_functions import chi2

from colibri.ultranest_fit import UltraNestLogLikelihood
from reportengine.table import table

log = logging.getLogger(__name__)


FLAV_INFO = [
    {
        "fl": "sng",
        "trainable": False,
        "smallx": [1.089, 1.119],
        "largex": [1.475, 3.119],
    },
    {
        "fl": "g",
        "trainable": False,
        "smallx": [0.7504, 1.098],
        "largex": [2.814, 5.669],
    },
    {
        "fl": "v",
        "trainable": False,
        "smallx": [0.479, 0.7384],
        "largex": [1.549, 3.532],
    },
    {
        "fl": "v3",
        "trainable": False,
        "smallx": [0.1073, 0.4397],
        "largex": [1.733, 3.458],
    },
    {
        "fl": "v8",
        "trainable": False,
        "smallx": [0.5507, 0.7837],
        "largex": [1.516, 3.356],
    },
    {
        "fl": "t3",
        "trainable": False,
        "smallx": [-0.4506, 0.9305],
        "largex": [1.745, 3.424],
    },
    {
        "fl": "t8",
        "trainable": False,
        "smallx": [0.5877, 0.8687],
        "largex": [1.522, 3.515],
    },
    {
        "fl": "t15",
        "trainable": False,
        "smallx": [1.089, 1.141],
        "largex": [1.492, 3.222],
    },
]
"""
Default preprocessing exponents ranges for the NNPDF40 parametrisation.
"""

RSS_MB = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
"""
Memory usage before loading of resources
"""
init = RSS_MB()


@table
def likelihood_time(
    _penalty_posdata,
    central_inv_covmat_index,
    fast_kernel_arrays,
    positivity_fast_kernel_arrays,
    _pred_data,
    FIT_XGRID,
    pdf_model,
    bayesian_prior,
    theoryid,
    ns_settings,
    n_prior_samples=1000,
    positivity_penalty_settings={},
):
    """
    This function calculates the time it takes to evaluate the likelihood
    function.

    Parameters
    ----------
    _penalty_posdata: function
        The positivity penalty function to use.

    central_inv_covmat_index: colibri.commondata_utils.CentralInvCovmatIndex

    fast_kernel_arrays: tuple
        The FK tables to use.

    positivity_fast_kernel_arrays: tuple
        The POS FK tables to use.

    _pred_data: function
        The prediction function to use.

    FIT_XGRID: array
        The xgrid to use.

    pdf_model: PDFModel

    bayesian_prior: function
        The prior function to use.

    theoryid: str

    ns_settings: dict

    n_prior_samples: int, 1000

    alpha: float, 1e-7

    lambda_positivity: int, 1000

    Returns
    -------
    df: pd.DataFrame
        The DataFrame containing the results.
    """
    log.info(f"Memory usage after loading of resources")
    res = RSS_MB()
    log.info(f"RSS: {res - init:.2f}MB")

    central_values = central_inv_covmat_index.central_values
    ndata = len(central_values)

    log_likelihood = UltraNestLogLikelihood(
        central_inv_covmat_index,
        pdf_model,
        FIT_XGRID,
        _pred_data,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        ns_settings,
        chi2,
        _penalty_posdata,
        positivity_penalty_settings,
    )

    # sample from prior
    rng = jax.random.PRNGKey(0)
    prior_samples = []
    for i in range(n_prior_samples):
        prior_samples.append(
            bayesian_prior(jax.random.uniform(rng, shape=(pdf_model.n_basis,)))
        )

    # compile likelihood
    log_likelihood(prior_samples[0])

    log.info(f"Memory usage after initialization of log_likelihood")
    log.info(f"RSS: {RSS_MB() - res:.2f}MB")
    # evaluate likelihood time
    start_time = time.perf_counter()
    for i in range(n_prior_samples):
        log_likelihood(prior_samples[i])
    end_time = time.perf_counter()

    time_per_eval = (end_time - start_time) / n_prior_samples

    df = pd.DataFrame(
        {
            "Ndata": [ndata],
            "Theory": [theoryid],
            "Likelihood eval time (s)": [time_per_eval],
        },
        index=["wmin"],
    )
    return df


def arclength_pdfgrid(xgrid: np.array, pdf_grid: np.array) -> np.array:
    """
    Calculate the arclength of the PDF grid.

    Parameters
    ----------
    xgrid: np.array
        array of shape (N_x, )
    pdf_grid: np.array
        array of shape (N_rep, N_fl, N_x)

    Returns
    -------
    np.array
        array of shape (N_rep, )
    """
    dx = np.diff(xgrid)
    dpdf = np.diff(pdf_grid, axis=-1)
    return np.sum(np.sum(np.sqrt(dx**2 + dpdf**2), axis=-1), axis=-1)


def arclength_outliers(arclength_pdfgrid: np.array) -> np.array:
    """
    Find the outliers in the arclength of the PDF grid using the interquartile range.

    Parameters
    ----------
    arclength_pdfgrid: np.array
        array of shape (N_rep, )

    Returns
    -------
    np.array
        array of shape (N_outliers, )
    """
    # Identify outlier replicas based on arclength and using interquartile range
    Q1 = np.percentile(arclength_pdfgrid, 25)
    Q3 = np.percentile(arclength_pdfgrid, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = np.where(
        (arclength_pdfgrid < lower_bound) | (arclength_pdfgrid > upper_bound)
    )[0]
    return outliers
