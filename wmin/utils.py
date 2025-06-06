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

# from colibri.ultranest_fit import UltraNestLogLikelihood
from colibri.constants import FLAVOUR_TO_ID_MAPPING, LHAPDF_XGRID
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
        "smallx": [-0.4506, 0.75],  # 0.9305],
        "largex": [1.745, 3.424],
    },
    {
        "fl": "t8",
        "trainable": False,
        "smallx": [0.5877, 0.75],  # 0.8687],
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


def sign_flips_counter(f: np.ndarray) -> int:
    """
    Counts the number of sign flips in the function values `f`.
    Sign flips are counted by checking the sign of the numerical derivative.

    Parameters
    ----------
    f: array-like, should represent x * pdf
        The function values for which to count sign flips.
    """

    # Ensure xgrid is LHAPDF_XGRID to correctly discard points
    if f.shape[-1] != len(LHAPDF_XGRID):
        raise ValueError("The function values should be defined on the LHAPDF_XGRID.")

    # count oscillations by taking numerical derivative, discard the last two points
    signs_v = np.sign(np.diff(f[:-25]))

    tmp_sign = signs_v[0]
    n_sign_flips = 0

    for sign in signs_v:
        if sign == -tmp_sign:
            n_sign_flips += 1
            tmp_sign = sign

    return n_sign_flips


def sign_flip_selection(pdf_array: np.ndarray) -> np.ndarray:
    """
    Given a PDF array, filters out the replicas for which the V, V3, and V8 flavours
    don't have exactly one sign flip.

    NOTE: valence flavours that oscillate more than once can lead to non-integrable
    basis functions when evolving the PDFs.

    Parameters
    ----------
    pdf_array: np.ndarray
        The PDF array of shape (N_rep, N_fl, N_x).

    Returns
    -------
    np.ndarray
        The filtered PDF array containing only replicas with exactly one sign flip
        for V, V3, and V8 flavours.
    """

    one_flips = []
    for rep in range(pdf_array.shape[0]):
        f_v = pdf_array[rep, FLAVOUR_TO_ID_MAPPING["V"]]
        f_v8 = pdf_array[rep, FLAVOUR_TO_ID_MAPPING["V8"]]
        f_v3 = pdf_array[rep, FLAVOUR_TO_ID_MAPPING["V3"]]
        f_t3 = pdf_array[rep, FLAVOUR_TO_ID_MAPPING["T3"]]
        f_t8 = pdf_array[rep, FLAVOUR_TO_ID_MAPPING["T8"]]

        n_flips_v = sign_flips_counter(f_v)
        n_flips_v8 = sign_flips_counter(f_v8)
        n_flips_v3 = sign_flips_counter(f_v3)
        n_flips_t3 = sign_flips_counter(f_t3)
        n_flips_t8 = sign_flips_counter(f_t8)

        # keep the replica only if all V, V3 and V8 have n_flips == 1
        if (
            n_flips_v <= 1
            and n_flips_v8 <= 1
            and n_flips_v3 <= 1
            and n_flips_t3 <= 1
            and n_flips_t8 <= 1
        ):
            one_flips.append(rep)
    log.info(
        f"Found {len(one_flips)} replicas with exactly one sign flip for V, V3, V8, T3 and T8 flavours."
    )
    return pdf_array[one_flips, :, :]
