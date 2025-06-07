"""
wmin.ultranest_fit.py

This module overrides the log likelihood defined in colibri.ultranest_fit so as to allow
the user to add model dependent terms to the likelihood

"""

import jax
import jax.numpy as jnp
from functools import partial
from colibri.likelihood import LogLikelihood
from colibri.loss_functions import chi2
from wmin.utils import wmin_l1_penalty, wmin_l2_penalty


class WminUltraNestLogLikelihood(LogLikelihood):
    """
    UltraNest log likelihood with additional terms for the wmin model.
    """

    def __init__(
        self,
        central_inv_covmat_index,
        pdf_model,
        fit_xgrid,
        forward_map,
        fast_kernel_arrays,
        positivity_fast_kernel_arrays,
        ns_settings,
        chi2,
        penalty_posdata,
        positivity_penalty_settings,
        integrability_penalty,
        wmin_regularisation_settings={},
    ):
        super().__init__(
            central_inv_covmat_index,
            pdf_model,
            fit_xgrid,
            forward_map,
            fast_kernel_arrays,
            positivity_fast_kernel_arrays,
            ns_settings,
            chi2,
            penalty_posdata,
            positivity_penalty_settings,
            integrability_penalty,
        )
        self.wmin_regularisation_settings = wmin_regularisation_settings

        if self.wmin_regularisation_settings:
            self.lambda_factor = self.wmin_regularisation_settings["lambda_factor"]

            if self.wmin_regularisation_settings["type"] == "l2_reg":
                self.regularisation_penalty = wmin_l2_penalty

            elif self.wmin_regularisation_settings["type"] == "l1_reg":
                self.regularisation_penalty = wmin_l1_penalty

    @partial(jax.jit, static_argnames=("self",))
    def log_likelihood(
        self,
        params: jnp.array,
        central_values: jnp.array,
        inv_covmat: jnp.array,
        fast_kernel_arrays: tuple,
        positivity_fast_kernel_arrays: tuple,
    ) -> jnp.array:
        """
        This function takes care of computing the log_likelihood that is defined in LogLikelihood.
        Function is jax.jit compiled for better performance.

        Parameters
        ----------
        params: jnp.array
        central_values: jnp.array
        inv_covmat: jnp.array
        fast_kernel_arrays: tuple
        positivity_fast_kernel_arrays: tuple

        Returns
        -------
        jnp.array
            jax array with the value of the log-likelihood.
        """
        predictions, pdf = self.pred_and_pdf(params, fast_kernel_arrays)

        if self.wmin_regularisation_settings:
            regularisation_term = self.regularisation_penalty(
                params, self.lambda_factor
            )
        else:
            regularisation_term = 0.0

        if self.positivity_penalty_settings["positivity_penalty"]:
            pos_penalty = jnp.sum(
                self.penalty_posdata(
                    pdf,
                    self.positivity_penalty_settings["alpha"],
                    self.positivity_penalty_settings["lambda_positivity"],
                    positivity_fast_kernel_arrays,
                ),
                axis=-1,
            )
        else:
            pos_penalty = 0

        integ_penalty = jnp.sum(
            self.integrability_penalty(
                pdf,
            ),
            axis=-1,
        )

        return -0.5 * (
            self.chi2(central_values, predictions, inv_covmat)
            + pos_penalty
            + integ_penalty
            + regularisation_term
        )


def log_likelihood(
    central_inv_covmat_index,
    pdf_model,
    FIT_XGRID,
    _pred_data,
    fast_kernel_arrays,
    positivity_fast_kernel_arrays,
    ns_settings,
    _penalty_posdata,
    positivity_penalty_settings,
    integrability_penalty,
    wmin_regularisation_settings={},
):
    """
    Overriding the log_likelihood function from colibri.ultranest_fit
    """
    return WminUltraNestLogLikelihood(
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
        integrability_penalty,
        wmin_regularisation_settings,
    )
