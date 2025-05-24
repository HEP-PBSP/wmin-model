"""
wmin.sr_normaliser.py

TODO

NOTE: currently supports only IC basis.
"""

import logging

import numpy as np
import tensorflow as tf
from colibri.constants import FLAVOUR_TO_ID_MAPPING

from torchquad import Boole, set_up_backend

log = logging.getLogger(__name__)

# TODO: test whether setting float64 has any effect on the accuracy of the results.
set_up_backend("jax", data_type="float64")  # Set up the backend for torchquad to use JAX

KNOWN_SUM_RULES_EV_BASIS = {
    "momentum": 1,
    "V": 3,
    "V3": 1,
    "V8": 3,
}
"""
Sum rules that are imposed in the intrinsic charm and evolution basis.
"""


LIMS = [(1e-9, 1e-8), (1e-8, 1e-7), (1e-7, 1e-5), (1e-5, 1e-3), (1e-3, 1)]
"""
Limits of the partial integration when computing (Sum) Rules.
"""


def pdf_integrand(n3fit_pdf_model):
    """
    TODO
    """

    def _integrand(x, rep_idx, sr_type):
        """
        Returns the PDF model for the given flavour.

        Parameters
        ----------
        x: float
            The x value in [0, 1].
        rep_idx: int
            The index of the replica to use.
        sr_type: str
            The type of sum rule to use. Can be "momentum", "V", "V3" or "V8".

        Returns
        -------
        np.array
            The PDF model for the given flavour.
        """
        # # x: shape (batch,1)  or (nx,1) depending on your sampler
        # x = jnp.squeeze(x, axis=-1)           # now shape (batch,) or (nx,)
        # x_in = x[None, :, None]               # shape (1, batch, 1)

        # # call your model in JAX form!
        # # It must accept JAX arrays and return a JAX array of shape (nreplicas, nflavours, batch)
        # pdf_grid = n3fit_pdf_model({
        #     "pdf_input":    x_in,
        #     "xgrid_integration": n3fit_pdf_model.x_in
        # })
        # # squeeze out that leading dim
        # pdf_grid = jnp.squeeze(pdf_grid, axis=0)   # (nreplicas, nflavours, batch)

        # # swap axes to (nreplicas, batch, nflavours) if that’s more convenient
        # pdf_array = jnp.transpose(pdf_grid, (0, 2, 1))


        x_in = tf.reshape(x, (1, max(x.shape), 1))
        input = {"pdf_input": x_in, "xgrid_integration": n3fit_pdf_model.x_in}

        pdf_grid = tf.squeeze(n3fit_pdf_model(input), axis=0)

        # shapes here are (nreplicas, nflavours, nx)
        pdf_array = np.array(tf.transpose(pdf_grid, perm=[0, 2, 1]))

        if sr_type == "momentum":
            x_gluon = pdf_array[rep_idx, FLAVOUR_TO_ID_MAPPING["g"], :]
            x_singlet = pdf_array[rep_idx, FLAVOUR_TO_ID_MAPPING[r"\Sigma"], :]
            return x_gluon + x_singlet

        else:
            x_valence_type = pdf_array[rep_idx, FLAVOUR_TO_ID_MAPPING[sr_type], :]
            return x_valence_type / x[:, 0]

    return _integrand


def _integral_torchquad(func, rep_idx, sr_type, lim, n_samples=500000):
    """
    TODO
    """

    def integrand(x):
        # torchquad calls integrand with shape (batch, dim)
        # here dim=1, so x.shape == (batch, 1)
        return func(x, rep_idx, sr_type)

    int_meth = (
        Boole()
    )  # Boole() #GaussLegendre() -> seems to be more accurate but also slower
    domain = [[lim[0], lim[1]]]

    log.info(f"[TorchQuad] Integrating {sr_type} for replica {rep_idx} in {lim}")
    res = int_meth.integrate(
        integrand, dim=1, N=n_samples, integration_domain=domain, backend="jax"
    )
    return res #float(res)


def _sum_rules(integrands, num_members, lims=LIMS):
    """
    Compute a sum rules grid for the an n3fit PDF model.

    Parameters
    ----------
    integrands: callable
        The integrand function to use.
    num_members: int
        The number of members in the PDF model.
    lims: list of tuples
        The limits of integration. Default is LIMS.

    Returns
    -------
    list of dict
        A list of dictionaries, where each dictionary contains the sum rules for a given limit.
        The keys are the sum rule types and the values are the integrals for each member.
    """
    return [
        {
            sr_type: [
                _integral_torchquad(integrands, rep_idx=i, sr_type=sr_type, lim=l)
                for i in range(num_members)
            ]
            for sr_type in KNOWN_SUM_RULES_EV_BASIS.keys()
        }
        for l in lims
    ]


def _combine_limits(res: list[dict]):
    """Sum the various limits together for all SR and return a dictionary."""
    return {k: np.sum([v[k] for v in res], axis=0) for k in res[0].keys()}


def sr_normalisation_factors(n3fit_pdf_model, lims=LIMS):
    """
    Returns the normalisation factors for the sum rules.
    """
    integrands = pdf_integrand(n3fit_pdf_model)
    num_members = n3fit_pdf_model.num_replicas

    # compute the integrals for all replicas for different limits
    sr = _sum_rules(integrands, num_members, lims)

    # combine the limits together for all SR and return a dictionary.
    combined_lims = _combine_limits(sr)

    return combined_lims


def sum_rules_normalise_pdf_array(pdf_array, sr_normalisation_factors):
    """
    Normalises the PDF array with the sum rules.

    Parameters
    ----------
    pdf_array: np.array
        The PDF array to normalise.
    sr_normalisation_factors: dict
        The normalisation factors for the sum rules.

    Returns
    -------
    np.array
        The normalised PDF array.
    """
    # impose momentum sum rules
    pdf_array[:, FLAVOUR_TO_ID_MAPPING["g"], :] /= sr_normalisation_factors["momentum"][
        :, None
    ]

    pdf_array[:, FLAVOUR_TO_ID_MAPPING[r"\Sigma"], :] /= sr_normalisation_factors[
        "momentum"
    ][:, None]

    # impose valence sum rules
    for flav in ["V", "V3", "V8"]:
        pdf_array[:, FLAVOUR_TO_ID_MAPPING[flav], :] *= (
            KNOWN_SUM_RULES_EV_BASIS[flav] / sr_normalisation_factors[flav][:, None]
        )

    # Impose Intrinsic charm basis relations
    pdf_array[:, FLAVOUR_TO_ID_MAPPING["V15"], :] = pdf_array[
        :, FLAVOUR_TO_ID_MAPPING["V"], :
    ]
    pdf_array[:, FLAVOUR_TO_ID_MAPPING["V24"], :] = pdf_array[
        :, FLAVOUR_TO_ID_MAPPING["V"], :
    ]
    pdf_array[:, FLAVOUR_TO_ID_MAPPING["V35"], :] = pdf_array[
        :, FLAVOUR_TO_ID_MAPPING["V"], :
    ]

    pdf_array[:, FLAVOUR_TO_ID_MAPPING["T24"], :] = pdf_array[
        :, FLAVOUR_TO_ID_MAPPING[r"\Sigma"], :
    ]
    pdf_array[:, FLAVOUR_TO_ID_MAPPING["T35"], :] = pdf_array[
        :, FLAVOUR_TO_ID_MAPPING[r"\Sigma"], :
    ]

    return pdf_array
